from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import re
import numpy as np
import torch
from loci.baselines.base import extract_answer
from loci.probe import CommitmentProbe
from loci.commitment import CommitmentGapTracker, CommitmentBreakingPolicy
from loci.hooks import HiddenStateCapture, MultiLayerHiddenStateCapture, AttentionHeadAblator, ablate_many, make_ablator, DirectionalSteeringHook, ResidualPatcher, RetrievalDonorPatcher, probe_to_raw_direction
from transformers import StoppingCriteria, StoppingCriteriaList

class _BatchEarlyStopOnAnswer(StoppingCriteria):
    __slots__ = ('tok', 'prompt_end', 'markers', 'check_every', 'decode_window', '_step', '_hit', '_batch_size')

    def __init__(self, tokenizer, prompt_end: int, batch_size: int, check_every: int=64, decode_window: int=48):
        self.tok = tokenizer
        self.prompt_end = prompt_end
        self.markers = ('the answer is', '\\boxed{', 'Final Answer')
        self.check_every = check_every
        self.decode_window = decode_window
        self._step = 0
        self._batch_size = batch_size
        self._hit = [False] * batch_size

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        self._step += 1
        if self._step % self.check_every != 0:
            return False
        gen_len = input_ids.shape[1] - self.prompt_end
        if gen_len < self.check_every:
            return False
        window = input_ids[:, -self.decode_window:]
        texts = self.tok.batch_decode(window, skip_special_tokens=True)
        for i in range(self._batch_size):
            if self._hit[i]:
                continue
            if any((m in texts[i] for m in self.markers)):
                self._hit[i] = True
        return all(self._hit)

class LOCIReasoner:
    name = 'loci'

    def __init__(self, model, tokenizer, max_new_tokens: int=512, temperature: float=0.6, is_multiple_choice: bool=False, max_steps: int=8, commitment_probe: Optional[CommitmentProbe]=None, commitment_layer: Union[int, Sequence[int]]=-1, commitment_head_spec: Optional[Sequence[Tuple[int, int]]]=None, commitment_target_residual: float=0.1, commitment_max_window: int=6, capture_hidden_states: bool=False, intervention_mode: str='weight_patch', steering_alpha: float=1.0, steering_layer: Optional[Union[int, str, Sequence[int]]]=None, residual_donor: Optional[Dict[int, np.ndarray]]=None, retrieval_data: Optional[Dict[str, Any]]=None, retrieval_K: int=8, retrieval_alpha: float=0.5, retrieval_metric: str='cosine', enable_self_reflection: bool=False, reflection_cue: str='Wait — I should reconsider. The previous step may have committed too quickly. Let me try a different approach:', reflection_trigger: str='probe'):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.is_multiple_choice = is_multiple_choice
        self.max_steps = max_steps
        self.commitment_probe = commitment_probe
        if isinstance(commitment_layer, (list, tuple)):
            self.commitment_layer_list: Optional[List[int]] = [int(x) for x in commitment_layer]
            if not self.commitment_layer_list:
                raise ValueError('commitment_layer list must be non-empty')
            self.commitment_layer = self.commitment_layer_list[0]
        else:
            self.commitment_layer_list = None
            self.commitment_layer = int(commitment_layer)
        self._layer_signed_to_resolved: Dict[int, int] = {}
        self.capture_hidden_states = bool(capture_hidden_states)
        self.commit_tracker: Optional[CommitmentGapTracker] = None
        self.commit_policy: Optional[CommitmentBreakingPolicy] = None
        self.commit_hiddens: List[np.ndarray] = []
        self.commit_hiddens_by_layer: Dict[int, List[np.ndarray]] = {}
        if commitment_probe is not None:
            self.commit_tracker = CommitmentGapTracker(commitment_probe)
            self.commit_policy = CommitmentBreakingPolicy(head_spec=commitment_head_spec or [], target_residual=commitment_target_residual, max_window=commitment_max_window, detect_threshold=commitment_probe.commit_threshold)
        self._ablator_cache: dict = {}
        if intervention_mode not in ('weight_patch', 'directional_steering', 'residual_patch', 'retrieval_patch'):
            raise ValueError(f'intervention_mode={intervention_mode!r} not recognised')
        self.intervention_mode = intervention_mode
        self._residual_donor = residual_donor or {}
        if intervention_mode == 'residual_patch' and (not self._residual_donor):
            raise ValueError('intervention_mode=residual_patch requires residual_donor (Dict[layer_idx, ndarray]); got empty dict')
        self._retrieval_data = retrieval_data or {}
        self._retrieval_K = int(retrieval_K)
        self._retrieval_alpha = float(retrieval_alpha)
        self._retrieval_metric = str(retrieval_metric)
        if intervention_mode == 'retrieval_patch' and (not self._retrieval_data):
            raise ValueError('intervention_mode=retrieval_patch requires retrieval_data (dict of per-layer key banks); got empty dict')
        self.steering_alpha = float(steering_alpha)
        if steering_layer is None:
            self._steering_layers_override: Optional[List[int]] = None
        elif isinstance(steering_layer, (list, tuple)):
            self._steering_layers_override = [int(x) for x in steering_layer]
            if not self._steering_layers_override:
                self._steering_layers_override = None
        elif isinstance(steering_layer, str):
            parts = [p.strip() for p in steering_layer.split(',') if p.strip()]
            if not parts:
                self._steering_layers_override = None
            else:
                self._steering_layers_override = [int(p) for p in parts]
        else:
            self._steering_layers_override = [int(steering_layer)]
        self._w_raw: Optional[np.ndarray] = None
        if intervention_mode == 'directional_steering' and commitment_probe is not None:
            self._w_raw = probe_to_raw_direction(commitment_probe)
        self.enable_self_reflection = bool(enable_self_reflection)
        self.reflection_cue = str(reflection_cue)
        if reflection_trigger not in ('probe', 'always', 'never'):
            raise ValueError(f'reflection_trigger={reflection_trigger!r} not recognised')
        self.reflection_trigger = reflection_trigger

    def _reset_state(self) -> None:
        if self.commit_tracker is not None:
            self.commit_tracker.reset()
        if self.commit_policy is not None:
            self.commit_policy.reset()
        self.commit_hiddens = []
        self.commit_hiddens_by_layer = {}

    def _commitment_enabled(self) -> bool:
        return self.commitment_probe is not None and self.commit_tracker is not None and (self.commit_policy is not None)

    def _hidden_capture_enabled(self) -> bool:
        return self._commitment_enabled() or self.capture_hidden_states

    def _resolve_layer_idx(self) -> int:
        layer_idx = self.commitment_layer
        base = getattr(self.model, 'model', self.model)
        if hasattr(base, 'layers'):
            n_layers = len(base.layers)
        elif hasattr(base, 'decoder') and hasattr(base.decoder, 'layers'):
            n_layers = len(base.decoder.layers)
        else:
            raise AttributeError(f'cannot resolve commitment_layer={self.commitment_layer} on {type(self.model).__name__}: no `model.layers` / `model.decoder.layers` found.')
        if layer_idx < 0:
            layer_idx = n_layers + layer_idx
        if not 0 <= layer_idx < n_layers:
            raise ValueError(f'commitment_layer={self.commitment_layer} resolves to index {layer_idx}, out of range for {n_layers}-layer model.')
        return layer_idx

    def _get_ablators(self, head_spec: Sequence[Tuple[int, int]]):
        if self.intervention_mode == 'residual_patch':
            if not head_spec:
                return []
            layers = self._steering_layers_override
            if layers is None:
                layers = [self._resolve_layer_idx()]
            layers = [int(li) for li in layers]
            missing = [li for li in layers if li not in self._residual_donor]
            if missing:
                raise RuntimeError(f'residual_patch: donor missing for layers {missing}; have {sorted(self._residual_donor.keys())}')
            key = ('residual_patch', tuple(layers))
            cached = self._ablator_cache.get(key)
            if cached is not None:
                return cached
            hooks = [ResidualPatcher(self.model, layer_idx=li, donor=self._residual_donor[li]) for li in layers]
            self._ablator_cache[key] = hooks
            return hooks
        if self.intervention_mode == 'retrieval_patch':
            if not head_spec:
                return []
            layers = self._steering_layers_override
            if layers is None:
                layers = [self._resolve_layer_idx()]
            layers = [int(li) for li in layers]
            missing = [li for li in layers if f'keys_layer_{li}' not in self._retrieval_data]
            if missing:
                raise RuntimeError(f"retrieval_patch: bank missing layers {missing}; have keys_layer_* for {[k for k in self._retrieval_data if k.startswith('keys_layer_')]}")
            key = ('retrieval_patch', tuple(layers), self._retrieval_K, round(self._retrieval_alpha, 6), self._retrieval_metric)
            cached = self._ablator_cache.get(key)
            if cached is not None:
                return cached
            hooks = []
            for li in layers:
                keys_arr = np.asarray(self._retrieval_data[f'keys_layer_{li}'])
                keys_unit_arr = np.asarray(self._retrieval_data.get(f'keys_unit_layer_{li}', keys_arr))
                hooks.append(RetrievalDonorPatcher(self.model, layer_idx=li, keys=keys_arr, keys_unit=keys_unit_arr, donor_values=keys_arr, K=self._retrieval_K, alpha=self._retrieval_alpha, metric=self._retrieval_metric))
            self._ablator_cache[key] = hooks
            return hooks
        if self.intervention_mode == 'directional_steering':
            if not head_spec:
                return []
            if self._w_raw is None:
                raise RuntimeError('directional_steering requested but no probe attached; cannot derive commitment direction')
            layers = self._steering_layers_override
            if layers is None:
                layers = [self._resolve_layer_idx()]
            layers = [int(li) for li in layers]
            key = ('directional', tuple(layers), round(float(self.steering_alpha), 6))
            cached = self._ablator_cache.get(key)
            if cached is not None:
                return cached
            hooks = [DirectionalSteeringHook(self.model, layer_idx=li, w_raw=self._w_raw, alpha=self.steering_alpha) for li in layers]
            self._ablator_cache[key] = hooks
            return hooks
        key = tuple(sorted(((int(l), int(h)) for l, h in head_spec)))
        cached = self._ablator_cache.get(key)
        if cached is not None:
            return cached
        per_layer: dict = {}
        for li, hi in head_spec:
            per_layer.setdefault(int(li), []).append(int(hi))
        ablators = [make_ablator(self.model, li, hs) for li, hs in per_layer.items()]
        self._ablator_cache[key] = ablators
        return ablators

    def _generate_step_with_capture(self, prompt: str, capture: Optional['HiddenStateCapture'], ablate_heads: Optional[List[Tuple[int, int]]]=None):
        if ablate_heads:
            from contextlib import ExitStack
            ablators = self._get_ablators(ablate_heads)
            with ExitStack() as stack:
                for a in ablators:
                    stack.enter_context(a)
                text, logp, ent, nt, n_in = self._generate_step(prompt)
        else:
            text, logp, ent, nt, n_in = self._generate_step(prompt)
        if capture is None:
            hidden_vec = np.zeros(1, dtype=np.float64)
        else:
            hs = capture.last_state
            if hs is None or hs.size == 0:
                hidden_vec = np.zeros(1, dtype=np.float64)
            else:
                hidden_vec = hs.flatten().astype(np.float64)
        return (text, logp, ent, nt, n_in, hidden_vec)

    def _generate_step(self, prompt: str):
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096).to(self.model.device)
        input_len = int(inputs.input_ids.shape[1])
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=self.temperature, do_sample=True, return_dict_in_generate=True, output_scores=True, pad_token_id=pad_id)
        generated_ids = out.sequences[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        _step_m = re.search('(?:^|\\n)Step\\d+', text)
        if _step_m:
            text = text[:_step_m.start()]
        n = min(len(out.scores), len(generated_ids))
        if n > 0:
            scores = torch.stack([s[0] for s in out.scores[:n]], dim=0)
            probs = torch.softmax(scores, dim=-1)
            vocab = probs.shape[-1]
            ids = generated_ids[:n].to(probs.device)
            valid = ids < vocab
            safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
            chosen = probs.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1)
            chosen = torch.where(valid, chosen, torch.full_like(chosen, 1e-10))
            chosen = torch.clamp(chosen, 1e-10, 1.0)
            logp_sum = torch.log(chosen).sum()
            mask = probs > 1e-06
            safe_probs = torch.where(mask, probs, torch.ones_like(probs))
            ent_per_pos = -(probs * torch.log(safe_probs) * mask.to(probs.dtype)).sum(dim=-1)
            ent_sum = ent_per_pos.sum()
            total_logp = float(logp_sum.item())
            total_entropy = float(ent_sum.item())
        else:
            total_logp = 0.0
            total_entropy = 0.0
        avg_logp = total_logp / max(n, 1)
        avg_entropy = total_entropy / max(n, 1)
        return (text.strip(), avg_logp, avg_entropy, n, input_len)

    def _build_prompt(self, question: str, traj: List[str], step_idx: int) -> str:
        messages = [{'role': 'user', 'content': f"Q: {question}\nSolve step by step. Write your final answer enclosed in \\boxed{{}}. End your response with exactly one line of the form: The answer is \\boxed{{<your final answer>}}. Start your solution with 'Step{step_idx}:'"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if step_idx > 0:
            prompt += '\n'.join(traj) + f'\nStep{step_idx}:'
        else:
            prompt += 'Step0:'
        return prompt

    def solve(self, question: str) -> Dict[str, Any]:
        self._reset_state()
        traj: List[str] = []
        total_tokens = 0
        total_input_tokens = 0
        total_calls = 0
        got_answer = False
        step_entropies: List[float] = []
        step_logps: List[float] = []
        step_input_tokens: List[int] = []
        step_output_tokens: List[int] = []
        commit_scores: List[float] = []
        commit_decisions: List[Dict[str, Any]] = []
        pending_ablation: List[Tuple[int, int]] = []
        ablation_steps_left: int = 0
        original_window_k: int = 0
        capture_enabled = self._hidden_capture_enabled()
        capture: Optional['HiddenStateCapture'] = None
        multi_capture: Optional['MultiLayerHiddenStateCapture'] = None
        if capture_enabled:
            if self.commitment_layer_list is not None:
                multi_capture = MultiLayerHiddenStateCapture(self.model, layer_indices=self.commitment_layer_list)
                multi_capture.__enter__()
                self._layer_signed_to_resolved = {}
                for signed, resolved in zip(self.commitment_layer_list, multi_capture.layer_indices):
                    self._layer_signed_to_resolved[signed] = resolved
                layer_idx = self._resolve_layer_idx()
                capture = multi_capture._captures[layer_idx]
                for signed in self.commitment_layer_list:
                    self.commit_hiddens_by_layer[signed] = []
            else:
                layer_idx = self._resolve_layer_idx()
                capture = HiddenStateCapture(self.model, layer_idx=layer_idx)
                capture.__enter__()
        try:
            for step_idx in range(self.max_steps):
                prompt = self._build_prompt(question, traj, step_idx)
                if capture_enabled:
                    step_was_ablated = ablation_steps_left > 0
                    active_heads = pending_ablation if step_was_ablated else None
                    text, logp, ent, nt, n_in, h_vec = self._generate_step_with_capture(prompt, capture, ablate_heads=active_heads)
                    if ablation_steps_left > 0:
                        ablation_steps_left -= 1
                else:
                    step_was_ablated = False
                    text, logp, ent, nt, n_in = self._generate_step(prompt)
                    h_vec = None
                total_tokens += nt
                total_input_tokens += n_in
                total_calls += 1
                if not text:
                    break
                traj.append(f'Step{step_idx}: {text}')
                step_entropies.append(ent)
                step_logps.append(logp)
                step_input_tokens.append(int(n_in))
                step_output_tokens.append(int(nt))
                if h_vec is not None:
                    self.commit_hiddens.append(np.asarray(h_vec, dtype=np.float32).copy())
                    if multi_capture is not None:
                        per_layer = multi_capture.last_state
                        for signed, resolved in self._layer_signed_to_resolved.items():
                            arr = per_layer.get(resolved)
                            if arr is None or arr.size == 0:
                                continue
                            vec = arr.flatten().astype(np.float32).copy()
                            self.commit_hiddens_by_layer.setdefault(signed, []).append(vec)
                if self._commitment_enabled() and h_vec is not None:
                    commit_score = self.commit_tracker.record_step(step_idx, h_vec)
                    self.commit_policy.observe(commit_score, was_ablated=step_was_ablated)
                    commit_scores.append(commit_score)
                    if ablation_steps_left == 0 and (not step_was_ablated):
                        decision = self.commit_policy.decide(commit_score, probe_threshold=self.commitment_probe.commit_threshold)
                        commit_decisions.append({'step': step_idx, 'score': float(commit_score), 'intervene': bool(decision.intervene), 'window_k': int(decision.window_k), 'predicted_residual': float(decision.predicted_residual)})
                        if decision.intervene and decision.head_spec:
                            pending_ablation = list(decision.head_spec)
                            ablation_steps_left = int(decision.window_k)
                            original_window_k = int(decision.window_k)
                    else:
                        commit_decisions.append({'step': step_idx, 'score': float(commit_score), 'intervene': bool(step_was_ablated), 'window_k': int(original_window_k), 'window_remaining': int(ablation_steps_left), 'predicted_residual': float(commit_score * self.commit_policy.spectral_radius)})
                low = text.lower()
                if 'the answer is' in low or '\\boxed{' in low:
                    got_answer = True
                    break
            commitment_summary: Dict[str, Any] = {}
            if self._commitment_enabled():
                if got_answer:
                    tau_output = step_idx if traj else 0
                else:
                    tau_output = len(traj) - 1 if traj else 0
                self.commit_tracker.finalize(tau_output=tau_output, was_error=False)
                trace = self.commit_tracker.trace
                commitment_summary = {'tau_commit': trace.tau_commit, 'tau_output': trace.tau_output, 'step_scores': list(trace.step_scores), 'n_triggered': int(self.commit_policy.n_triggered), 'n_skipped': int(self.commit_policy.n_skipped), 'spectral_radius': float(self.commit_policy.spectral_radius), 'decisions': commit_decisions}
            full_text = '\n'.join(traj)
            answer = extract_answer(full_text, self.is_multiple_choice)
            if step_logps:
                logps_arr = np.asarray(step_logps, dtype=np.float64)
                weights = np.asarray(step_output_tokens, dtype=np.float64)
                if weights.sum() > 0:
                    mean_logp = float(np.sum(logps_arr * weights) / weights.sum())
                else:
                    mean_logp = float(np.mean(logps_arr))
                confidence = float(np.exp(mean_logp))
            else:
                confidence = 0.5
            confidence = float(np.clip(confidence, 0.0, 1.0))
            mean_entropy = float(np.mean(step_entropies)) if step_entropies else float('nan')
            max_entropy = float(np.max(step_entropies)) if step_entropies else float('nan')
            return {'answer': answer, 'trajectory': full_text, 'n_tokens': total_tokens, 'n_input_tokens': total_input_tokens, 'n_calls': total_calls, 'raw_outputs': list(traj), 'got_answer': got_answer, 'confidence': confidence, 'mean_entropy': mean_entropy, 'max_entropy': max_entropy, 'step_entropies': list(step_entropies), 'step_logps': list(step_logps), 'step_input_tokens': list(step_input_tokens), 'step_output_tokens': list(step_output_tokens), 'commitment': commitment_summary}
        finally:
            if multi_capture is not None:
                multi_capture.__exit__(None, None, None)
            elif capture is not None:
                capture.__exit__(None, None, None)

    def _generate_batch_phaseA(self, prompts: List[str], *, early_stop: bool=False):
        if not prompts:
            return ([], [], [], [], [])
        orig_side = getattr(self.tokenizer, 'padding_side', 'right')
        self.tokenizer.padding_side = 'left'
        try:
            enc = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=4096)
        finally:
            self.tokenizer.padding_side = orig_side
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        B, L_in = input_ids.shape
        n_in_list = attention_mask.sum(dim=1).tolist()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        stopping = None
        if early_stop:
            stopping = StoppingCriteriaList([_BatchEarlyStopOnAnswer(self.tokenizer, prompt_end=L_in, batch_size=B, check_every=64, decode_window=48)])
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens, temperature=max(self.temperature, 1e-05), do_sample=self.temperature > 0, pad_token_id=pad_id, return_dict_in_generate=True, output_scores=True, stopping_criteria=stopping)
        gen_ids = out.sequences[:, L_in:]
        L_gen = gen_ids.shape[1]
        per_row_real = torch.ones(B, L_gen, dtype=torch.bool, device=gen_ids.device)
        for b in range(B):
            row = gen_ids[b]
            end = L_gen
            if eos_id is not None:
                eos_pos = (row == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    end = min(end, int(eos_pos[0, 0].item()) + 1)
            if pad_id is not None and pad_id != eos_id:
                pad_pos = (row == pad_id).nonzero(as_tuple=False)
                if pad_pos.numel() > 0:
                    end = min(end, int(pad_pos[0, 0].item()))
            per_row_real[b, end:] = False
        if L_gen > 0 and len(out.scores) > 0:
            vocab = out.scores[0].shape[-1]
            safe_ids = gen_ids.clone()
            safe_ids = torch.where(safe_ids < vocab, safe_ids, torch.zeros_like(safe_ids))
            chosen_lp = torch.zeros(B, L_gen, device=gen_ids.device, dtype=torch.float32)
            ent_per_pos = torch.zeros(B, L_gen, device=gen_ids.device, dtype=torch.float32)
            scores_list = list(out.scores)
            out.scores = None
            for t in range(L_gen):
                step = scores_list[t]
                scores_list[t] = None
                lp = torch.log_softmax(step.float(), dim=-1)
                lp = torch.where(torch.isfinite(lp), lp, torch.zeros_like(lp))
                chosen_lp[:, t] = lp.gather(-1, safe_ids[:, t:t + 1]).squeeze(-1)
                ent_per_pos[:, t] = -(lp.exp() * lp).sum(dim=-1)
                del step, lp
            del scores_list, safe_ids
        else:
            chosen_lp = torch.zeros(B, 0, device=out.sequences.device)
            ent_per_pos = torch.zeros(B, 0, device=out.sequences.device)
        texts: List[str] = []
        avg_logps: List[float] = []
        avg_ents: List[float] = []
        n_outs: List[int] = []
        for b in range(B):
            n_real = int(per_row_real[b].sum().item())
            if n_real > 0:
                row_ids = gen_ids[b, :n_real]
                text = self.tokenizer.decode(row_ids, skip_special_tokens=True)
                m = re.search('(?:^|\\n)Step\\d+', text)
                if m:
                    text = text[:m.start()]
                rm = per_row_real[b]
                sum_lp = float(chosen_lp[b][rm].sum().item())
                sum_ent = float(ent_per_pos[b][rm].sum().item())
                avg_lp = sum_lp / n_real
                avg_ent = sum_ent / n_real
            else:
                text = ''
                avg_lp = 0.0
                avg_ent = 0.0
            texts.append(text.strip())
            avg_logps.append(avg_lp)
            avg_ents.append(avg_ent)
            n_outs.append(n_real)
        return (texts, avg_logps, avg_ents, n_outs, n_in_list)

    def solve_phaseA_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        if self.commitment_probe is not None:
            raise RuntimeError('solve_phaseA_batch() is a probe=None fast path; attach no commitment_probe for Phase A.')
        B = len(questions)
        if B == 0:
            return []
        trajs: List[List[str]] = [[] for _ in range(B)]
        step_logps: List[List[float]] = [[] for _ in range(B)]
        step_entropies: List[List[float]] = [[] for _ in range(B)]
        step_out_tokens: List[List[int]] = [[] for _ in range(B)]
        step_in_tokens: List[List[int]] = [[] for _ in range(B)]
        hiddens: List[List[np.ndarray]] = [[] for _ in range(B)]
        hiddens_by_layer_list: List[Dict[int, List[np.ndarray]]] = [{} for _ in range(B)]
        done = [False] * B
        got_answer = [False] * B
        capture: Optional[HiddenStateCapture] = None
        multi_capture: Optional[MultiLayerHiddenStateCapture] = None
        if self.commitment_layer_list is not None:
            multi_capture = MultiLayerHiddenStateCapture(self.model, layer_indices=self.commitment_layer_list)
            multi_capture.__enter__()
            self._layer_signed_to_resolved = {signed: resolved for signed, resolved in zip(self.commitment_layer_list, multi_capture.layer_indices)}
            layer_idx = self._resolve_layer_idx()
            capture = multi_capture._captures[layer_idx]
        else:
            layer_idx = self._resolve_layer_idx()
            capture = HiddenStateCapture(self.model, layer_idx=layer_idx)
            capture.__enter__()
        try:
            for step_idx in range(self.max_steps):
                active = [i for i in range(B) if not done[i]]
                if not active:
                    break
                prompts = [self._build_prompt(questions[i], trajs[i], step_idx) for i in active]
                texts, logps, ents, n_outs, n_ins = self._generate_batch_phaseA(prompts)
                per_example_h = capture.last_state_batch if capture is not None else None
                per_layer_batch = multi_capture.last_state_batch if multi_capture is not None else None
                for pos, i in enumerate(active):
                    text = texts[pos]
                    n_out = n_outs[pos]
                    if n_out == 0 and (not text):
                        done[i] = True
                        continue
                    trajs[i].append(f'Step{step_idx}: {text}')
                    step_logps[i].append(float(logps[pos]))
                    step_entropies[i].append(float(ents[pos]))
                    step_out_tokens[i].append(int(n_out))
                    step_in_tokens[i].append(int(n_ins[pos]))
                    if per_example_h is not None and pos < per_example_h.shape[0]:
                        h_vec = per_example_h[pos].astype(np.float32).copy()
                        hiddens[i].append(h_vec)
                    if per_layer_batch:
                        for signed, resolved in self._layer_signed_to_resolved.items():
                            arr = per_layer_batch.get(resolved)
                            if arr is None or pos >= arr.shape[0]:
                                continue
                            vec = arr[pos].astype(np.float32).copy()
                            hiddens_by_layer_list[i].setdefault(signed, []).append(vec)
                    low = text.lower()
                    if 'the answer is' in low or '\\boxed{' in low:
                        got_answer[i] = True
                        done[i] = True
        finally:
            if multi_capture is not None:
                multi_capture.__exit__(None, None, None)
            elif capture is not None:
                capture.__exit__(None, None, None)
        results: List[Dict[str, Any]] = []
        for i in range(B):
            if step_logps[i]:
                lp_arr = np.asarray(step_logps[i], dtype=np.float64)
                w = np.asarray(step_out_tokens[i], dtype=np.float64)
                if w.sum() > 0:
                    mean_lp = float(np.sum(lp_arr * w) / w.sum())
                else:
                    mean_lp = float(np.mean(lp_arr))
                confidence = float(np.clip(np.exp(mean_lp), 0.0, 1.0))
            else:
                confidence = 0.5
            full_text = '\n'.join(trajs[i])
            answer = extract_answer(full_text, self.is_multiple_choice)
            results.append({'answer': answer, 'trajectory': full_text, 'raw_outputs': list(trajs[i]), 'got_answer': got_answer[i], 'confidence': confidence, 'step_entropies': list(step_entropies[i]), 'step_logps': list(step_logps[i]), 'step_input_tokens': list(step_in_tokens[i]), 'step_output_tokens': list(step_out_tokens[i]), 'commit_hiddens': list(hiddens[i]), 'commit_hiddens_by_layer': {k: list(v) for k, v in hiddens_by_layer_list[i].items()}})
        return results

    def solve_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        B = len(questions)
        if B == 0:
            return []
        trajs: List[List[str]] = [[] for _ in range(B)]
        step_entropies: List[List[float]] = [[] for _ in range(B)]
        step_logps: List[List[float]] = [[] for _ in range(B)]
        step_input_tokens: List[List[int]] = [[] for _ in range(B)]
        step_output_tokens: List[List[int]] = [[] for _ in range(B)]
        total_tokens = [0] * B
        total_input_tokens = [0] * B
        total_calls = [0] * B
        done = [False] * B
        got_answer = [False] * B
        final_step = [0] * B
        commit_enabled = self._commitment_enabled()
        if commit_enabled:
            trackers = [CommitmentGapTracker(self.commitment_probe) for _ in range(B)]
            policies = [CommitmentBreakingPolicy(head_spec=self.commit_policy.head_spec, target_residual=self.commit_policy.target_residual, min_window=self.commit_policy.min_window, max_window=self.commit_policy.max_window, detect_threshold=self.commit_policy.detect_threshold) for _ in range(B)]
        else:
            trackers = [None] * B
            policies = [None] * B
        commit_scores: List[List[float]] = [[] for _ in range(B)]
        commit_decisions: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
        pending_ablation: List[List[Tuple[int, int]]] = [[] for _ in range(B)]
        ablation_left: List[int] = [0] * B
        original_window_k: List[int] = [0] * B
        capture_enabled = self._hidden_capture_enabled()
        capture: Optional[HiddenStateCapture] = None
        multi_capture: Optional[MultiLayerHiddenStateCapture] = None
        if capture_enabled:
            if self.commitment_layer_list is not None:
                multi_capture = MultiLayerHiddenStateCapture(self.model, layer_indices=self.commitment_layer_list)
                multi_capture.__enter__()
                self._layer_signed_to_resolved = {signed: resolved for signed, resolved in zip(self.commitment_layer_list, multi_capture.layer_indices)}
                layer_idx = self._resolve_layer_idx()
                capture = multi_capture._captures[layer_idx]
            else:
                layer_idx = self._resolve_layer_idx()
                capture = HiddenStateCapture(self.model, layer_idx=layer_idx)
                capture.__enter__()
        hiddens_by_layer_list: List[Dict[int, List[np.ndarray]]] = [{} for _ in range(B)]
        reflection_pending: List[bool] = [False] * B
        reflection_done: List[bool] = [False] * B
        try:
            for step_idx in range(self.max_steps):
                active = [i for i in range(B) if not done[i]]
                if not active:
                    break
                if self.enable_self_reflection:
                    for i in active:
                        if reflection_pending[i] and (not reflection_done[i]):
                            trajs[i].append(self.reflection_cue)
                            reflection_done[i] = True
                            reflection_pending[i] = False
                ablated_rows = [i for i in active if ablation_left[i] > 0]
                clean_rows = [i for i in active if ablation_left[i] == 0]
                sub_results: Dict[int, Tuple[str, float, float, int, int, Optional[np.ndarray]]] = {}
                if clean_rows:
                    prompts = [self._build_prompt(questions[i], trajs[i], step_idx) for i in clean_rows]
                    texts, logps, ents, n_outs, n_ins = self._generate_batch_phaseA(prompts, early_stop=True)
                    h_batch = capture.last_state_batch if capture is not None else None
                    per_layer_batch = multi_capture.last_state_batch if multi_capture is not None else None
                    for pos, i in enumerate(clean_rows):
                        h_vec = None
                        if h_batch is not None and pos < h_batch.shape[0]:
                            h_vec = h_batch[pos].astype(np.float32).copy()
                        if per_layer_batch:
                            for signed, resolved in self._layer_signed_to_resolved.items():
                                arr = per_layer_batch.get(resolved)
                                if arr is None or pos >= arr.shape[0]:
                                    continue
                                vec = arr[pos].astype(np.float32).copy()
                                hiddens_by_layer_list[i].setdefault(signed, []).append(vec)
                        sub_results[i] = (texts[pos], logps[pos], ents[pos], n_outs[pos], n_ins[pos], h_vec)
                if ablated_rows:
                    prompts = [self._build_prompt(questions[i], trajs[i], step_idx) for i in ablated_rows]
                    head_spec_shared = pending_ablation[ablated_rows[0]]
                    from contextlib import ExitStack
                    ablators = self._get_ablators(head_spec_shared)
                    with ExitStack() as stack:
                        for a in ablators:
                            stack.enter_context(a)
                        texts, logps, ents, n_outs, n_ins = self._generate_batch_phaseA(prompts, early_stop=True)
                        h_batch = capture.last_state_batch if capture is not None else None
                        per_layer_batch = multi_capture.last_state_batch if multi_capture is not None else None
                        if h_batch is not None:
                            h_batch = h_batch.copy()
                        if per_layer_batch:
                            per_layer_batch = {k: v.copy() for k, v in per_layer_batch.items()}
                    for pos, i in enumerate(ablated_rows):
                        h_vec = None
                        if h_batch is not None and pos < h_batch.shape[0]:
                            h_vec = h_batch[pos].astype(np.float32).copy()
                        if per_layer_batch:
                            for signed, resolved in self._layer_signed_to_resolved.items():
                                arr = per_layer_batch.get(resolved)
                                if arr is None or pos >= arr.shape[0]:
                                    continue
                                vec = arr[pos].astype(np.float32).copy()
                                hiddens_by_layer_list[i].setdefault(signed, []).append(vec)
                        sub_results[i] = (texts[pos], logps[pos], ents[pos], n_outs[pos], n_ins[pos], h_vec)
                for i in active:
                    text, logp, ent, nt, n_in, h_vec = sub_results[i]
                    step_was_ablated = i in ablated_rows
                    if ablation_left[i] > 0:
                        ablation_left[i] -= 1
                    total_tokens[i] += nt
                    total_input_tokens[i] += n_in
                    total_calls[i] += 1
                    if not text and nt == 0:
                        done[i] = True
                        continue
                    trajs[i].append(f'Step{step_idx}: {text}')
                    step_entropies[i].append(float(ent))
                    step_logps[i].append(float(logp))
                    step_input_tokens[i].append(int(n_in))
                    step_output_tokens[i].append(int(nt))
                    final_step[i] = step_idx
                    if commit_enabled and h_vec is not None:
                        score = trackers[i].record_step(step_idx, h_vec)
                        policies[i].observe(score, was_ablated=step_was_ablated)
                        commit_scores[i].append(score)
                        if ablation_left[i] == 0 and (not step_was_ablated):
                            decision = policies[i].decide(score, probe_threshold=self.commitment_probe.commit_threshold)
                            commit_decisions[i].append({'step': step_idx, 'score': float(score), 'intervene': bool(decision.intervene), 'window_k': int(decision.window_k), 'predicted_residual': float(decision.predicted_residual)})
                            if decision.intervene and decision.head_spec:
                                pending_ablation[i] = list(decision.head_spec)
                                ablation_left[i] = int(decision.window_k)
                                original_window_k[i] = int(decision.window_k)
                        else:
                            commit_decisions[i].append({'step': step_idx, 'score': float(score), 'intervene': bool(step_was_ablated), 'window_k': int(original_window_k[i]), 'window_remaining': int(ablation_left[i]), 'predicted_residual': float(score * policies[i].spectral_radius)})
                        if self.enable_self_reflection and self.reflection_trigger == 'probe' and (not reflection_done[i]) and (not reflection_pending[i]) and (score > self.commitment_probe.commit_threshold):
                            reflection_pending[i] = True
                    if self.enable_self_reflection and self.reflection_trigger == 'always' and (step_idx == 0) and (not reflection_done[i]) and (not reflection_pending[i]):
                        reflection_pending[i] = True
                    low = text.lower()
                    if 'the answer is' in low or '\\boxed{' in low:
                        got_answer[i] = True
                        done[i] = True
        finally:
            if multi_capture is not None:
                multi_capture.__exit__(None, None, None)
            elif capture is not None:
                capture.__exit__(None, None, None)
        results: List[Dict[str, Any]] = []
        for i in range(B):
            commitment_summary: Dict[str, Any] = {}
            if commit_enabled:
                if got_answer[i]:
                    tau_output = final_step[i] if trajs[i] else 0
                else:
                    tau_output = len(trajs[i]) - 1 if trajs[i] else 0
                trackers[i].finalize(tau_output=tau_output, was_error=False)
                trace = trackers[i].trace
                commitment_summary = {'tau_commit': trace.tau_commit, 'tau_output': trace.tau_output, 'step_scores': list(trace.step_scores), 'n_triggered': int(policies[i].n_triggered), 'n_skipped': int(policies[i].n_skipped), 'spectral_radius': float(policies[i].spectral_radius), 'decisions': commit_decisions[i]}
            full_text = '\n'.join(trajs[i])
            answer = extract_answer(full_text, self.is_multiple_choice)
            if step_logps[i]:
                lp_arr = np.asarray(step_logps[i], dtype=np.float64)
                w = np.asarray(step_output_tokens[i], dtype=np.float64)
                if w.sum() > 0:
                    mean_lp = float(np.sum(lp_arr * w) / w.sum())
                else:
                    mean_lp = float(np.mean(lp_arr))
                confidence = float(np.clip(np.exp(mean_lp), 0.0, 1.0))
            else:
                confidence = 0.5
            mean_entropy = float(np.mean(step_entropies[i])) if step_entropies[i] else float('nan')
            max_entropy = float(np.max(step_entropies[i])) if step_entropies[i] else float('nan')
            results.append({'answer': answer, 'trajectory': full_text, 'n_tokens': int(total_tokens[i]), 'n_input_tokens': int(total_input_tokens[i]), 'n_calls': int(total_calls[i]), 'raw_outputs': list(trajs[i]), 'got_answer': bool(got_answer[i]), 'confidence': confidence, 'mean_entropy': mean_entropy, 'max_entropy': max_entropy, 'step_entropies': list(step_entropies[i]), 'step_logps': list(step_logps[i]), 'step_input_tokens': list(step_input_tokens[i]), 'step_output_tokens': list(step_output_tokens[i]), 'commitment': commitment_summary, 'commit_hiddens_by_layer': {k: list(v) for k, v in hiddens_by_layer_list[i].items()}, 'reflection_injected': bool(reflection_done[i])})
        return results
