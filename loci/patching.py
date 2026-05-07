from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence
import numpy as np
import torch
from loci.hooks import AttentionHeadAblator, make_ablator

@dataclass
class PatchTrajectory:
    question: str
    prefix_text: str
    gold: str
    is_multiple_choice: bool
    original_answer: str
    n_remaining_steps: Optional[int] = None
    tau_commit: Optional[int] = None
    meta: dict = field(default_factory=dict)

class HeadPatcher:

    def __init__(self, model, tokenizer, trajectories: Sequence[PatchTrajectory], max_new_tokens: int=256, max_replay_steps: int=4, grader: Optional[Callable[[str, str, bool], bool]]=None, answer_extractor: Optional[Callable[[str, bool], str]]=None):
        self.model = model
        self.tokenizer = tokenizer
        self.trajectories: List[PatchTrajectory] = list(trajectories)
        self.max_new_tokens = int(max_new_tokens)
        self.max_replay_steps = int(max_replay_steps)
        if grader is None:
            from loci.metrics import grade as _grade
            grader = _grade
        if answer_extractor is None:
            from loci.baselines.base import extract_answer as _extract
            answer_extractor = _extract
        self.grader = grader
        self.answer_extractor = answer_extractor
        self._clean_cache: dict[int, str] = {}
        self._prompt_cache: dict[int, dict] = {}
        self._clean_correct: dict[int, bool] = {}
        _pad = self.tokenizer.pad_token_id
        self._pad_token_id = _pad if _pad is not None else self.tokenizer.eos_token_id
        self._ablator_cache: dict[tuple, 'AttentionHeadAblator'] = {}

    def _build_continuation_prompt(self, traj: PatchTrajectory) -> str:
        start_step = int(traj.tau_commit or 0) + 1
        messages = [{'role': 'user', 'content': f"Q: {traj.question}\nSolve step by step. Write your final answer enclosed in \\boxed{{}}. End your response with exactly one line of the form: The answer is \\boxed{{<your final answer>}}. Start your solution with 'Step{start_step}:'"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if traj.prefix_text:
            prompt += traj.prefix_text + f'\nStep{start_step}:'
        else:
            prompt += f'Step{start_step}:'
        return prompt

    def _get_cached_inputs(self, traj_idx: int) -> dict:
        cached = self._prompt_cache.get(traj_idx)
        if cached is not None:
            return cached
        traj = self.trajectories[traj_idx]
        prompt = self._build_continuation_prompt(traj)
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=4096).to(self.model.device)
        cached = {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'input_len': int(inputs.input_ids.shape[1]), 'prefix_text': traj.prefix_text}
        self._prompt_cache[traj_idx] = cached
        return cached

    def _replay(self, traj_idx: int, head_spec: Optional[List[tuple]]=None) -> str:
        traj = self.trajectories[traj_idx]
        cached = self._get_cached_inputs(traj_idx)
        budget = int(traj.n_remaining_steps if traj.n_remaining_steps is not None else self.max_replay_steps)
        budget = max(1, budget)
        max_new = self.max_new_tokens * budget
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _AnswerMarkerStop(StoppingCriteria):
            __slots__ = ('tok', 'input_len', 'markers', 'check_every', 'decode_window', '_step')

            def __init__(self, tokenizer, input_len, check_every: int=8, decode_window: int=40):
                self.tok = tokenizer
                self.input_len = input_len
                self.markers = ('the answer is', '\\boxed{', 'Final Answer')
                self.check_every = check_every
                self.decode_window = decode_window
                self._step = 0

            def __call__(self, input_ids, scores, **kwargs):
                self._step += 1
                if self._step % self.check_every != 0:
                    return False
                gen_ids = input_ids[0, self.input_len:]
                if gen_ids.numel() < self.check_every:
                    return False
                tail = gen_ids[-self.decode_window:]
                text = self.tok.decode(tail, skip_special_tokens=True)
                return any((m in text for m in self.markers))
        stopping = StoppingCriteriaList([_AnswerMarkerStop(self.tokenizer, cached['input_len'], check_every=8, decode_window=40)])

        def _do_generate() -> str:
            with torch.no_grad():
                out = self.model.generate(input_ids=cached['input_ids'], attention_mask=cached['attention_mask'], max_new_tokens=max_new, do_sample=False, temperature=1.0, pad_token_id=self._pad_token_id, stopping_criteria=stopping)
            generated = out[0][cached['input_len']:]
            return self.tokenizer.decode(generated, skip_special_tokens=True)
        if head_spec:
            from contextlib import ExitStack
            with ExitStack() as stack:
                for layer_idx, head_idx in head_spec:
                    layer_idx, head_idx = (int(layer_idx), int(head_idx))
                    key = (layer_idx, head_idx)
                    ablator = self._ablator_cache.get(key)
                    if ablator is None:
                        ablator = make_ablator(self.model, layer_idx=layer_idx, head_indices=[head_idx])
                        self._ablator_cache[key] = ablator
                    stack.enter_context(ablator)
                generated_text = _do_generate()
        else:
            generated_text = _do_generate()
        if cached['prefix_text']:
            final_text = cached['prefix_text'] + '\n' + generated_text
        else:
            final_text = generated_text
        return self.answer_extractor(final_text, traj.is_multiple_choice)

    def clean_answer(self, traj_idx: int) -> str:
        if traj_idx not in self._clean_cache:
            self._clean_cache[traj_idx] = self._replay(traj_idx, head_spec=None)
        return self._clean_cache[traj_idx]

    def _is_clean_correct(self, trajectory_id: int) -> bool:
        if trajectory_id in self._clean_correct:
            return self._clean_correct[trajectory_id]
        traj = self.trajectories[trajectory_id]
        clean = self.clean_answer(trajectory_id)
        result = bool(self.grader(clean, traj.gold, traj.is_multiple_choice))
        self._clean_correct[trajectory_id] = result
        return result

    def patch_fn(self, trajectory_id: int, layer_idx: int, head_idx: int) -> float:
        if trajectory_id < 0 or trajectory_id >= len(self.trajectories):
            return 0.0
        if self._is_clean_correct(trajectory_id):
            return 0.0
        traj = self.trajectories[trajectory_id]
        patched = self._replay(trajectory_id, head_spec=[(layer_idx, head_idx)])
        patched_correct = bool(self.grader(patched, traj.gold, traj.is_multiple_choice))
        return 1.0 if patched_correct else 0.0

    def _replay_batch(self, trajectory_ids: List[int], head_spec: Optional[List[tuple]]=None) -> List[str]:
        if not trajectory_ids:
            return []
        items = [self._get_cached_inputs(tid) for tid in trajectory_ids]
        input_lens = [it['input_len'] for it in items]
        max_len = max(input_lens)
        B = len(items)
        device = self.model.device
        pad_id = self._pad_token_id
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
        attn_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for i, it in enumerate(items):
            L = it['input_len']
            input_ids[i, -L:] = it['input_ids'][0]
            attn_mask[i, -L:] = it['attention_mask'][0]
        budgets = [int(self.trajectories[tid].n_remaining_steps if self.trajectories[tid].n_remaining_steps is not None else self.max_replay_steps) for tid in trajectory_ids]
        budget = max(1, max(budgets))
        max_new = self.max_new_tokens * budget
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _BatchedAnswerStop(StoppingCriteria):
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

            def __call__(self, input_ids, scores, **kwargs):
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
                    text = texts[i]
                    if any((m in text for m in self.markers)):
                        self._hit[i] = True
                return all(self._hit)
        stopping = StoppingCriteriaList([_BatchedAnswerStop(self.tokenizer, prompt_end=max_len, batch_size=B, check_every=64, decode_window=48)])

        def _do_generate() -> List[str]:
            with torch.no_grad():
                out = self.model.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_new, do_sample=False, temperature=1.0, pad_token_id=self._pad_token_id, stopping_criteria=stopping)
            answers: List[str] = []
            for b, tid in enumerate(trajectory_ids):
                generated = out[b, max_len:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                prefix = items[b]['prefix_text']
                final = prefix + '\n' + text if prefix else text
                is_mc = self.trajectories[tid].is_multiple_choice
                answers.append(self.answer_extractor(final, is_mc))
            return answers
        if head_spec:
            from contextlib import ExitStack
            with ExitStack() as stack:
                for layer_idx, head_idx in head_spec:
                    key = (int(layer_idx), int(head_idx))
                    ablator = self._ablator_cache.get(key)
                    if ablator is None:
                        ablator = make_ablator(self.model, layer_idx=key[0], head_indices=[key[1]])
                        self._ablator_cache[key] = ablator
                    stack.enter_context(ablator)
                return _do_generate()
        return _do_generate()

    def prime_clean_cache_batched(self, trajectory_ids: Optional[List[int]]=None) -> None:
        if trajectory_ids is None:
            trajectory_ids = list(range(len(self.trajectories)))
        to_run = [tid for tid in trajectory_ids if tid not in self._clean_cache]
        if not to_run:
            return
        answers = self._replay_batch(to_run, head_spec=None)
        for tid, ans in zip(to_run, answers):
            self._clean_cache[tid] = ans
            traj = self.trajectories[tid]
            self._clean_correct[tid] = bool(self.grader(ans, traj.gold, traj.is_multiple_choice))
    _batch_diag_count: int = 0

    def patch_fn_batch(self, trajectory_ids: Sequence[int], layer_idx: int, head_idx: int) -> List[float]:
        import time as _time
        flips: dict[int, float] = {}
        active: List[int] = []
        for tid in trajectory_ids:
            if tid < 0 or tid >= len(self.trajectories):
                flips[tid] = 0.0
                continue
            if self._is_clean_correct(tid):
                flips[tid] = 0.0
                continue
            active.append(tid)
        n_total = len(trajectory_ids)
        n_active = len(active)
        t0 = _time.time()
        if active:
            patched = self._replay_batch(active, head_spec=[(int(layer_idx), int(head_idx))])
            for tid, ans in zip(active, patched):
                traj = self.trajectories[tid]
                ok = bool(self.grader(ans, traj.gold, traj.is_multiple_choice))
                flips[tid] = 1.0 if ok else 0.0
        dt = _time.time() - t0
        HeadPatcher._batch_diag_count += 1
        idx = HeadPatcher._batch_diag_count
        if idx == 1 or idx % 16 == 0:
            max_len = 0
            if active:
                max_len = max((self._get_cached_inputs(t)['input_len'] for t in active))
            print(f'[patch_fn_batch#{idx}] L{layer_idx} H{head_idx} active={n_active}/{n_total} max_len={max_len} dt={dt:.2f}s', flush=True)
        return [flips.get(tid, 0.0) for tid in trajectory_ids]

    def clean_stats(self) -> dict:
        n = len(self.trajectories)
        self.prime_clean_cache_batched()
        n_error = 0
        for i, traj in enumerate(self.trajectories):
            ans = self.clean_answer(i)
            if not self.grader(ans, traj.gold, traj.is_multiple_choice):
                n_error += 1
        return {'n_trajectories': int(n), 'n_clean_error': int(n_error), 'clean_error_rate': n_error / n if n else float('nan')}
