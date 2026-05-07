from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, STEP_PROMPT_TEMPLATE, system_prompt_for
from loci.baselines.runner_base import BaselineRunner

class MURRunner(BaselineRunner):
    name = 'mur'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, momentum_rate: float=0.9, scaling_rate: float=0.8, model_name: str='unknown', grader=None) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        self.momentum_rate = float(momentum_rate)
        self.scaling_rate = float(scaling_rate)

    def _build_prompt(self, question: str, traj: list[str], step_idx: int, system_prompt: str) -> str:
        user = STEP_PROMPT_TEMPLATE.format(question=question, step_idx=step_idx)
        prompt = self.build_chat_prompt(user, system_message=system_prompt or None)
        if step_idx > 0:
            prompt += '\n'.join(traj) + f'\nStep{step_idx}:'
        else:
            prompt += 'Step0:'
        return prompt

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        question = case.question
        system_prompt = system_prompt_for(case.category)
        traj: list[str] = []
        step_records = []
        candidate_log = []
        momentum = 0.0
        got_answer = False
        for step_idx in range(self.config.max_steps):
            prompt = self._build_prompt(question, traj, step_idx, system_prompt)
            out = self.generate(prompt, stop_at_step_marker=True)
            cur_signal = out['avg_logprob']
            text = out['text']
            case.n_calls += 1
            case.n_input_tokens += out['n_input_tokens']
            case.n_output_tokens += out['n_output_tokens']
            case.n_policy_output_tokens += out['n_output_tokens']
            case.raw_outputs.append(text)
            traj.append(f'Step{step_idx}: {text}')
            triggered = False
            best_idx = 0
            cand_logps = []
            cand_texts = []
            if step_idx > 0 and np.exp(cur_signal) < self.scaling_rate * np.exp(momentum):
                triggered = True
                prompt_retry = self._build_prompt(question, traj[:-1], step_idx, system_prompt)
                cand_entropies = []
                for _ in range(self.config.n_candidates):
                    cand = self.generate(prompt_retry, stop_at_step_marker=True)
                    case.n_calls += 1
                    case.n_input_tokens += cand['n_input_tokens']
                    case.n_output_tokens += cand['n_output_tokens']
                    case.n_policy_output_tokens += cand['n_output_tokens']
                    cand_logps.append(cand['avg_logprob'])
                    cand_texts.append(cand['text'])
                    cand_entropies.append(cand['avg_entropy'])
                best_idx = int(np.argmax(cand_logps))
                cur_signal = cand_logps[best_idx]
                traj[-1] = f'Step{step_idx}: {cand_texts[best_idx]}'
                case.raw_outputs.extend(cand_texts)
                candidate_log.append({'step_idx': step_idx, 'step_uncertainty': float(np.exp(-cur_signal)), 'momentum_uncertainty': float(np.exp(-momentum)), 'selected_idx': best_idx, 'candidates': cand_texts, 'candidate_logps': [float(l) for l in cand_logps]})
            momentum = self.momentum_rate * momentum + (1.0 - self.momentum_rate) * cur_signal
            accepted_text = cand_texts[best_idx] if triggered else text
            accepted_entropy = cand_entropies[best_idx] if triggered else out['avg_entropy']
            step_records.append({'step': step_idx, 'text': accepted_text, 'logprob': float(cur_signal), 'entropy': float(accepted_entropy), 'token_count': int(out['n_output_tokens']), 'triggered': bool(triggered), 'n_candidates_tried': int(len(cand_logps))})
            if 'the answer is' in ''.join(traj).lower() or '\\boxed{' in ''.join(traj):
                got_answer = True
                break
        case.trajectory = '\n'.join(traj)
        case.step_records = step_records
        case.intermediate = {'momentum_rate': self.momentum_rate, 'scaling_rate': self.scaling_rate, 'got_answer': got_answer, 'candidate_traj': candidate_log, 'final_momentum': float(momentum)}
        case.config_overrides = {'momentum_rate': self.momentum_rate, 'scaling_rate': self.scaling_rate}

    def solve_batch(self, examples, cases):
        B = len(cases)
        questions = [c.question for c in cases]
        sys_prompts = [system_prompt_for(c.category) for c in cases]
        traj: List[List[str]] = [[] for _ in range(B)]
        step_records: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
        candidate_log: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
        momentum = [0.0] * B
        got_answer = [False] * B
        done = [False] * B
        for step_idx in range(self.config.max_steps):
            active = [i for i in range(B) if not done[i]]
            if not active:
                break
            prompts = [self._build_prompt(questions[i], traj[i], step_idx, sys_prompts[i]) for i in active]
            outs = self.generate_batch(prompts, stop_at_step_marker=True)
            triggered: List[int] = []
            base_cache = {}
            for j, i in enumerate(active):
                out = outs[j]
                cases[i].n_calls += 1
                cases[i].n_input_tokens += out['n_input_tokens']
                cases[i].n_output_tokens += out['n_output_tokens']
                cases[i].n_policy_output_tokens += out['n_output_tokens']
                cases[i].raw_outputs.append(out['text'])
                traj[i].append(f"Step{step_idx}: {out['text']}")
                base_cache[i] = out
                cur_signal = out['avg_logprob']
                if step_idx > 0 and np.exp(cur_signal) < self.scaling_rate * np.exp(momentum[i]):
                    triggered.append(j)
            retry_results: Dict[int, List[Dict[str, Any]]] = {}
            if triggered:
                retry_prompts = [self._build_prompt(questions[active[j]], traj[active[j]][:-1], step_idx, sys_prompts[active[j]]) for j in triggered]
                for _k in range(self.config.n_candidates):
                    outs_k = self.generate_batch(retry_prompts, stop_at_step_marker=True)
                    for pos, j in enumerate(triggered):
                        retry_results.setdefault(j, []).append(outs_k[pos])
            for j, i in enumerate(active):
                out = base_cache[i]
                if j in retry_results:
                    cands = retry_results[j]
                    cand_logps = [c['avg_logprob'] for c in cands]
                    cand_texts = [c['text'] for c in cands]
                    cand_entropies = [c['avg_entropy'] for c in cands]
                    best_idx = int(np.argmax(cand_logps))
                    cur_signal = cand_logps[best_idx]
                    traj[i][-1] = f'Step{step_idx}: {cand_texts[best_idx]}'
                    for c in cands:
                        cases[i].n_calls += 1
                        cases[i].n_input_tokens += c['n_input_tokens']
                        cases[i].n_output_tokens += c['n_output_tokens']
                        cases[i].n_policy_output_tokens += c['n_output_tokens']
                    cases[i].raw_outputs.extend(cand_texts)
                    candidate_log[i].append({'step_idx': step_idx, 'step_uncertainty': float(np.exp(-cur_signal)), 'momentum_uncertainty': float(np.exp(-momentum[i])), 'selected_idx': best_idx, 'candidates': cand_texts, 'candidate_logps': [float(l) for l in cand_logps]})
                    accepted_entropy = cand_entropies[best_idx]
                    triggered_flag = True
                    n_tried = self.config.n_candidates
                    step_text = cand_texts[best_idx]
                    step_tok = int(out['n_output_tokens'])
                else:
                    cur_signal = out['avg_logprob']
                    accepted_entropy = out['avg_entropy']
                    triggered_flag = False
                    n_tried = 0
                    step_text = out['text']
                    step_tok = int(out['n_output_tokens'])
                momentum[i] = self.momentum_rate * momentum[i] + (1.0 - self.momentum_rate) * cur_signal
                step_records[i].append({'step': step_idx, 'text': step_text, 'logprob': float(cur_signal), 'entropy': float(accepted_entropy), 'token_count': step_tok, 'triggered': bool(triggered_flag), 'n_candidates_tried': int(n_tried)})
                joined = ''.join(traj[i])
                if 'the answer is' in joined.lower() or '\\boxed{' in joined:
                    got_answer[i] = True
                    done[i] = True
        for i, case in enumerate(cases):
            case.trajectory = '\n'.join(traj[i])
            case.step_records = step_records[i]
            case.intermediate = {'momentum_rate': self.momentum_rate, 'scaling_rate': self.scaling_rate, 'got_answer': got_answer[i], 'candidate_traj': candidate_log[i], 'final_momentum': float(momentum[i]), 'batched': True}
            case.config_overrides = {'momentum_rate': self.momentum_rate, 'scaling_rate': self.scaling_rate}
