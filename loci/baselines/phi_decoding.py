from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, STEP_PROMPT_TEMPLATE, system_prompt_for
from loci.baselines.runner_base import BaselineRunner

class PhiDecodingRunner(BaselineRunner):
    name = 'phi_decoding'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, n_candidates: Optional[int]=None, foresight_d: int=64, model_name: str='unknown', grader=None) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        self.n_candidates = int(n_candidates or config.n_candidates)
        self.foresight_d = int(foresight_d)

    def _build_prompt(self, question: str, traj: List[str], step_idx: int, system_prompt: str) -> str:
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
        traj: List[str] = []
        step_records: List[Dict[str, Any]] = []
        got_answer = False
        for step_idx in range(self.config.max_steps):
            base_prompt = self._build_prompt(question, traj, step_idx, system_prompt)
            candidates: List[Dict[str, Any]] = []
            for _ in range(self.n_candidates):
                out = self.generate(base_prompt, stop_at_step_marker=True)
                case.n_calls += 1
                case.n_input_tokens += out['n_input_tokens']
                case.n_output_tokens += out['n_output_tokens']
                case.n_policy_output_tokens += out['n_output_tokens']
                case.raw_outputs.append(out['text'])
                candidates.append(out)
            foresight_scores: List[float] = []
            for cand in candidates:
                lookahead_prompt = base_prompt + cand['text'] + '\n'
                fc = self.generate(lookahead_prompt, max_new_tokens=self.foresight_d, stop_at_step_marker=False)
                case.n_calls += 1
                case.n_input_tokens += fc['n_input_tokens']
                case.n_output_tokens += fc['n_output_tokens']
                case.n_policy_output_tokens += fc['n_output_tokens']
                case.raw_outputs.append(fc['text'])
                foresight_scores.append(float(fc['avg_logprob']))
            best_idx = int(np.argmax(foresight_scores))
            chosen = candidates[best_idx]
            traj.append(f"Step{step_idx}: {chosen['text']}")
            step_records.append({'step': step_idx, 'text': chosen['text'][:200], 'logprob': float(chosen['avg_logprob']), 'entropy': float(chosen['avg_entropy']), 'foresight_score': float(foresight_scores[best_idx]), 'n_candidates_evaluated': len(candidates), 'selected_idx': best_idx})
            low = chosen['text'].lower()
            if 'the answer is' in low or '\\boxed{' in chosen['text']:
                got_answer = True
                break
        case.trajectory = '\n'.join(traj)
        case.step_records = step_records
        case.intermediate = {'n_candidates': self.n_candidates, 'foresight_d': self.foresight_d, 'got_answer': got_answer, 'n_steps_used': len(step_records)}
        case.config_overrides = {'n_candidates': self.n_candidates, 'foresight_d': self.foresight_d}

    def solve_batch(self, examples, cases):
        B = len(cases)
        questions = [c.question for c in cases]
        sys_prompts = [system_prompt_for(c.category) for c in cases]
        traj: List[List[str]] = [[] for _ in range(B)]
        step_records: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
        got_answer = [False] * B
        done = [False] * B
        for step_idx in range(self.config.max_steps):
            active = [i for i in range(B) if not done[i]]
            if not active:
                break
            base_prompts = [self._build_prompt(questions[i], traj[i], step_idx, sys_prompts[i]) for i in active]
            candidate_outs: List[List[Dict[str, Any]]] = [[] for _ in active]
            for _k in range(self.n_candidates):
                outs = self.generate_batch(base_prompts, stop_at_step_marker=True)
                for j, out in enumerate(outs):
                    candidate_outs[j].append(out)
                    i = active[j]
                    cases[i].n_calls += 1
                    cases[i].n_input_tokens += out['n_input_tokens']
                    cases[i].n_output_tokens += out['n_output_tokens']
                    cases[i].n_policy_output_tokens += out['n_output_tokens']
                    cases[i].raw_outputs.append(out['text'])
            foresight_outs: List[List[Dict[str, Any]]] = [[] for _ in active]
            for k in range(self.n_candidates):
                look_prompts = [base_prompts[j] + candidate_outs[j][k]['text'] + '\n' for j in range(len(active))]
                f_outs = self.generate_batch(look_prompts, max_new_tokens=self.foresight_d, stop_at_step_marker=False)
                for j, fo in enumerate(f_outs):
                    foresight_outs[j].append(fo)
                    i = active[j]
                    cases[i].n_calls += 1
                    cases[i].n_input_tokens += fo['n_input_tokens']
                    cases[i].n_output_tokens += fo['n_output_tokens']
                    cases[i].n_policy_output_tokens += fo['n_output_tokens']
                    cases[i].raw_outputs.append(fo['text'])
            for j, i in enumerate(active):
                scores = [float(fo['avg_logprob']) for fo in foresight_outs[j]]
                best_idx = int(np.argmax(scores))
                chosen = candidate_outs[j][best_idx]
                traj[i].append(f"Step{step_idx}: {chosen['text']}")
                step_records[i].append({'step': step_idx, 'text': chosen['text'][:200], 'logprob': float(chosen['avg_logprob']), 'entropy': float(chosen['avg_entropy']), 'foresight_score': float(scores[best_idx]), 'n_candidates_evaluated': len(candidate_outs[j]), 'selected_idx': best_idx})
                low = chosen['text'].lower()
                if 'the answer is' in low or '\\boxed{' in chosen['text']:
                    got_answer[i] = True
                    done[i] = True
        for i, case in enumerate(cases):
            case.trajectory = '\n'.join(traj[i])
            case.step_records = step_records[i]
            case.intermediate = {'n_candidates': self.n_candidates, 'foresight_d': self.foresight_d, 'got_answer': got_answer[i], 'n_steps_used': len(step_records[i]), 'batched': True}
            case.config_overrides = {'n_candidates': self.n_candidates, 'foresight_d': self.foresight_d}
