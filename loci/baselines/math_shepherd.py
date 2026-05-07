from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, ONESHOT_PROMPT_TEMPLATE, system_prompt_for
from loci.baselines.runner_base import BaselineRunner
PrmStepScorer = Callable[[str, str], float]

def _split_steps(text: str) -> List[str]:
    if 'Step' in text:
        parts = []
        for chunk in text.split('Step'):
            chunk = chunk.strip()
            if chunk:
                parts.append('Step' + chunk if not chunk[0].isdigit() else chunk)
        return parts or [text.strip()]
    parts = [p.strip() for p in text.split('\n\n') if p.strip()]
    return parts or [text.strip()]

class MathShepherdRunner(BaselineRunner):
    name = 'math_shepherd'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, n_samples: Optional[int]=None, prm_step_scorer: Optional[PrmStepScorer]=None, model_name: str='unknown', grader=None) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        self.n_samples = int(n_samples or config.n_candidates)
        self.prm_step_scorer = prm_step_scorer

    def _proxy_step_score(self, prefix: str, step_text: str, mean_logprob: float) -> float:
        return float(np.exp(mean_logprob))

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        question = case.question
        system_prompt = system_prompt_for(case.category)
        user = ONESHOT_PROMPT_TEMPLATE.format(question=question)
        prompt = self.build_chat_prompt(user, system_message=system_prompt or None)
        completions: List[Dict[str, Any]] = []
        for _ in range(self.n_samples):
            out = self.generate(prompt)
            case.n_calls += 1
            case.n_input_tokens += out['n_input_tokens']
            case.n_output_tokens += out['n_output_tokens']
            case.n_policy_output_tokens += out['n_output_tokens']
            case.raw_outputs.append(out['text'])
            completions.append(out)
        traj_scores: List[float] = []
        all_step_records: List[List[Dict[str, Any]]] = []
        for c in completions:
            text = c['text']
            steps = _split_steps(text)
            step_log = []
            log_acc = 0.0
            prefix_so_far = question.strip() + '\n'
            for s_idx, step in enumerate(steps):
                if self.prm_step_scorer is not None:
                    step_score = float(self.prm_step_scorer(prefix_so_far, step))
                else:
                    step_score = self._proxy_step_score(prefix_so_far, step, c['avg_logprob'])
                step_score = max(step_score, 1e-06)
                log_acc += float(np.log(step_score))
                step_log.append({'step': s_idx, 'text': step[:200], 'step_score': float(step_score)})
                prefix_so_far += step + '\n'
            for sr in step_log:
                sr['logprob'] = float(c['avg_logprob'])
                sr['entropy'] = float(c['avg_entropy'])
            traj_scores.append(log_acc)
            all_step_records.append(step_log)
        best_idx = int(np.argmax(traj_scores))
        case.trajectory = completions[best_idx]['text']
        case.step_records = all_step_records[best_idx]
        case.intermediate = {'n_samples': self.n_samples, 'scorer': 'real_prm' if self.prm_step_scorer else 'exp_logprob_proxy', 'selected_idx': best_idx, 'traj_scores_log': [float(s) for s in traj_scores], 'voting': 'aggregate_log_step_score'}
        case.config_overrides = {'n_samples': self.n_samples}

    def solve_batch(self, examples, cases):
        prompts = []
        for case in cases:
            sys_prompt = system_prompt_for(case.category)
            user = ONESHOT_PROMPT_TEMPLATE.format(question=case.question)
            prompts.append(self.build_chat_prompt(user, system_message=sys_prompt or None))
        B = len(cases)
        all_completions: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
        for _k in range(self.n_samples):
            outs = self.generate_batch(prompts)
            for i, out in enumerate(outs):
                all_completions[i].append(out)
        for i, case in enumerate(cases):
            completions = all_completions[i]
            case.n_calls = self.n_samples
            case.n_input_tokens = sum((c['n_input_tokens'] for c in completions))
            case.n_output_tokens = sum((c['n_output_tokens'] for c in completions))
            case.n_policy_output_tokens = case.n_output_tokens
            case.n_critic_output_tokens = 0
            case.raw_outputs = [c['text'] for c in completions]
            traj_scores: List[float] = []
            all_step_records: List[List[Dict[str, Any]]] = []
            for c in completions:
                text = c['text']
                steps = _split_steps(text)
                step_log = []
                log_acc = 0.0
                prefix_so_far = case.question.strip() + '\n'
                for s_idx, step in enumerate(steps):
                    if self.prm_step_scorer is not None:
                        step_score = float(self.prm_step_scorer(prefix_so_far, step))
                    else:
                        step_score = self._proxy_step_score(prefix_so_far, step, c['avg_logprob'])
                    step_score = max(step_score, 1e-06)
                    log_acc += float(np.log(step_score))
                    step_log.append({'step': s_idx, 'text': step[:200], 'step_score': float(step_score)})
                    prefix_so_far += step + '\n'
                for sr in step_log:
                    sr['logprob'] = float(c['avg_logprob'])
                    sr['entropy'] = float(c['avg_entropy'])
                traj_scores.append(log_acc)
                all_step_records.append(step_log)
            best_idx = int(np.argmax(traj_scores))
            case.trajectory = completions[best_idx]['text']
            case.step_records = all_step_records[best_idx]
            case.intermediate = {'n_samples': self.n_samples, 'scorer': 'real_prm' if self.prm_step_scorer else 'exp_logprob_proxy', 'selected_idx': best_idx, 'traj_scores_log': [float(s) for s in traj_scores], 'voting': 'aggregate_log_step_score', 'batched': True}
            case.config_overrides = {'n_samples': self.n_samples}
