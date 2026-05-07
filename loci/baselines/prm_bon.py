from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, ONESHOT_PROMPT_TEMPLATE, system_prompt_for
from loci.baselines.runner_base import BaselineRunner
PrmScorer = Callable[[str, List[str]], List[float]]

class PRMBoNRunner(BaselineRunner):
    name = 'prm_bon'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, n_samples: Optional[int]=None, prm_scorer: Optional[PrmScorer]=None, model_name: str='unknown', grader=None) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        self.n_samples = int(n_samples or config.n_candidates)
        self.prm_scorer = prm_scorer

    def _proxy_score(self, completions: List[Dict[str, Any]]) -> List[float]:
        return [float(c['avg_logprob']) for c in completions]

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        question = case.question
        system_prompt = system_prompt_for(case.category)
        user = ONESHOT_PROMPT_TEMPLATE.format(question=question)
        prompt = self.build_chat_prompt(user, system_message=system_prompt or None)
        completions: List[Dict[str, Any]] = []
        for k in range(self.n_samples):
            out = self.generate(prompt, max_new_tokens=self.config.max_new_tokens)
            completions.append(out)
            case.n_calls += 1
            case.n_input_tokens += out['n_input_tokens']
            case.n_output_tokens += out['n_output_tokens']
            case.n_policy_output_tokens += out['n_output_tokens']
            case.raw_outputs.append(out['text'])
        if self.prm_scorer is not None:
            scores = self.prm_scorer(question, [c['text'] for c in completions])
        else:
            scores = self._proxy_score(completions)
        best_idx = int(np.argmax(scores))
        case.trajectory = completions[best_idx]['text']
        case.step_records = [{'candidate_idx': k, 'score': float(scores[k]), 'logprob': float(completions[k]['avg_logprob']), 'entropy': float(completions[k]['avg_entropy']), 'token_count': int(completions[k]['n_output_tokens']), 'selected': k == best_idx} for k in range(self.n_samples)]
        case.intermediate = {'n_samples': self.n_samples, 'scorer': 'prm' if self.prm_scorer is not None else 'proxy_logprob', 'selected_idx': best_idx, 'scores': [float(s) for s in scores]}
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
            outs = self.generate_batch(prompts, max_new_tokens=self.config.max_new_tokens)
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
            if self.prm_scorer is not None:
                scores = self.prm_scorer(case.question, [c['text'] for c in completions])
            else:
                scores = self._proxy_score(completions)
            best_idx = int(np.argmax(scores))
            case.trajectory = completions[best_idx]['text']
            case.step_records = [{'candidate_idx': k, 'score': float(scores[k]), 'logprob': float(completions[k]['avg_logprob']), 'entropy': float(completions[k]['avg_entropy']), 'token_count': int(completions[k]['n_output_tokens']), 'selected': k == best_idx} for k in range(self.n_samples)]
            case.intermediate = {'n_samples': self.n_samples, 'scorer': 'prm' if self.prm_scorer is not None else 'proxy_logprob', 'selected_idx': best_idx, 'scores': [float(s) for s in scores], 'batched': True}
            case.config_overrides = {'n_samples': self.n_samples}
