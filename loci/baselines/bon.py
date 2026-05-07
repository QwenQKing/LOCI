from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Optional
from loci.baselines.base import extract_answer
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, ONESHOT_PROMPT_TEMPLATE, system_prompt_for
from loci.baselines.runner_base import BaselineRunner

class BoNRunner(BaselineRunner):
    name = 'bon'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, n_samples: Optional[int]=None, model_name: str='unknown', grader=None) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        self.n_samples = int(n_samples or config.n_candidates)

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        question = case.question
        from loci.baselines.runner_base import is_example_multiple_choice
        is_mc = is_example_multiple_choice(example)
        system_prompt = system_prompt_for(case.category)
        user = ONESHOT_PROMPT_TEMPLATE.format(question=question)
        prompt = self.build_chat_prompt(user, system_message=system_prompt or None)
        completions: List[Dict[str, Any]] = []
        answers: List[str] = []
        for k in range(self.n_samples):
            out = self.generate(prompt, max_new_tokens=self.config.max_new_tokens)
            completions.append(out)
            case.n_calls += 1
            case.n_input_tokens += out['n_input_tokens']
            case.n_output_tokens += out['n_output_tokens']
            case.n_policy_output_tokens += out['n_output_tokens']
            case.raw_outputs.append(out['text'])
            answers.append(extract_answer(out['text'], is_mc))
        vote_pool = [a for a in answers if a]
        if vote_pool:
            counter = Counter(vote_pool)
            top_count = counter.most_common(1)[0][1]
            top_answers = [a for a, c in counter.items() if c == top_count]
            if len(top_answers) == 1:
                winner = top_answers[0]
                best_idx = answers.index(winner)
            else:
                tied_idxs = [i for i, a in enumerate(answers) if a in top_answers]
                best_idx = max(tied_idxs, key=lambda i: completions[i]['avg_logprob'])
        else:
            best_idx = max(range(len(completions)), key=lambda i: completions[i]['avg_logprob'])
        case.trajectory = completions[best_idx]['text']
        case.predicted = answers[best_idx]
        case.step_records = [{'candidate_idx': k, 'logprob': float(completions[k]['avg_logprob']), 'entropy': float(completions[k]['avg_entropy']), 'extracted_answer': answers[k], 'n_input_tokens': int(completions[k]['n_input_tokens']), 'n_output_tokens': int(completions[k]['n_output_tokens']), 'selected': k == best_idx} for k in range(self.n_samples)]
        case.intermediate = {'n_samples': self.n_samples, 'selection': 'majority_vote', 'vote_counts': dict(Counter(answers)), 'selected_idx': best_idx}
        case.config_overrides = {'n_samples': self.n_samples}

    def solve_batch(self, examples, cases):
        from loci.baselines.runner_base import is_example_multiple_choice
        is_mc_list = [is_example_multiple_choice(ex) for ex in examples]
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
        for i, (case, is_mc) in enumerate(zip(cases, is_mc_list)):
            completions = all_completions[i]
            answers = [extract_answer(c['text'], is_mc) for c in completions]
            case.n_calls = self.n_samples
            case.n_input_tokens = sum((c['n_input_tokens'] for c in completions))
            case.n_output_tokens = sum((c['n_output_tokens'] for c in completions))
            case.n_policy_output_tokens = case.n_output_tokens
            case.n_critic_output_tokens = 0
            case.raw_outputs = [c['text'] for c in completions]
            vote_pool = [a for a in answers if a]
            if vote_pool:
                counter = Counter(vote_pool)
                top_count = counter.most_common(1)[0][1]
                top_answers = [a for a, c in counter.items() if c == top_count]
                if len(top_answers) == 1:
                    winner = top_answers[0]
                    best_idx = answers.index(winner)
                else:
                    tied_idxs = [j for j, a in enumerate(answers) if a in top_answers]
                    best_idx = max(tied_idxs, key=lambda k: completions[k]['avg_logprob'])
            else:
                best_idx = max(range(len(completions)), key=lambda k: completions[k]['avg_logprob'])
            case.trajectory = completions[best_idx]['text']
            case.predicted = answers[best_idx]
            case.step_records = [{'candidate_idx': k, 'logprob': float(completions[k]['avg_logprob']), 'entropy': float(completions[k]['avg_entropy']), 'extracted_answer': answers[k], 'n_input_tokens': int(completions[k]['n_input_tokens']), 'n_output_tokens': int(completions[k]['n_output_tokens']), 'selected': k == best_idx} for k in range(self.n_samples)]
            case.intermediate = {'n_samples': self.n_samples, 'selection': 'majority_vote', 'vote_counts': dict(Counter(answers)), 'selected_idx': best_idx, 'batched': True}
            case.config_overrides = {'n_samples': self.n_samples}
