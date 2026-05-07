from __future__ import annotations
from typing import Any, Dict
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, system_prompt_for
from loci.baselines.runner_base import BaselineRunner
ZERO_SHOT_PROMPT_TEMPLATE = 'Q: {question}\nAnswer with your final answer enclosed in \\boxed{{}}.'

class ZeroShotRunner(BaselineRunner):
    name = 'zero_shot'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, model_name: str='unknown', grader=None) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        question = case.question
        system_prompt = system_prompt_for(case.category)
        user = ZERO_SHOT_PROMPT_TEMPLATE.format(question=question)
        prompt = self.build_chat_prompt(user, system_message=system_prompt or None)
        out = self.generate(prompt, max_new_tokens=self.config.max_new_tokens)
        case.n_calls = 1
        case.n_input_tokens = int(out['n_input_tokens'])
        case.n_output_tokens = int(out['n_output_tokens'])
        case.n_policy_output_tokens = int(out['n_output_tokens'])
        case.n_critic_output_tokens = 0
        case.trajectory = out['text']
        case.raw_outputs = [out['text']]
        case.step_records = [{'step': 0, 'logprob': float(out['avg_logprob']), 'entropy': float(out['avg_entropy']), 'n_input_tokens': int(out['n_input_tokens']), 'n_output_tokens': int(out['n_output_tokens'])}]
        case.intermediate = {'prompt_style': 'zero_shot_direct'}
        case.config_overrides = {}

    def solve_batch(self, examples, cases):
        prompts = []
        for case in cases:
            sys_prompt = system_prompt_for(case.category)
            user = ZERO_SHOT_PROMPT_TEMPLATE.format(question=case.question)
            prompts.append(self.build_chat_prompt(user, system_message=sys_prompt or None))
        outs = self.generate_batch(prompts, max_new_tokens=self.config.max_new_tokens)
        for case, out in zip(cases, outs):
            case.n_calls = 1
            case.n_input_tokens = int(out['n_input_tokens'])
            case.n_output_tokens = int(out['n_output_tokens'])
            case.n_policy_output_tokens = int(out['n_output_tokens'])
            case.n_critic_output_tokens = 0
            case.trajectory = out['text']
            case.raw_outputs = [out['text']]
            case.step_records = [{'step': 0, 'logprob': float(out['avg_logprob']), 'entropy': float(out['avg_entropy']), 'n_input_tokens': int(out['n_input_tokens']), 'n_output_tokens': int(out['n_output_tokens'])}]
            case.intermediate = {'prompt_style': 'zero_shot_direct_batched'}
            case.config_overrides = {}
