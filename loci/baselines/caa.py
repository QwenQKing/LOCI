from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import torch
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, system_prompt_for
from loci.baselines.runner_base import BaselineRunner
CAA_PROMPT_TEMPLATE = 'Q: {question}\nAnswer with your final answer enclosed in \\boxed{{}}.'

class CAARunner(BaselineRunner):
    name = 'caa'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, model_name: str='unknown', grader=None, vector_path: Optional[str]=None, layer: Optional[int]=None, coef: float=2.0, **kwargs) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        if vector_path is None:
            raise ValueError('CAARunner requires --caa_vector_path')
        if layer is None:
            raise ValueError('CAARunner requires --caa_layer')
        self.vector_path = vector_path
        self.layer = int(layer)
        self.coef = float(coef)
        self._hook_handle = None
        self._steering_vec = None
        self._load_vector()
        self._install_hook()

    def _load_vector(self) -> None:
        data = np.load(self.vector_path, allow_pickle=True)
        layers = data['layers'].tolist()
        vectors = data['vectors']
        if self.layer not in layers:
            raise ValueError(f'layer {self.layer} not in calibrated layers {layers}; calibrate first with scripts/calibrate_caa.py')
        idx = layers.index(self.layer)
        v = torch.from_numpy(vectors[idx].astype('float32'))
        self._steering_vec = v.to(device=self.model.device, dtype=next(self.model.parameters()).dtype)

    def _install_hook(self) -> None:
        try:
            target = self.model.model.layers[self.layer]
        except (AttributeError, IndexError):
            target = self.model.transformer.h[self.layer]
        v = self._steering_vec
        coef = self.coef

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h = h + coef * v
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        self._hook_handle = target.register_forward_hook(hook)

    def remove_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self):
        try:
            self.remove_hook()
        except Exception:
            pass

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        question = case.question
        system_prompt = system_prompt_for(case.category)
        user = CAA_PROMPT_TEMPLATE.format(question=question)
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
        case.intermediate = {'method': 'caa', 'layer': self.layer, 'coef': self.coef, 'vector_path': self.vector_path}
        case.config_overrides = {}

    def solve_batch(self, examples, cases):
        prompts = []
        for case in cases:
            sys_prompt = system_prompt_for(case.category)
            user = CAA_PROMPT_TEMPLATE.format(question=case.question)
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
            case.intermediate = {'method': 'caa', 'layer': self.layer, 'coef': self.coef, 'vector_path': self.vector_path}
            case.config_overrides = {}
