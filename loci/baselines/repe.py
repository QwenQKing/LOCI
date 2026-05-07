from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, system_prompt_for
from loci.baselines.runner_base import BaselineRunner
REPE_PROMPT_TEMPLATE = 'Q: {question}\nAnswer with your final answer enclosed in \\boxed{{}}.'

class RepERunner(BaselineRunner):
    name = 'repe'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, model_name: str='unknown', grader=None, vector_path: Optional[str]=None, layer_start: Optional[int]=None, layer_end: Optional[int]=None, coef: float=1.0, **kwargs) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        if vector_path is None:
            raise ValueError('RepERunner requires --repe_vector_path')
        self.vector_path = vector_path
        self.coef = float(coef)
        self._hook_handles: List = []
        self._load_and_install(layer_start, layer_end)

    def _load_and_install(self, layer_start, layer_end) -> None:
        data = np.load(self.vector_path, allow_pickle=True)
        layers = data['layers'].tolist()
        directions = data['directions']
        signs = data['signs']
        n_layers = len(self.model.model.layers) if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers') else len(self.model.transformer.h)
        if layer_start is None:
            layer_start = max(0, n_layers - 32)
        if layer_end is None:
            layer_end = max(0, n_layers - 10)
        layer_start = int(layer_start)
        layer_end = int(layer_end)
        if layer_end <= layer_start:
            raise ValueError(f'layer_end ({layer_end}) must be > layer_start ({layer_start})')
        self.layer_start = layer_start
        self.layer_end = layer_end
        param_dtype = next(self.model.parameters()).dtype
        device = self.model.device
        for L in range(layer_start, layer_end):
            if L not in layers:
                continue
            idx = layers.index(L)
            d = directions[idx].astype('float32')
            s = float(signs[idx])
            v = torch.from_numpy(self.coef * s * d).to(device=device, dtype=param_dtype)
            try:
                target = self.model.model.layers[L]
            except AttributeError:
                target = self.model.transformer.h[L]

            def make_hook(vec):

                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    h = h + vec
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h
                return hook
            handle = target.register_forward_hook(make_hook(v))
            self._hook_handles.append(handle)

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles = []

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        question = case.question
        system_prompt = system_prompt_for(case.category)
        user = REPE_PROMPT_TEMPLATE.format(question=question)
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
        case.intermediate = {'method': 'repe', 'layer_start': self.layer_start, 'layer_end': self.layer_end, 'coef': self.coef, 'vector_path': self.vector_path}
        case.config_overrides = {}

    def solve_batch(self, examples, cases):
        prompts = []
        for case in cases:
            sys_prompt = system_prompt_for(case.category)
            user = REPE_PROMPT_TEMPLATE.format(question=case.question)
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
            case.intermediate = {'method': 'repe', 'layer_start': self.layer_start, 'layer_end': self.layer_end, 'coef': self.coef, 'vector_path': self.vector_path}
            case.config_overrides = {}
