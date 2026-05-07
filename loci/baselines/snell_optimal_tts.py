from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
import numpy as np
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG, ONESHOT_PROMPT_TEMPLATE, system_prompt_for
from loci.baselines.runner_base import BaselineRunner
SnellMode = Literal['parallel', 'sequential', 'adaptive']
REVISE_INSTRUCTION = "\n\nLook back at your reasoning above. Find any mistake or unclear step. Then produce a corrected, complete solution. End with 'the answer is \\boxed{<your final answer>}'."

class SnellOptimalTTSRunner(BaselineRunner):
    name = 'snell_optimal_tts'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, budget_K: Optional[int]=None, mode: SnellMode='adaptive', model_name: str='unknown', grader=None) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        self.budget_K = int(budget_K or config.n_candidates)
        self.mode: SnellMode = mode

    def _parallel(self, base_prompt: str, case: BaselineCase) -> List[Dict[str, Any]]:
        outs = []
        for _ in range(self.budget_K):
            out = self.generate(base_prompt, max_new_tokens=self.config.max_new_tokens)
            case.n_calls += 1
            case.n_input_tokens += out['n_input_tokens']
            case.n_output_tokens += out['n_output_tokens']
            case.n_policy_output_tokens += out['n_output_tokens']
            case.raw_outputs.append(out['text'])
            outs.append(out)
        return outs

    def _sequential(self, base_prompt: str, case: BaselineCase) -> List[Dict[str, Any]]:
        chain: List[Dict[str, Any]] = []
        prompt = base_prompt
        for k in range(self.budget_K):
            out = self.generate(prompt, max_new_tokens=self.config.max_new_tokens)
            case.n_calls += 1
            case.n_input_tokens += out['n_input_tokens']
            case.n_output_tokens += out['n_output_tokens']
            case.n_policy_output_tokens += out['n_output_tokens']
            case.raw_outputs.append(out['text'])
            chain.append(out)
            prompt = base_prompt + out['text'] + REVISE_INSTRUCTION
        return chain

    def _choose_mode(self, case: BaselineCase, base_prompt: str) -> SnellMode:
        if self.mode != 'adaptive':
            return self.mode
        probe = self.generate(base_prompt, max_new_tokens=64, do_sample=True)
        case.n_calls += 1
        case.n_input_tokens += probe['n_input_tokens']
        case.n_output_tokens += probe['n_output_tokens']
        case.n_policy_output_tokens += probe['n_output_tokens']
        case.intermediate['adaptive_probe_entropy'] = float(probe['avg_entropy'])
        return 'parallel' if probe['avg_entropy'] < 0.4 else 'sequential'

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        question = case.question
        system_prompt = system_prompt_for(case.category)
        user = ONESHOT_PROMPT_TEMPLATE.format(question=question)
        base_prompt = self.build_chat_prompt(user, system_message=system_prompt or None)
        case.intermediate = {}
        chosen_mode = self._choose_mode(case, base_prompt)
        raw_outs: List[Dict[str, Any]]
        if chosen_mode == 'parallel':
            raw_outs = self._parallel(base_prompt, case)
            scores = [o['avg_logprob'] for o in raw_outs]
            best = int(np.argmax(scores))
            case.trajectory = raw_outs[best]['text']
            case.intermediate.update({'mode_fired': 'parallel', 'selected_idx': best, 'scores': [float(s) for s in scores]})
        else:
            raw_outs = self._sequential(base_prompt, case)
            case.trajectory = raw_outs[-1]['text']
            case.intermediate.update({'mode_fired': 'sequential', 'n_revisions': len(raw_outs) - 1, 'revision_logps': [float(c['avg_logprob']) for c in raw_outs]})
        case.step_records = [{'index': i, 'logprob': float(c['avg_logprob']), 'entropy': float(c['avg_entropy']), 'token_count': int(c['n_output_tokens'])} for i, c in enumerate(raw_outs)]
        case.config_overrides = {'budget_K': self.budget_K, 'mode': self.mode}

    def solve_batch(self, examples, cases):
        B = len(cases)
        questions = [c.question for c in cases]
        sys_prompts = [system_prompt_for(c.category) for c in cases]
        base_prompts = [self.build_chat_prompt(ONESHOT_PROMPT_TEMPLATE.format(question=q), system_message=sp or None) for q, sp in zip(questions, sys_prompts)]
        for case in cases:
            case.intermediate = {}
        chosen_modes: List[SnellMode]
        if self.mode == 'adaptive':
            probes = self.generate_batch(base_prompts, max_new_tokens=64)
            chosen_modes = []
            for j, p in enumerate(probes):
                cases[j].n_calls += 1
                cases[j].n_input_tokens += p['n_input_tokens']
                cases[j].n_output_tokens += p['n_output_tokens']
                cases[j].n_policy_output_tokens += p['n_output_tokens']
                cases[j].intermediate['adaptive_probe_entropy'] = float(p['avg_entropy'])
                chosen_modes.append('parallel' if p['avg_entropy'] < 0.4 else 'sequential')
        else:
            chosen_modes = [self.mode] * B
        par_idx = [i for i, m in enumerate(chosen_modes) if m == 'parallel']
        seq_idx = [i for i, m in enumerate(chosen_modes) if m == 'sequential']
        raw_outs_per_case: List[List[Dict[str, Any]]] = [[] for _ in range(B)]
        if par_idx:
            par_prompts = [base_prompts[i] for i in par_idx]
            for _k in range(self.budget_K):
                outs = self.generate_batch(par_prompts, max_new_tokens=self.config.max_new_tokens)
                for pos, i in enumerate(par_idx):
                    o = outs[pos]
                    cases[i].n_calls += 1
                    cases[i].n_input_tokens += o['n_input_tokens']
                    cases[i].n_output_tokens += o['n_output_tokens']
                    cases[i].n_policy_output_tokens += o['n_output_tokens']
                    cases[i].raw_outputs.append(o['text'])
                    raw_outs_per_case[i].append(o)
        if seq_idx:
            seq_prompts = [base_prompts[i] for i in seq_idx]
            for _k in range(self.budget_K):
                outs = self.generate_batch(seq_prompts, max_new_tokens=self.config.max_new_tokens)
                for pos, i in enumerate(seq_idx):
                    o = outs[pos]
                    cases[i].n_calls += 1
                    cases[i].n_input_tokens += o['n_input_tokens']
                    cases[i].n_output_tokens += o['n_output_tokens']
                    cases[i].n_policy_output_tokens += o['n_output_tokens']
                    cases[i].raw_outputs.append(o['text'])
                    raw_outs_per_case[i].append(o)
                    seq_prompts[pos] = base_prompts[i] + o['text'] + REVISE_INSTRUCTION
        for i, case in enumerate(cases):
            raw_outs = raw_outs_per_case[i]
            mode_fired = chosen_modes[i]
            if mode_fired == 'parallel':
                scores = [o['avg_logprob'] for o in raw_outs]
                best = int(np.argmax(scores)) if scores else 0
                case.trajectory = raw_outs[best]['text'] if raw_outs else ''
                case.intermediate.update({'mode_fired': 'parallel', 'selected_idx': best, 'scores': [float(s) for s in scores], 'batched': True})
            else:
                case.trajectory = raw_outs[-1]['text'] if raw_outs else ''
                case.intermediate.update({'mode_fired': 'sequential', 'n_revisions': max(len(raw_outs) - 1, 0), 'revision_logps': [float(c['avg_logprob']) for c in raw_outs], 'batched': True})
            case.step_records = [{'index': j, 'logprob': float(c['avg_logprob']), 'entropy': float(c['avg_entropy']), 'token_count': int(c['n_output_tokens'])} for j, c in enumerate(raw_outs)]
            case.config_overrides = {'budget_K': self.budget_K, 'mode': self.mode}
