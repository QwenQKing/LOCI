from __future__ import annotations
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional
import torch
from loci.baselines.base import extract_answer
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG
GraderFn = Callable[[str, str, bool], bool]
MC_DATASETS = frozenset({'gpqa_diamond', 'arc_challenge', 'mmlu_pro_stem'})

def is_example_multiple_choice(example: Dict[str, Any]) -> bool:
    explicit = example.get('is_multiple_choice')
    if explicit is not None:
        return bool(explicit)
    src = str(example.get('source', '')).lower()
    if src in MC_DATASETS:
        return True
    if str(example.get('category', '')).lower() == 'science' and str(example.get('level', '')).lower() == 'diamond':
        return True
    return False

def _default_grader(predicted: str, gold: str, is_mc: bool) -> bool:
    try:
        from loci.metrics import grade
        return bool(grade(predicted, gold, is_mc))
    except Exception:
        return predicted.strip() == gold.strip()

class BaselineRunner(ABC):
    name: str = 'baseline'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, grader: Optional[GraderFn]=None, model_name: str='unknown') -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.grader = grader or _default_grader
        self.model_name = model_name

    def _make_case(self, example: Dict[str, Any]) -> BaselineCase:
        return BaselineCase(method=self.name, dataset=example.get('source', 'unknown'), model=self.model_name, example_id=str(example.get('id', '')), category=str(example.get('category', '')), level=example.get('level'), question=str(example.get('input', '')), ground_truth=str(example.get('target', '')), config=self.config.to_dict())

    def _grade_and_fill(self, case: BaselineCase, is_mc: bool) -> None:
        if not case.predicted and case.trajectory:
            case.predicted = extract_answer(case.trajectory, is_mc)
        case.correct = bool(self.grader(case.predicted, case.ground_truth, is_mc))
        if case.confidence is None or case.mean_entropy is None:
            self._fill_confidence_from_steps(case)

    def solve_and_pack(self, example: Dict[str, Any]) -> BaselineCase:
        case = self._make_case(example)
        is_mc = is_example_multiple_choice(example)
        t0 = time.time()
        try:
            self.solve(example, case)
        except Exception as e:
            case.error = f'{type(e).__name__}: {e}'
        case.latency_s = time.time() - t0
        self._grade_and_fill(case, is_mc)
        return case

    def solve_batch(self, examples: list, cases: list) -> None:
        for ex, case in zip(examples, cases):
            self.solve(ex, case)

    def solve_and_pack_batch(self, examples: list) -> list:
        cases = [self._make_case(ex) for ex in examples]
        is_mc_list = [is_example_multiple_choice(ex) for ex in examples]
        t0 = time.time()
        try:
            self.solve_batch(examples, cases)
        except Exception as e:
            err = f'{type(e).__name__}: {e}'
            for c in cases:
                c.error = err
        per_case_lat = (time.time() - t0) / max(1, len(examples))
        for case, is_mc in zip(cases, is_mc_list):
            if not case.latency_s:
                case.latency_s = per_case_lat
            self._grade_and_fill(case, is_mc)
        return cases

    @staticmethod
    def _fill_confidence_from_steps(case: BaselineCase) -> None:
        import numpy as np
        logps = []
        ents = []
        for s in case.step_records or []:
            if isinstance(s, dict):
                if 'logprob' in s and s['logprob'] is not None:
                    logps.append(float(s['logprob']))
                if 'entropy' in s and s['entropy'] is not None:
                    ents.append(float(s['entropy']))
        if not logps and case.intermediate:
            for k in ('confidence', 'avg_logprob', 'final_momentum'):
                v = case.intermediate.get(k)
                if isinstance(v, (int, float)):
                    logps.append(float(v))
                    break
        if logps:
            mean_logp = float(np.mean(logps))
            case.confidence = float(np.clip(np.exp(mean_logp), 0.0, 1.0))
        if ents:
            case.mean_entropy = float(np.mean(ents))

    @abstractmethod
    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        raise NotImplementedError

    def build_chat_prompt(self, user_message: str, system_message: Optional[str]=None) -> str:
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': user_message})
        import os as _os
        want_no_think = _os.environ.get('LOCI_ENABLE_THINK', '0') != '1'
        attempts = ({'enable_thinking': False},) if want_no_think else ()
        attempts = attempts + ({},)
        for kwargs in attempts:
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            except TypeError:
                continue
            except Exception:
                break
        sys_part = f'{system_message}\n\n' if system_message else ''
        return f'{sys_part}User: {user_message}\nAssistant:'

    def generate(self, prompt: str, max_new_tokens: Optional[int]=None, temperature: Optional[float]=None, do_sample: bool=True, stop_at_step_marker: bool=False) -> Dict[str, Any]:
        import os as _os
        import numpy as np
        max_nt = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=8192).to(self.model.device)
        n_in = int(inputs.input_ids.shape[1])
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        capture_scores = _os.environ.get('LOCI_CAPTURE_SCORES', '0') == '1'
        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=max_nt, temperature=max(temp, 1e-05), top_p=self.config.top_p, do_sample=do_sample and temp > 0, pad_token_id=pad_id)
            if capture_scores:
                gen_kwargs['return_dict_in_generate'] = True
                gen_kwargs['output_scores'] = True
            out = self.model.generate(**inputs, **gen_kwargs)
        if capture_scores:
            seq = out.sequences[0]
            gen_ids = seq[n_in:]
        else:
            gen_ids = out[0][n_in:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if stop_at_step_marker:
            m = re.search('(?:^|\\n)Step\\d+', text)
            if m:
                text = text[:m.start()]
        n = int(gen_ids.shape[0])
        total_logp = 0.0
        total_ent = 0.0
        if capture_scores and n > 0:
            scores = torch.stack([s[0] for s in out.scores[:n]], dim=0)
            probs = torch.softmax(scores, dim=-1)
            vocab = probs.shape[-1]
            ids = gen_ids[:n].to(probs.device)
            valid = ids < vocab
            safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
            chosen = probs.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1)
            chosen = torch.where(valid, chosen, torch.full_like(chosen, 1e-10))
            chosen = torch.clamp(chosen, 1e-10, 1.0)
            total_logp = float(torch.log(chosen).sum().item())
            mask = probs > 1e-06
            safe_probs = torch.where(mask, probs, torch.ones_like(probs))
            ent_per_pos = -(probs * torch.log(safe_probs) * mask.to(probs.dtype)).sum(dim=-1)
            total_ent = float(ent_per_pos.sum().item())
        elif n > 0:
            full = torch.cat([inputs.input_ids[0], gen_ids.to(inputs.input_ids.device)], dim=0).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(full).logits[0]
            start = n_in - 1
            gen_logits = logits[start:start + n]
            log_probs = torch.log_softmax(gen_logits, dim=-1)
            ids = gen_ids[:n].to(gen_logits.device)
            vocab = gen_logits.shape[-1]
            valid = ids < vocab
            safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
            chosen_lp = log_probs.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1)
            total_logp = float(chosen_lp.masked_select(valid).sum().item())
            probs = torch.softmax(gen_logits, dim=-1)
            mask = probs > 1e-06
            safe_probs = torch.where(mask, probs, torch.ones_like(probs))
            ent_per_pos = -(probs * torch.log(safe_probs) * mask.to(probs.dtype)).sum(dim=-1)
            total_ent = float(ent_per_pos.sum().item())
        return {'text': text.strip(), 'n_input_tokens': n_in, 'n_output_tokens': n, 'avg_logprob': total_logp / max(n, 1), 'avg_entropy': total_ent / max(n, 1)}

    def generate_batch(self, prompts: list, max_new_tokens: Optional[int]=None, temperature: Optional[float]=None, do_sample: bool=True, stop_at_step_marker: bool=False) -> list:
        import numpy as np
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, stop_at_step_marker=stop_at_step_marker)]
        max_nt = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        orig_side = getattr(self.tokenizer, 'padding_side', 'right')
        self.tokenizer.padding_side = 'left'
        try:
            enc = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True, max_length=8192)
        finally:
            self.tokenizer.padding_side = orig_side
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        input_ids = enc['input_ids']
        attention_mask = enc['attention_mask']
        B, L_in = input_ids.shape
        n_in_list = attention_mask.sum(dim=1).tolist()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        eos_id = self.tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_nt, temperature=max(temp, 1e-05), top_p=self.config.top_p, do_sample=do_sample and temp > 0, pad_token_id=pad_id)
        gen_ids_all = out[:, L_in:]
        L_gen = gen_ids_all.shape[1]
        per_row_real = torch.zeros(B, L_gen, dtype=torch.bool, device=gen_ids_all.device)
        for b in range(B):
            row = gen_ids_all[b]
            end = L_gen
            if eos_id is not None:
                eos_pos = (row == eos_id).nonzero(as_tuple=False)
                if eos_pos.numel() > 0:
                    end = int(eos_pos[0, 0].item()) + 1
            if pad_id is not None and pad_id != eos_id:
                pad_pos = (row == pad_id).nonzero(as_tuple=False)
                if pad_pos.numel() > 0:
                    first_pad = int(pad_pos[0, 0].item())
                    if first_pad < end:
                        end = first_pad
            per_row_real[b, :end] = True
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import os as _os
        sub = int(_os.environ.get('LOCI_TELEMETRY_SUBBATCH', '1'))
        sub = max(1, min(sub, B))
        if L_gen > 0:
            full_attn = torch.cat([attention_mask, per_row_real.to(attention_mask.dtype)], dim=1)
            chosen_lp = torch.zeros(B, L_gen, dtype=torch.float32, device=out.device)
            ent_per_pos = torch.zeros(B, L_gen, dtype=torch.float32, device=out.device)
            for s in range(0, B, sub):
                e = min(s + sub, B)
                with torch.no_grad():
                    logits_sub = self.model(out[s:e], attention_mask=full_attn[s:e]).logits
                gen_logits = logits_sub[:, L_in - 1:L_in - 1 + L_gen, :]
                log_probs = torch.log_softmax(gen_logits, dim=-1)
                vocab = log_probs.shape[-1]
                safe_ids = gen_ids_all[s:e].clone()
                safe_ids[safe_ids >= vocab] = 0
                chosen_lp[s:e] = log_probs.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1).float()
                probs = torch.softmax(gen_logits, dim=-1)
                mask_nonzero = probs > 1e-06
                safe_probs = torch.where(mask_nonzero, probs, torch.ones_like(probs))
                ent_per_pos[s:e] = -(probs * torch.log(safe_probs) * mask_nonzero.to(probs.dtype)).sum(dim=-1).float()
                del logits_sub, gen_logits, log_probs, probs, safe_probs, mask_nonzero, safe_ids
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            chosen_lp = torch.zeros(B, 0, device=out.device)
            ent_per_pos = torch.zeros(B, 0, device=out.device)
        results = []
        for b in range(B):
            row_mask = per_row_real[b]
            n_out = int(row_mask.sum().item())
            if n_out > 0:
                gen_row = gen_ids_all[b, :n_out]
            else:
                gen_row = gen_ids_all[b, :0]
            text = self.tokenizer.decode(gen_row, skip_special_tokens=True)
            if stop_at_step_marker:
                m = re.search('(?:^|\\n)Step\\d+', text)
                if m:
                    text = text[:m.start()]
            if n_out > 0:
                rm = row_mask.to(chosen_lp.dtype)
                sum_lp = float((chosen_lp[b] * rm).sum().item())
                sum_ent = float((ent_per_pos[b] * rm).sum().item())
                avg_lp = sum_lp / n_out
                avg_ent = sum_ent / n_out
            else:
                avg_lp = 0.0
                avg_ent = 0.0
            results.append({'text': text.strip(), 'n_input_tokens': int(n_in_list[b]), 'n_output_tokens': n_out, 'avg_logprob': avg_lp, 'avg_entropy': avg_ent})
        return results
