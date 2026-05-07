from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from loci.baselines.case_io import BaselineCase
from loci.baselines.fair_config import FairConfig, DEFAULT_FAIR_CONFIG
from loci.baselines.runner_base import BaselineRunner
from loci.probe import CommitmentProbe
from loci.reasoner import LOCIReasoner
DEFAULT_PROBE_PATH = 'results/gate1_smoke_qwen2vl.probe.npz'
DEFAULT_HCOMMIT_PATH = 'results/gate2_smoke_qwen2vl.h_commit.json'

def _load_probe(path: Optional[str]) -> Optional[CommitmentProbe]:
    if path is None or not os.path.isfile(path):
        return None
    return CommitmentProbe.load(path)

def _load_h_commit(path: Optional[str]) -> List[Tuple[int, int]]:
    if path is None or not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return [(int(x[0]), int(x[1])) for x in raw]

class LOCIRunner(BaselineRunner):
    name = 'loci'

    def __init__(self, model, tokenizer, config: FairConfig=DEFAULT_FAIR_CONFIG, arm: str='break', probe_path: Optional[str]=DEFAULT_PROBE_PATH, h_commit_path: Optional[str]=DEFAULT_HCOMMIT_PATH, commit_threshold: float=0.55, target_residual: float=0.1, max_window: int=6, commitment_layer: int=-1, fixed_k: int=3, random_seed: int=42, model_name: str='unknown', grader=None) -> None:
        super().__init__(model, tokenizer, config, grader=grader, model_name=model_name)
        self.arm = arm
        self.fixed_k = int(fixed_k)
        self.random_seed = int(random_seed)
        wants_probe = arm in {'capture', 'random_heads', 'fixed_window', 'break'}
        wants_real_heads = arm in {'fixed_window', 'break'}
        probe = _load_probe(probe_path) if wants_probe else None
        head_spec = _load_h_commit(h_commit_path) if wants_real_heads else []
        if probe is not None:
            probe.commit_threshold = float(commit_threshold)
        self.degraded = False
        if arm == 'break' and (probe is None or not head_spec):
            self.degraded = True
            probe_missing = probe is None
            head_missing = not head_spec
            arm = 'base'
            probe = None
            head_spec = []
            print(f"[WARN] LOCI arm='break' degraded to 'base': probe={('missing' if probe_missing else 'ok')}, head_spec={('empty' if head_missing else 'ok')}")
        elif arm == 'fixed_window' and (probe is None or not head_spec):
            self.degraded = True
            arm = 'base'
            probe = None
            head_spec = []
        elif arm == 'random_heads' and probe is None:
            self.degraded = True
            arm = 'base'
            probe = None
        elif arm == 'capture' and probe is None:
            self.degraded = True
            arm = 'base'
        if arm == 'random_heads' and probe is not None:
            head_spec = self._sample_random_heads(model, h_commit_path, seed=self.random_seed)
            self._is_random_heads = True
        else:
            self._is_random_heads = False
        self._effective_arm = arm
        self._probe_loaded = probe is not None
        self._n_h_commit = len(head_spec)
        self.reasoner = LOCIReasoner(model=model, tokenizer=tokenizer, max_new_tokens=config.max_new_tokens, temperature=config.temperature, max_steps=config.max_steps, commitment_probe=probe, commitment_layer=commitment_layer, commitment_head_spec=head_spec, commitment_target_residual=target_residual, commitment_max_window=max_window, capture_hidden_states=False)
        if arm == 'fixed_window' and self.reasoner.commit_policy is not None:
            policy = self.reasoner.commit_policy
            const_k = self.fixed_k
            policy.optimal_window = lambda current_score: const_k

    @staticmethod
    def _sample_random_heads(model, h_commit_path: Optional[str], seed: int=42) -> List[Tuple[int, int]]:
        import numpy as np
        rng = np.random.default_rng(int(seed))
        base = getattr(model, 'model', model)
        if hasattr(base, 'layers'):
            n_layers = len(base.layers)
        else:
            n_layers = int(getattr(model.config, 'num_hidden_layers', 24))
        n_heads = int(getattr(model.config, 'num_attention_heads', 14))
        target_n = 0
        layer_lo, layer_hi = (n_layers // 3, 2 * n_layers // 3 + 1)
        if h_commit_path and os.path.isfile(h_commit_path):
            real = _load_h_commit(h_commit_path)
            target_n = len(real)
            if real:
                layer_lo = min((l for l, _ in real))
                layer_hi = max((l for l, _ in real)) + 1
        if target_n == 0:
            target_n = max(1, int(0.05 * n_layers * n_heads))
        all_pairs = [(l, h) for l in range(layer_lo, layer_hi) for h in range(n_heads)]
        if target_n >= len(all_pairs):
            return all_pairs
        idx = rng.choice(len(all_pairs), size=target_n, replace=False)
        return [all_pairs[i] for i in idx]

    def _fill_case_from_result(self, case: BaselineCase, result: Dict[str, Any]) -> None:
        case.trajectory = result.get('trajectory', '')
        case.predicted = result.get('answer', '')
        case.n_calls = int(result.get('n_calls', 0))
        case.n_output_tokens = int(result.get('n_tokens', 0))
        case.n_policy_output_tokens = int(result.get('n_tokens', 0))
        case.n_critic_output_tokens = 0
        case.n_input_tokens = int(result.get('n_input_tokens', 0))
        case.raw_outputs = list(result.get('raw_outputs', []))
        commit = result.get('commitment', {}) or {}
        case.intermediate = {'arm': self._effective_arm, 'requested_arm': self.arm, 'degraded': self.degraded, 'probe_loaded': self._probe_loaded, 'n_h_commit': self._n_h_commit, 'tau_commit': commit.get('tau_commit'), 'tau_output': commit.get('tau_output'), 'n_triggered': commit.get('n_triggered', 0), 'n_skipped': commit.get('n_skipped', 0), 'spectral_radius': commit.get('spectral_radius', 0.0), 'decisions': commit.get('decisions', []), 'got_answer': result.get('got_answer', False), 'confidence': result.get('confidence', 0.0)}
        decisions = commit.get('decisions', []) or []
        step_entropies = list(result.get('step_entropies', []))
        step_logps = list(result.get('step_logps', []))
        step_in_toks = list(result.get('step_input_tokens', []))
        step_out_toks = list(result.get('step_output_tokens', []))
        traj_conf = float(result.get('confidence', 0.0))
        import math
        fallback_logp = math.log(max(traj_conf, 1e-10))
        fallback_ent = float(result.get('mean_entropy', 0.0))
        n_rows = max(len(step_entropies), len(decisions))
        case.step_records = []
        for i in range(n_rows):
            d = decisions[i] if i < len(decisions) else {}
            row = {'step': int(d.get('step', i)), 'logprob': float(step_logps[i]) if i < len(step_logps) else fallback_logp, 'entropy': float(step_entropies[i]) if i < len(step_entropies) else fallback_ent, 'score': float(d.get('score', 0.0)), 'intervene': bool(d.get('intervene', False)), 'window_k': int(d.get('window_k', 0)), 'predicted_residual': float(d.get('predicted_residual', 0.0)), 'n_input_tokens': int(step_in_toks[i]) if i < len(step_in_toks) else 0, 'n_output_tokens': int(step_out_toks[i]) if i < len(step_out_toks) else 0}
            case.step_records.append(row)
        case.confidence = traj_conf
        case.config_overrides = {'arm': self._effective_arm, 'commit_threshold': float(getattr(self.reasoner.commitment_probe, 'commit_threshold', 0.0)) if self.reasoner.commitment_probe else None, 'n_h_commit': self._n_h_commit, 'is_random_heads': self._is_random_heads, 'fixed_k': self.fixed_k if self._effective_arm == 'fixed_window' else None, 'commitment_layer': int(self.reasoner.commitment_layer), 'max_window': int(self.reasoner.commit_policy.max_window) if self.reasoner.commit_policy else None}

    def solve(self, example: Dict[str, Any], case: BaselineCase) -> None:
        from loci.baselines.runner_base import is_example_multiple_choice
        is_mc = is_example_multiple_choice(example)
        self.reasoner.is_multiple_choice = is_mc
        result = self.reasoner.solve(case.question)
        self._fill_case_from_result(case, result)

    def solve_batch(self, examples: List[Dict[str, Any]], cases: List[BaselineCase]) -> None:
        from loci.baselines.runner_base import is_example_multiple_choice
        import gc as _gc
        try:
            import torch as _t
        except Exception:
            _t = None
        n = len(examples)
        if n == 0:
            return
        groups = {True: [], False: []}
        for i, ex in enumerate(examples):
            groups[is_example_multiple_choice(ex)].append(i)
        results: List[Optional[Dict[str, Any]]] = [None] * n
        for is_mc_val, idx_list in groups.items():
            if not idx_list:
                continue
            self.reasoner.is_multiple_choice = is_mc_val
            qs = [cases[i].question for i in idx_list]
            sub_results = self.reasoner.solve_batch(qs)
            for slot, r in zip(idx_list, sub_results):
                results[slot] = r
            del sub_results
        for case, result in zip(cases, results):
            self._fill_case_from_result(case, result or {})
        try:
            if hasattr(self.reasoner, '_reset_state'):
                self.reasoner._reset_state()
        except Exception:
            pass
        del results
        _gc.collect()
        if _t is not None:
            try:
                _t.cuda.empty_cache()
                _t.cuda.ipc_collect()
            except Exception:
                pass

class LOCIBaseRunner(LOCIRunner):
    name = 'loci_base'

    def __init__(self, *args, **kwargs):
        kwargs['arm'] = 'base'
        super().__init__(*args, **kwargs)

class LOCIProbeOnlyRunner(LOCIRunner):
    name = 'loci_probe_only'

    def __init__(self, *args, **kwargs):
        kwargs['arm'] = 'capture'
        super().__init__(*args, **kwargs)

class LOCIRandomHeadsRunner(LOCIRunner):
    name = 'loci_random_heads'

    def __init__(self, *args, **kwargs):
        kwargs['arm'] = 'random_heads'
        super().__init__(*args, **kwargs)

class LOCIFixedWindowRunner(LOCIRunner):
    name = 'loci_fixed_window'

    def __init__(self, *args, **kwargs):
        kwargs['arm'] = 'fixed_window'
        super().__init__(*args, **kwargs)
