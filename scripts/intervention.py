from __future__ import annotations
import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from loci.probe import CommitmentProbe
from scripts.train_probe import BENCHMARK_FILES, _load_model, load_benchmark

def _seed_align(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _build_reasoner(model, tok, *, commitment_probe: Optional[CommitmentProbe], commitment_head_spec: List[Tuple[int, int]], args) -> 'LOCIReasoner':
    from loci.reasoner import LOCIReasoner
    from loci.baselines.runner_base import MC_DATASETS
    return LOCIReasoner(model=model, tokenizer=tok, max_steps=args.max_steps, is_multiple_choice=args.benchmark in MC_DATASETS, commitment_layer=args.layer, capture_hidden_states=False, commitment_probe=commitment_probe, commitment_head_spec=commitment_head_spec, commitment_target_residual=args.target_residual, commitment_max_window=args.max_window, intervention_mode=getattr(args, 'intervention_mode', 'weight_patch'), steering_alpha=getattr(args, 'steering_alpha', 1.0), steering_layer=getattr(args, 'steering_layer', None), residual_donor=getattr(args, '_residual_donor', None), retrieval_data=getattr(args, '_retrieval_data', None), retrieval_K=getattr(args, 'retrieval_K', 8), retrieval_alpha=getattr(args, 'retrieval_alpha', 0.5), retrieval_metric=getattr(args, 'retrieval_metric', 'cosine'))

def _run_arm(reasoner, examples, *, arm_name: str, is_mc: bool, batch_size: int=1, rng_seed: int=42) -> Dict[str, Any]:
    from loci.metrics import grade
    per_example: List[Dict[str, Any]] = []
    n_correct = 0
    n_tokens_total = 0
    n_calls_total = 0
    n_triggered = 0
    n_committed = 0
    n_errors = 0
    n_committed_errors = 0
    k_windows: List[int] = []

    def _accumulate(ex, result):
        nonlocal n_correct, n_tokens_total, n_calls_total
        nonlocal n_triggered, n_committed, n_errors, n_committed_errors
        gold = ex.get('target') or ex.get('answer')
        predicted = result.get('answer', '')
        correct = bool(grade(predicted, gold, is_mc))
        n_correct += int(correct)
        n_tokens_total += int(result.get('n_tokens', 0))
        n_calls_total += int(result.get('n_calls', 0))
        commitment = result.get('commitment') or {}
        decisions = commitment.get('decisions') or []
        for d in decisions:
            if d.get('intervene'):
                n_triggered += 1
                k_windows.append(int(d.get('window_k', 0)))
        is_committed = commitment.get('tau_commit') is not None
        if is_committed:
            n_committed += 1
        if not correct:
            n_errors += 1
            if is_committed:
                n_committed_errors += 1
        per_example.append({'id': ex.get('id'), 'correct': correct, 'predicted': predicted, 'gold': gold, 'tokens': result.get('n_tokens', 0), 'calls': result.get('n_calls', 0), 'tau_commit': commitment.get('tau_commit'), 'tau_output': commitment.get('tau_output'), 'n_decisions': len(decisions)})

    def _question_of(ex):
        return ex.get('input') or ex.get('question') or ex.get('problem')
    if batch_size <= 1:
        for i, ex in enumerate(examples):
            _seed_align(rng_seed + i)
            result = reasoner.solve(_question_of(ex))
            _accumulate(ex, result)
    else:
        import time
        n_total = len(examples)
        t0 = time.time()
        for chunk_start in range(0, n_total, batch_size):
            chunk = examples[chunk_start:chunk_start + batch_size]
            questions = [_question_of(ex) for ex in chunk]
            _seed_align(rng_seed + chunk_start)
            batch_t0 = time.time()
            results = reasoner.solve_batch(questions)
            batch_dt = time.time() - batch_t0
            for ex, result in zip(chunk, results):
                _accumulate(ex, result)
            print(f'[gate3/{arm_name}]   batch {chunk_start}-{chunk_start + len(chunk)}/{n_total}  dt={batch_dt:.1f}s  elapsed={time.time() - t0:.0f}s', flush=True)
    n = len(examples)
    accuracy = n_correct / n if n else float('nan')
    return {'arm': arm_name, 'n': n, 'accuracy': accuracy, 'n_correct': n_correct, 'mean_tokens': n_tokens_total / n if n else float('nan'), 'mean_calls': n_calls_total / n if n else float('nan'), 'n_triggered': int(n_triggered), 'n_committed': int(n_committed), 'n_errors': int(n_errors), 'n_committed_errors': int(n_committed_errors), 'p_detect': n_committed_errors / n_errors if n_errors else float('nan'), 'p_commit_all': n_committed / n if n else float('nan'), 'mean_window_k': float(np.mean(k_windows)) if k_windows else float('nan'), 'per_example': per_example}

def regret_bound(epsilon: float, p_detect: float, alpha_base: float) -> float:
    eps = float(np.clip(epsilon, 0.0, 1.0))
    p = float(np.clip(p_detect, 0.0, 1.0))
    a = float(np.clip(alpha_base, 0.0, 1.0))
    return float((1.0 - eps) * p * (1.0 - a))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--benchmark', default='gsm8k', choices=sorted(BENCHMARK_FILES))
    parser.add_argument('--probe', required=True, help='Path to trained Gate-1 probe (.npz).')
    parser.add_argument('--h_commit', required=True, help='Path to Gate-2 H_commit spec (JSON list of [layer, head] pairs).')
    parser.add_argument('--n_eval', type=int, default=150)
    parser.add_argument('--offset', type=int, default=0, help='Benchmark slice offset. Default 0.')
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--commit_threshold', type=float, default=None, help="Override probe's stored (calibrated) threshold. Omit to preserve the value baked into the .npz.")
    parser.add_argument('--target_residual', type=float, default=0.1)
    parser.add_argument('--max_window', type=int, default=6)
    parser.add_argument('--max_steps', type=int, default=8)
    parser.add_argument('--gate1_report', default=None, help='Optional Gate-1 JSON to pick up the measured irreversibility epsilon for the regret bound.')
    parser.add_argument('--batch_size', type=int, default=1, help='Row-parallel batch size for LOCI solve. 1 => legacy per-example solve() path (bit-identical to older runs). >1 routes through solve_batch with per-row ablation bucketing; 4-32 typical on 1.7B-8B models.')
    parser.add_argument('--intervention_mode', choices=['weight_patch', 'directional_steering', 'residual_patch', 'retrieval_patch'], default='weight_patch', help="Intervention family for arm_break. 'weight_patch' (default) zeros o_proj columns of H_commit heads during the k*-window (I3 original spec). 'directional_steering' subtracts alpha * projection onto the probe's commitment direction at commitment_layer's output during the window (I3 refined spec, minimal-destruction intervention). 'residual_patch' replaces the last-token residual at listed steering_layer(s) with a donor vector from --residual_donor (strictly stronger than projection steering).")
    parser.add_argument('--residual_donor', type=str, default=None, help="Path to .npz containing 'donor' (shape [n_layers, hidden_dim]) and 'layers' (shape [n_layers], signed layer indices) for intervention_mode=residual_patch. Typically the mean residual at commit-firing tokens on correctly-answered trajectories, produced by scripts/compute_residual_donor.py.")
    parser.add_argument('--steering_alpha', type=float, default=1.0, help='Projection subtraction coefficient for directional_steering. 0 disables; 1.0 ~ full removal of commitment component; >1 overshoots (anti-commitment push). Ignored when intervention_mode=weight_patch.')
    parser.add_argument('--retrieval_index', type=str, default=None, help='Path to .npz produced by build_retrieval_donor_index.py. Required for intervention_mode=retrieval_patch (LOCI). Contains keys_layer_<idx>, keys_unit_layer_<idx> per target layer; donor values double as their own keys.')
    parser.add_argument('--retrieval_K', type=int, default=8, help='Number of nearest neighbours to average for LOCI donor.')
    parser.add_argument('--retrieval_alpha', type=float, default=0.5, help='Patch strength for LOCI: h_new = (1-alpha)*h + alpha*donor.')
    parser.add_argument('--retrieval_metric', choices=['cosine', 'l2'], default='cosine', help='Distance metric for kNN retrieval in LOCI.')
    parser.add_argument('--steering_layer', type=str, default=None, help="Residual-stream layer(s) where directional steering applies. Single int like '17' or comma-separated list '16,17,22' for multi-layer steering (same probe direction + alpha at every listed layer). Defaults to --layer (probe's commitment_layer). Accepts negative indices. Ignored when intervention_mode=weight_patch.")
    parser.add_argument('--out', default='results/gate3_commitment_breaking.json')
    parser.add_argument('--skip_arms', type=str, default='', help='Comma-separated list of arms to skip (saves ~33-66%% wall time). Choices: base,capture. arm_break always runs. Skipped arms are recorded as null in output. Note: skipping breaks T6 regret bound check and Gate-3 GO criterion — paper still gets alpha_int via baseline_suite zero_shot for base.')
    parser.add_argument('--rng_seed', type=int, default=42, help='Base seed for per-batch RNG reset. seed = rng_seed + chunk_start, so arm_base, arm_capture, arm_break share sampling streams on identical example indices. This kills the ~5%% cross-arm CUDA-nondeterminism noise that otherwise drowns the ±2%% intervention signal.')
    args = parser.parse_args()
    from loci.baselines.runner_base import MC_DATASETS
    is_mc = args.benchmark in MC_DATASETS
    args._residual_donor = None
    if args.intervention_mode == 'residual_patch':
        if not args.residual_donor or not os.path.isfile(args.residual_donor):
            raise SystemExit(f'[gate3] --intervention_mode=residual_patch requires --residual_donor PATH pointing to an existing .npz; got {args.residual_donor!r}')
        dz = np.load(args.residual_donor)
        layers = [int(x) for x in dz['layers']]
        donor = np.asarray(dz['donor'], dtype=np.float64)
        if donor.ndim != 2 or donor.shape[0] != len(layers):
            raise SystemExit(f'[gate3] residual_donor shape mismatch: layers={len(layers)} but donor.shape={donor.shape}')
        args._residual_donor = {li: donor[i] for i, li in enumerate(layers)}
        print(f'[gate3] residual_patch donor: layers={layers}, hidden_dim={donor.shape[1]}', flush=True)
    args._retrieval_data = None
    if args.intervention_mode == 'retrieval_patch':
        if not args.retrieval_index or not os.path.isfile(args.retrieval_index):
            raise SystemExit(f'[gate3] --intervention_mode=retrieval_patch requires --retrieval_index PATH pointing to an existing .npz; got {args.retrieval_index!r}')
        rz = np.load(args.retrieval_index)
        bank = {k: np.asarray(rz[k]) for k in rz.files}
        layers = [int(x) for x in bank.get('target_layers', [])]
        n_correct = int(bank.get('n_correct', [0])[0])
        args._retrieval_data = bank
        print(f'[gate3] retrieval_patch bank: layers={layers}, n_correct={n_correct}, K={args.retrieval_K}, alpha={args.retrieval_alpha}, metric={args.retrieval_metric}', flush=True)
    print(f'[gate3] loading probe {args.probe}', flush=True)
    probe = CommitmentProbe.load(args.probe)
    if args.commit_threshold is not None:
        probe.commit_threshold = float(args.commit_threshold)
        print(f'[gate3]   threshold override (CLI): {probe.commit_threshold:.4f}', flush=True)
    else:
        print(f'[gate3]   threshold from probe (calibrated): {probe.commit_threshold:.4f}', flush=True)
    print(f'[gate3] loading H_commit {args.h_commit}', flush=True)
    with open(args.h_commit, 'r', encoding='utf-8') as f:
        head_spec_raw = json.load(f)
    head_spec: List[Tuple[int, int]] = [(int(x[0]), int(x[1])) for x in head_spec_raw]
    print(f'[gate3]   H_commit = {head_spec}', flush=True)
    epsilon = float('nan')
    if args.gate1_report and os.path.isfile(args.gate1_report):
        with open(args.gate1_report, 'r', encoding='utf-8') as f:
            g1 = json.load(f)
        epsilon = float(g1.get('irreversibility', {}).get('epsilon_upper', float('nan')))
        print(f'[gate3]   epsilon from Gate 1 = {epsilon}', flush=True)
    print(f'[gate3] loading model {args.model}', flush=True)
    model, tok = _load_model(args.model)
    examples = load_benchmark(args.benchmark, limit=args.n_eval, offset=args.offset)
    print(f'[gate3]   eval slice: {len(examples)} examples', flush=True)
    skip_arms = {a.strip() for a in args.skip_arms.split(',') if a.strip()}
    if skip_arms:
        print(f'[gate3] skip_arms = {sorted(skip_arms)} (saves wall time, breaks T6/Gate-3 GO check)', flush=True)
    if 'base' in skip_arms:
        print('[gate3] arm_base: SKIPPED', flush=True)
        arm_base = None
    else:
        print('[gate3] arm_base: legacy LOCI', flush=True)
        reasoner_base = _build_reasoner(model, tok, commitment_probe=None, commitment_head_spec=[], args=args)
        arm_base = _run_arm(reasoner_base, examples, arm_name='base', is_mc=is_mc, batch_size=args.batch_size, rng_seed=args.rng_seed)
        print(f"[gate3]   {arm_base['accuracy']:.3f} (tokens={arm_base['mean_tokens']:.0f})", flush=True)
    if 'capture' in skip_arms:
        print('[gate3] arm_capture: SKIPPED', flush=True)
        arm_capture = None
    else:
        print('[gate3] arm_capture: probe detection, no ablation', flush=True)
        reasoner_cap = _build_reasoner(model, tok, commitment_probe=probe, commitment_head_spec=[], args=args)
        arm_capture = _run_arm(reasoner_cap, examples, arm_name='capture', is_mc=is_mc, batch_size=args.batch_size, rng_seed=args.rng_seed)
        print(f"[gate3]   {arm_capture['accuracy']:.3f} (p_detect={arm_capture['p_detect']:.2f})", flush=True)
    print(f'[gate3] arm_break: ablating {len(head_spec)} heads within k*', flush=True)
    reasoner_brk = _build_reasoner(model, tok, commitment_probe=probe, commitment_head_spec=head_spec, args=args)
    arm_break = _run_arm(reasoner_brk, examples, arm_name='break', is_mc=is_mc, batch_size=args.batch_size, rng_seed=args.rng_seed)
    print(f"[gate3]   {arm_break['accuracy']:.3f} (n_triggered={arm_break['n_triggered']}, mean_k={arm_break['mean_window_k']})", flush=True)
    if arm_base is not None:
        alpha_base = arm_base['accuracy']
        alpha_int = arm_break['accuracy']
        delta = alpha_int - alpha_base
        p_detect = arm_break['p_detect']
        eps_for_bound = epsilon if not np.isnan(epsilon) else 0.5
        lower_bound = regret_bound(eps_for_bound, p_detect, alpha_base)
        bound_ok = bool(delta + 0.02 >= lower_bound)
        passed = bool(delta >= 0.02 and bound_ok)
    else:
        alpha_base = float('nan')
        alpha_int = arm_break['accuracy']
        delta = float('nan')
        p_detect = arm_break['p_detect']
        eps_for_bound = epsilon if not np.isnan(epsilon) else 0.5
        lower_bound = float('nan')
        bound_ok = False
        passed = False
    report = {'args': {k: v for k, v in vars(args).items() if not k.startswith('_')}, 'head_spec': head_spec, 'epsilon_from_gate1': epsilon, 'arm_base': arm_base, 'arm_capture': arm_capture, 'arm_break': arm_break, 'delta_accuracy': float(delta), 'regret': {'epsilon': float(eps_for_bound), 'p_detect': float(p_detect), 'alpha_base': float(alpha_base), 'lower_bound': float(lower_bound), 'observed_delta': float(delta), 'bound_satisfied': bound_ok}, 'go_condition': {'delta_accuracy': float(delta), 'threshold_delta': 0.02, 'bound_satisfied': bound_ok, 'passed': passed}}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f'[gate3] wrote {args.out}', flush=True)
    if passed:
        print('[gate3] GO — I3 clears the theorem-3-oracle bar')
        return 0
    print('[gate3] NO-GO — accuracy lift too small or regret bound violated')
    return 1
if __name__ == '__main__':
    sys.exit(main())
