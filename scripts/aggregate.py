from __future__ import annotations
import csv
import json
import os
import sys
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Optional
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loci.metrics import expected_calibration_error, adaptive_ece, maximum_calibration_error, brier_score, nll_binary, auroc, auprc, detection_recall_at_fpr, fpr_at_tpr, youden_j, confident_error_rate, cost_adjusted_accuracy

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out

def discover(out_dir: str) -> List[str]:
    return sorted(glob(os.path.join(out_dir, '**', '*.cases.jsonl'), recursive=True))

def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float('nan')
    return float(np.percentile(xs, q))

def bootstrap_ci(values: List[float], n_resamples: int=5000, alpha: float=0.05, seed: int=42) -> tuple[float, float]:
    if not values:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    n = len(arr)
    boots = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (lo, hi)
MODEL_PARAMS_TABLE = {'Qwen3-4B': 4000000000.0, 'Qwen3-8B': 8000000000.0, 'Qwen3-14B': 14000000000.0, 'Qwen3-32B': 32000000000.0, 'Qwen3_5-2B': 2000000000.0, 'Qwen3.5-2B': 2000000000.0, 'Qwen3_5-4B': 4000000000.0, 'Qwen3.5-4B': 4000000000.0, 'Qwen3_5-9B': 9000000000.0, 'Qwen3.5-9B': 9000000000.0, 'Qwen3_5-27B': 27000000000.0, 'Qwen3.5-27B': 27000000000.0}

def lookup_model_params(model_tag: str) -> float:
    if model_tag in MODEL_PARAMS_TABLE:
        return MODEL_PARAMS_TABLE[model_tag]
    low = model_tag.lower().replace('-', '').replace('_', '').replace('.', '')
    for k, v in MODEL_PARAMS_TABLE.items():
        kl = k.lower().replace('-', '').replace('_', '').replace('.', '')
        if kl in low or low in kl:
            return v
    return 0.0

def estimate_flops(case: Dict[str, Any], n_params: float) -> float:
    if n_params <= 0:
        return 0.0
    total_tokens = int(case.get('n_input_tokens', 0)) + int(case.get('n_output_tokens', 0)) + int(case.get('n_critic_output_tokens', 0))
    return 2.0 * float(n_params) * float(total_tokens)

def aggregate_method(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cases:
        return {}
    n = len(cases)
    correct = [int(bool(c.get('correct', False))) for c in cases]
    confidences = [c.get('confidence') for c in cases]
    confidences = [float(x) if x is not None else None for x in confidences]
    valid = [(c, conf) for c, conf in zip(correct, confidences) if conf is not None]
    if valid:
        v_correct, v_conf = zip(*valid)
        v_correct = list(v_correct)
        v_conf = list(v_conf)
    else:
        v_correct, v_conf = ([], [])
    pol_tok = [c.get('n_policy_output_tokens', 0) for c in cases]
    cri_tok = [c.get('n_critic_output_tokens', 0) for c in cases]
    in_tok = [c.get('n_input_tokens', 0) for c in cases]
    out_tok = [c.get('n_output_tokens', 0) for c in cases]
    calls = [c.get('n_calls', 0) for c in cases]
    lat = [float(c.get('latency_s', 0.0)) for c in cases]
    ents = [c.get('mean_entropy') for c in cases if c.get('mean_entropy') is not None]
    n_err = sum((1 for c in cases if c.get('error')))
    accuracy = float(np.mean(correct)) if correct else 0.0
    acc_lo, acc_hi = bootstrap_ci(correct)
    if v_conf:
        ece = expected_calibration_error(v_conf, v_correct)
        aece = adaptive_ece(v_conf, v_correct)
        mce = maximum_calibration_error(v_conf, v_correct)
        brier = brier_score(v_conf, v_correct)
        nll = nll_binary(v_conf, v_correct)
        v_wrong = [1 - c for c in v_correct]
        neg_conf = [-x for x in v_conf]
        det_auroc = auroc(neg_conf, v_wrong)
        det_auprc = auprc(neg_conf, v_wrong)
        recall_at_10 = detection_recall_at_fpr(neg_conf, v_wrong, 0.1)
        fpr_at_95 = fpr_at_tpr(neg_conf, v_wrong, 0.95)
        youden = youden_j(neg_conf, v_wrong)
        cer_08 = confident_error_rate(v_conf, v_correct, threshold=0.8)
        cer_09 = confident_error_rate(v_conf, v_correct, threshold=0.9)
    else:
        ece = aece = mce = brier = nll = float('nan')
        det_auroc = det_auprc = recall_at_10 = fpr_at_95 = float('nan')
        youden = cer_08 = cer_09 = float('nan')
    mean_pol = float(np.mean(pol_tok)) if pol_tok else 0.0
    mean_lat = float(np.mean(lat)) if lat else 0.0
    n_correct = sum(correct)
    tok_per_correct = sum(pol_tok) / n_correct if n_correct else float('inf')
    sec_per_correct = sum(lat) / n_correct if n_correct else float('inf')
    cost_adj = cost_adjusted_accuracy(accuracy, mean_pol, lam=0.0001)
    model_tag = cases[0].get('model', 'unknown')
    n_params = lookup_model_params(model_tag)
    flops_per_case = [estimate_flops(c, n_params) for c in cases]
    mean_flops = float(np.mean(flops_per_case)) if flops_per_case else 0.0
    total_flops = float(sum(flops_per_case))
    flops_per_correct = total_flops / n_correct if n_correct else float('inf')
    return {'n': n, 'n_correct': n_correct, 'n_errors': n_err, 'pass_at_1': accuracy, 'accuracy': accuracy, 'accuracy_ci_lo': acc_lo, 'accuracy_ci_hi': acc_hi, 'n_model_params': n_params, 'mean_flops': mean_flops, 'total_flops': total_flops, 'flops_per_correct': flops_per_correct, 'ece': ece, 'adaptive_ece': aece, 'mce': mce, 'brier': brier, 'nll': nll, 'mean_entropy': float(np.mean(ents)) if ents else float('nan'), 'det_auroc': det_auroc, 'det_auprc': det_auprc, 'recall_at_fpr10': recall_at_10, 'fpr_at_tpr95': fpr_at_95, 'youden_j': youden.get('j', float('nan')) if isinstance(youden, dict) else youden, 'cer_08': cer_08, 'cer_09': cer_09, 'mean_n_policy_output_tokens': mean_pol, 'mean_n_critic_output_tokens': float(np.mean(cri_tok)) if cri_tok else 0.0, 'mean_n_input_tokens': float(np.mean(in_tok)) if in_tok else 0.0, 'mean_n_output_tokens': float(np.mean(out_tok)) if out_tok else 0.0, 'mean_n_calls': float(np.mean(calls)) if calls else 0.0, 'mean_latency_s': mean_lat, 'latency_p50': percentile(lat, 50), 'latency_p95': percentile(lat, 95), 'latency_p99': percentile(lat, 99), 'tokens_per_correct': tok_per_correct, 'seconds_per_correct': sec_per_correct, 'cost_adjusted_accuracy': cost_adj}
HEADLINE_COLS = [('Pass@1', 'pass_at_1', '{:.3f}'), ('ci_lo', 'accuracy_ci_lo', '{:.3f}'), ('ci_hi', 'accuracy_ci_hi', '{:.3f}'), ('ECE', 'ece', '{:.3f}'), ('Brier', 'brier', '{:.3f}'), ('NLL', 'nll', '{:.3f}'), ('AUROC', 'det_auroc', '{:.3f}'), ('AUPRC', 'det_auprc', '{:.3f}'), ('CER@.8', 'cer_08', '{:.3f}'), ('pol_tok', 'mean_n_policy_output_tokens', '{:.0f}'), ('FLOPS_G', 'mean_flops', '{:.2g}'), ('calls', 'mean_n_calls', '{:.1f}'), ('lat(s)', 'mean_latency_s', '{:.2f}'), ('p95', 'latency_p95', '{:.1f}'), ('tok/cor', 'tokens_per_correct', '{:.0f}'), ('FLOP/cor', 'flops_per_correct', '{:.2g}'), ('cost_adj', 'cost_adjusted_accuracy', '{:.4f}')]

def fmt_value(v: Any, fmt: str) -> str:
    if v is None:
        return '-'
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return '-'
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)

def print_table(per_method: Dict[str, Dict[str, Any]], methods: List[str]) -> None:
    print()
    print('=' * 130)
    print('FULL METRIC COMPARISON (pooled across datasets)')
    print('=' * 130)
    header = f"{'method':22s} " + ' '.join((f'{label:>8s}' for label, _, _ in HEADLINE_COLS))
    print(header)
    print('-' * 130)
    for m in methods:
        a = per_method[m]
        marker = '*' if m == 'loci' else ' '
        cells = [fmt_value(a.get(key), fmt) for _, key, fmt in HEADLINE_COLS]
        print(f'{marker} {m:20s} ' + ' '.join((f'{c:>8s}' for c in cells)))
    print('-' * 130)
    print()

def print_per_dataset_accuracy(per_method_per_ds: Dict[str, Dict[str, List[Dict[str, Any]]]], methods: List[str]) -> None:
    all_ds = sorted({ds for m in methods for ds in per_method_per_ds.get(m, {})})
    if not all_ds:
        return
    print('PER-DATASET ACCURACY')
    print('-' * 130)
    head = f"{'method':22s} " + ' '.join((f'{ds[:14]:>14s}' for ds in all_ds))
    print(head)
    print('-' * 130)
    for m in methods:
        marker = '*' if m == 'loci' else ' '
        row = f'{marker} {m:20s} '
        cells = []
        for ds in all_ds:
            cs = per_method_per_ds.get(m, {}).get(ds, [])
            if cs:
                acc = sum((int(bool(c.get('correct', False))) for c in cs)) / len(cs)
                cells.append(f'{acc:>14.3f}')
            else:
                cells.append(f"{'-':>14s}")
        print(row + ' '.join(cells))
    print('-' * 130)
    print()

def write_summary(per_method: Dict[str, Dict[str, Any]], out_dir: str) -> None:
    agg_path = os.path.join(out_dir, 'aggregate.json')

    def _sanitize_for_json(obj):
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        if isinstance(obj, (float, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return obj
    with open(agg_path, 'w', encoding='utf-8') as f:
        json.dump(_sanitize_for_json(per_method), f, indent=2)
    print(f'wrote {agg_path}')
    csv_path = os.path.join(out_dir, 'summary.csv')
    if not per_method:
        return
    cols = sorted({k for v in per_method.values() for k in v})
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method'] + cols)
        for m, agg in sorted(per_method.items()):
            row = [m]
            for c in cols:
                v = agg.get(c)
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    row.append('')
                else:
                    row.append(v)
            w.writerow(row)
    print(f'wrote {csv_path}')

def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print('usage: python experiments/aggregate_baseline_suite.py <out_dir>')
        return 1
    out_dir = argv[1]
    case_files = discover(out_dir)
    if not case_files:
        print(f'no .cases.jsonl found under {out_dir}')
        return 1
    pooled: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_dataset: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path in case_files:
        cases = load_jsonl(path)
        if not cases:
            continue
        method = cases[0].get('method', 'unknown')
        dataset = cases[0].get('dataset', 'unknown')
        pooled[method].extend(cases)
        per_dataset[method][dataset].extend(cases)
    methods = sorted(pooled.keys())
    per_method = {m: aggregate_method(pooled[m]) for m in methods}
    print_table(per_method, methods)
    print_per_dataset_accuracy(per_dataset, methods)
    write_summary(per_method, out_dir)
    if 'loci' in per_method:
        baselines_acc = {m: per_method[m]['accuracy'] for m in methods if m != 'loci'}
        if baselines_acc:
            best_b = max(baselines_acc, key=baselines_acc.get)
            print()
            print(f'HEADLINE delta vs best baseline:')
            print(f'  best baseline = {best_b} (acc={baselines_acc[best_b]:.3f})')
            print(f"  LOCI  acc  = {per_method['loci']['accuracy']:.3f}")
            print(f"  delta acc     = {per_method['loci']['accuracy'] - baselines_acc[best_b]:+.3f}")
    return 0
if __name__ == '__main__':
    sys.exit(main(sys.argv))
