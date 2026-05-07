from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loci.commitment import CommitmentTrace, summarize_commitment_gaps
from loci.probe import CommitmentProbe
_CACHE_VERSION = 'v1'
_CACHE_DIR = os.path.join('cache', 'gate1_phaseA')

def _parse_layer_list(spec: str) -> List[int]:
    if not spec:
        return []
    out: List[int] = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    if not out:
        raise ValueError(f'--collect_layers spec {spec!r} parsed empty')
    return out

def _phase_a_cache_path(args: argparse.Namespace) -> str:
    model_tag = os.path.basename(str(args.model).rstrip('/')) or 'unknown_model'
    layers_part = f'ly{args.layer}'
    if args.collect_layers:
        signed = _parse_layer_list(args.collect_layers)
        layers_part = 'cly' + '_'.join((str(x) for x in signed))
    key = f'{_CACHE_VERSION}__{model_tag}__{args.benchmark}__ntr{args.n_train}__nev{args.n_eval}__ms{args.max_steps}__{layers_part}'
    return os.path.join(_CACHE_DIR, f'{key}.pkl')
BENCHMARK_FILES = {'gsm8k': 'data/unified/gsm8k.json', 'math_500': 'data/unified/math_500.json', 'aime2024': 'data/unified/aime2024.json', 'aime2025': 'data/unified/aime2025.json', 'gpqa_diamond': 'data/unified/gpqa_diamond.json', 'amc2023': 'data/unified/amc2023.json', 'olympiadbench': 'data/unified/olympiadbench.json', 'theoremqa': 'data/unified/theoremqa.json', 'mmlu_pro_stem': 'data/unified/mmlu_pro_stem.json', 'arc_challenge': 'data/unified/arc_challenge.json', 'bbh': 'data/unified/bbh.json', 'musr': 'data/unified/musr.json', 'sample_256': 'data/unified/sample_256.json', 'sample_2502': 'data/unified/sample_2502.json', 'sample_e4_n200': 'data/unified/sample_e4_n200.json', 'sample_256_probetrain': 'data/unified/sample_256_probetrain.json', 'sample_biased_1280': 'data/unified/sample_biased_1280.json', 'sample_biased_1280_shard0of4': 'data/unified/sample_biased_1280_shard0of4.json', 'sample_biased_1280_shard1of4': 'data/unified/sample_biased_1280_shard1of4.json', 'sample_biased_1280_shard2of4': 'data/unified/sample_biased_1280_shard2of4.json', 'sample_biased_1280_shard3of4': 'data/unified/sample_biased_1280_shard3of4.json'}

def load_benchmark(name: str, limit: Optional[int]=None, offset: int=0) -> List[Dict[str, Any]]:
    path = BENCHMARK_FILES.get(name)
    if path is None:
        raise ValueError(f'unknown benchmark {name!r}; known: {sorted(BENCHMARK_FILES)}')
    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    sliced = items[offset:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced

def _load_model(model_id: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    sf_path = os.path.join(model_id, 'model.safetensors')
    use_manual = torch.cuda.is_available() and os.path.isdir(model_id) and os.path.isfile(sf_path)
    if not use_manual:
        attn_impl = os.environ.get('LOCI_ATTN_IMPL', '').strip()
        if not attn_impl:
            try:
                import flash_attn
                attn_impl = 'flash_attention_2'
            except Exception:
                attn_impl = 'sdpa'
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True, attn_implementation=attn_impl)
        except (TypeError, ValueError) as e:
            print(f'[gate1] attn_impl={attn_impl} rejected ({e}); retrying without attn_implementation', flush=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
        print(f'[gate1] loaded with attn_implementation={attn_impl}', flush=True)
        model.eval()
        return (model, tok)
    import gc
    import json as _json
    import struct
    from accelerate import init_empty_weights
    gpu_device = torch.device('cuda', torch.cuda.current_device())
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16)

    def _materialise(module):
        for name, param in list(module.named_parameters(recurse=False)):
            if param.device.type == 'meta':
                module._parameters[name] = torch.nn.Parameter(torch.empty(param.shape, dtype=torch.float16, device=gpu_device), requires_grad=False)
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type == 'meta':
                dt = torch.float16 if buf.dtype == torch.float32 else buf.dtype
                module._buffers[name] = torch.empty(buf.shape, dtype=dt, device=gpu_device)
        for child in module.children():
            _materialise(child)
    _materialise(model)
    gc.collect()
    torch.cuda.empty_cache()
    dtype_map = {'F16': torch.float16, 'BF16': torch.bfloat16, 'F32': torch.float32}
    with open(sf_path, 'rb') as fh:
        hlen = struct.unpack('<Q', fh.read(8))[0]
        header = _json.loads(fh.read(hlen))
        data_start = 8 + hlen
        entries = [(k, v) for k, v in header.items() if k != '__metadata__']
        param_map = dict(model.named_parameters())
        buffer_map = dict(model.named_buffers())
        with torch.no_grad():
            for k, meta in entries:
                off_s, off_e = meta['data_offsets']
                nbytes = off_e - off_s
                fh.seek(data_start + off_s)
                raw = bytearray(nbytes)
                fh.readinto(raw)
                t_cpu = torch.frombuffer(raw, dtype=dtype_map[meta['dtype']]).reshape(meta['shape'])
                t_gpu = t_cpu.to(gpu_device, dtype=torch.float16)
                del t_cpu, raw
                if k in param_map:
                    param_map[k].data.copy_(t_gpu)
                elif k in buffer_map:
                    buffer_map[k].data.copy_(t_gpu)
                del t_gpu
    cpu_bufs = [n for n, b in model.named_buffers() if b.device.type == 'cpu']
    if cpu_bufs:
        model = model.to(gpu_device)
        gc.collect()
        torch.cuda.empty_cache()
    model.eval()
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
    return (model, tok)

def collect_trajectories(reasoner, examples: List[Dict[str, Any]], is_mc: bool=False, batch_size: int=1) -> List[Dict[str, Any]]:
    from loci.metrics import grade
    records: List[Dict[str, Any]] = []

    def _extract_qg(ex: Dict[str, Any]):
        q = ex.get('input') or ex.get('question') or ex.get('problem')
        g = ex.get('target') or ex.get('answer')
        return (q, g)

    def _build_record(ex: Dict[str, Any], question: str, gold: Any, result: Dict[str, Any], hiddens: List[np.ndarray], hiddens_by_layer: Dict[int, List[np.ndarray]]) -> Dict[str, Any]:
        predicted = result.get('answer')
        correct = bool(grade(predicted, gold, is_mc))
        raw_steps = list(result.get('raw_outputs') or [])
        rec = {'id': ex.get('id'), 'question': question, 'gold': gold, 'predicted': predicted, 'correct': correct, 'confidence': float(result.get('confidence', 0.0)), 'n_steps': len(hiddens), 'tau_output': len(hiddens) - 1 if hiddens else 0, 'hiddens': hiddens, 'raw_outputs': raw_steps}
        if hiddens_by_layer:
            rec['hiddens_by_layer'] = hiddens_by_layer
        return rec
    if batch_size > 1:
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            qs_golds = [_extract_qg(ex) for ex in chunk]
            questions = [q for q, _ in qs_golds]
            batch_results = reasoner.solve_phaseA_batch(questions)
            for ex, (q, g), res in zip(chunk, qs_golds, batch_results):
                hiddens = [h.copy() for h in res.get('commit_hiddens', [])]
                hbl_in = res.get('commit_hiddens_by_layer') or {}
                hiddens_by_layer: Dict[int, List[np.ndarray]] = {int(li): [h.copy() for h in hs] for li, hs in hbl_in.items()}
                records.append(_build_record(ex, q, g, res, hiddens, hiddens_by_layer))
        return records
    for ex in examples:
        question, gold = _extract_qg(ex)
        result = reasoner.solve(question)
        hiddens = [h.copy() for h in reasoner.commit_hiddens]
        hiddens_by_layer: Dict[int, List[np.ndarray]] = {}
        if reasoner.commit_hiddens_by_layer:
            for li, hs in reasoner.commit_hiddens_by_layer.items():
                hiddens_by_layer[int(li)] = [h.copy() for h in hs]
        records.append(_build_record(ex, question, gold, result, hiddens, hiddens_by_layer))
    return records

def _trajectory_label(record: Dict[str, Any], conf_floor: float) -> Optional[int]:
    if record['confidence'] < conf_floor:
        return None
    return 0 if record['correct'] else 1

def diagnose_confidence(records: List[Dict[str, Any]]) -> Dict[str, float]:
    confs = np.asarray([r['confidence'] for r in records], dtype=np.float64)
    if confs.size == 0:
        return {}
    qs = {f'q{int(q * 100):02d}': float(np.quantile(confs, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    qs['mean'] = float(confs.mean())
    qs['min'] = float(confs.min())
    qs['max'] = float(confs.max())
    return qs

def auto_pick_conf_floor(records: List[Dict[str, Any]], requested: float, min_per_class: int=5) -> float:

    def counts_at(floor: float) -> Tuple[int, int]:
        pos = neg = 0
        for r in records:
            if r['confidence'] < floor:
                continue
            if r['correct']:
                neg += 1
            else:
                pos += 1
        return (pos, neg)
    pos, neg = counts_at(requested)
    if pos >= min_per_class and neg >= min_per_class:
        return requested
    confs = sorted({float(r['confidence']) for r in records})
    for c in confs:
        p, n = counts_at(c)
        if p >= min_per_class and n >= min_per_class:
            return c
    return 0.0

def build_probe_matrix(records: List[Dict[str, Any]], conf_floor: float, first_k: int=0, last_k: int=0) -> Tuple[np.ndarray, np.ndarray, int]:
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    for r in records:
        label = _trajectory_label(r, conf_floor)
        if label is None:
            continue
        hiddens = r['hiddens']
        if first_k > 0:
            hiddens = hiddens[:first_k]
        elif last_k > 0:
            hiddens = hiddens[-last_k:]
        for h in hiddens:
            X_rows.append(h.astype(np.float64))
            y_rows.append(label)
    if not X_rows:
        raise RuntimeError('no confident trajectories to train on; lower conf_floor or collect more data')
    X = np.stack(X_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.int64)
    return (X, y, X.shape[1])

def fit_pca(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:k]
    return (basis.astype(np.float64), mean.astype(np.float64))

def project_records(records: List[Dict[str, Any]], basis: np.ndarray, mean: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        new_h = []
        for h in r['hiddens']:
            v = h.astype(np.float64) - mean
            new_h.append((v @ basis.T).astype(np.float32))
        nr = dict(r)
        nr['hiddens'] = new_h
        out.append(nr)
    return out

def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float(np.mean(y_pred == y_true))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {'accuracy': acc, 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1), 'n': int(y_true.shape[0]), 'n_pos': int(np.sum(y_true == 1)), 'n_neg': int(np.sum(y_true == 0))}

def calibrate_commit_threshold(probe: CommitmentProbe, records: List[Dict[str, Any]], conf_floor: float, fpr_target: float) -> Dict[str, float]:
    cor_max: List[float] = []
    err_max: List[float] = []
    for r in records:
        label = _trajectory_label(r, conf_floor)
        if label is None:
            continue
        hiddens = r['hiddens']
        if not hiddens:
            continue
        X = np.stack([h.astype(np.float64) for h in hiddens], axis=0)
        scores = probe.predict_proba(X)
        m = float(np.max(scores))
        (err_max if label == 1 else cor_max).append(m)
    if not cor_max:
        return {'enabled': 1.0, 'fpr_target': float(fpr_target), 'n_cor_used': 0.0, 'n_err_used': float(len(err_max)), 'threshold': float('nan')}
    thr = float(np.quantile(cor_max, 1.0 - fpr_target))
    n_err_above = int(sum((1 for m in err_max if m >= thr))) if err_max else 0
    return {'enabled': 1.0, 'fpr_target': float(fpr_target), 'n_cor_used': float(len(cor_max)), 'n_err_used': float(len(err_max)), 'threshold': thr, 'cor_max_median': float(np.median(cor_max)), 'err_max_median': float(np.median(err_max)) if err_max else float('nan'), 'n_err_above_threshold': float(n_err_above), 'tpr_error_at_fpr': n_err_above / len(err_max) if err_max else float('nan')}

def train_probe(records: List[Dict[str, Any]], conf_floor: float, commit_threshold: float, l2: float=1.0, holdout_records: Optional[List[Dict[str, Any]]]=None, train_first_k: int=0, train_last_k: int=0, calibration_fpr: float=0.0) -> Tuple[CommitmentProbe, Dict[str, Any]]:
    X, y, dim = build_probe_matrix(records, conf_floor, first_k=train_first_k, last_k=train_last_k)
    probe = CommitmentProbe(dim=dim, l2=l2, commit_threshold=commit_threshold)
    probe.fit(X, y)
    train_pred = (probe.predict_proba(X) >= 0.5).astype(np.int64)
    train_metrics = _classification_metrics(y, train_pred)
    stats: Dict[str, Any] = {'n_samples': int(X.shape[0]), 'n_positive': int(np.sum(y == 1)), 'n_negative': int(np.sum(y == 0)), 'train_accuracy': train_metrics['accuracy'], 'train_precision': train_metrics['precision'], 'train_recall': train_metrics['recall']}
    if calibration_fpr and calibration_fpr > 0.0:
        cal = calibrate_commit_threshold(probe, records, conf_floor=conf_floor, fpr_target=float(calibration_fpr))
        stats['calibration'] = cal
        thr = cal.get('threshold', float('nan'))
        if thr == thr:
            stats['commit_threshold_before'] = float(commit_threshold)
            probe.commit_threshold = float(thr)
            stats['commit_threshold_calibrated'] = float(thr)
    if holdout_records:
        try:
            Xh, yh, _ = build_probe_matrix(holdout_records, conf_floor, first_k=train_first_k, last_k=train_last_k)
            ph = (probe.predict_proba(Xh) >= 0.5).astype(np.int64)
            hm = _classification_metrics(yh, ph)
            stats['holdout'] = hm
            stats['holdout_accuracy'] = hm['accuracy']
            stats['holdout_precision'] = hm['precision']
            stats['holdout_recall'] = hm['recall']
            stats['overfit_gap'] = train_metrics['accuracy'] - hm['accuracy']
        except RuntimeError:
            stats['holdout_accuracy'] = float('nan')
            stats['holdout_precision'] = float('nan')
            stats['holdout_recall'] = float('nan')
            stats['overfit_gap'] = float('nan')
    return (probe, stats)

def estimate_irreversibility(records: List[Dict[str, Any]], traces: List[CommitmentTrace], conf_floor: float) -> Dict[str, float]:
    n_locked_total = 0
    n_committed = 0
    n_committed_and_correct = 0
    for r, t in zip(records, traces):
        if t.tau_commit is None:
            continue
        n_locked_total += 1
        if r['confidence'] < conf_floor:
            continue
        n_committed += 1
        if r['correct']:
            n_committed_and_correct += 1
    epsilon = n_committed_and_correct / n_committed if n_committed else float('nan')
    return {'conf_floor': float(conf_floor), 'n_locked_total': float(n_locked_total), 'n_locked_confident': float(n_committed), 'n_committed_and_correct': float(n_committed_and_correct), 'epsilon_upper': float(epsilon)}

def temporal_distribution_report(traces: List[CommitmentTrace]) -> Dict[str, Any]:
    err_total = [t for t in traces if t.final_was_error is True]
    cor_total = [t for t in traces if t.final_was_error is False]
    err_locked = [t for t in err_total if t.tau_commit is not None]
    cor_locked = [t for t in cor_total if t.tau_commit is not None]

    def _q(vals: List[float]) -> Dict[str, float]:
        if not vals:
            n = float('nan')
            return {'p25': n, 'p50': n, 'p75': n, 'n': 0.0}
        arr = np.asarray(vals, dtype=np.float64)
        return {'p25': float(np.quantile(arr, 0.25)), 'p50': float(np.quantile(arr, 0.5)), 'p75': float(np.quantile(arr, 0.75)), 'n': float(arr.size)}

    def _mwu(a: List[float], b: List[float]) -> Tuple[float, float, float]:
        if not a or not b:
            n = float('nan')
            return (n, n, n)
        try:
            from scipy.stats import mannwhitneyu
            _, pg = mannwhitneyu(a, b, alternative='greater')
            _, pl = mannwhitneyu(a, b, alternative='less')
            _, p2 = mannwhitneyu(a, b, alternative='two-sided')
            return (float(pg), float(pl), float(p2))
        except Exception:
            n = float('nan')
            return (n, n, n)
    tau_err = [float(t.tau_commit) for t in err_locked]
    tau_cor = [float(t.tau_commit) for t in cor_locked]
    gap_err = [float(t.commitment_gap) for t in err_locked if t.commitment_gap is not None]
    gap_cor = [float(t.commitment_gap) for t in cor_locked if t.commitment_gap is not None]
    tau_pg, tau_pl, tau_p2 = _mwu(tau_err, tau_cor)
    gap_pg, gap_pl, gap_p2 = _mwu(gap_err, gap_cor)
    n_err_total = len(err_total)
    n_cor_total = len(cor_total)
    n_err_locked = len(err_locked)
    n_cor_locked = len(cor_locked)
    n_err_unlocked = n_err_total - n_err_locked
    n_cor_unlocked = n_cor_total - n_cor_locked
    lock_contingency = {'n_err_locked': int(n_err_locked), 'n_err_unlocked': int(n_err_unlocked), 'n_cor_locked': int(n_cor_locked), 'n_cor_unlocked': int(n_cor_unlocked)}
    if n_err_total and n_cor_total and (n_err_locked + n_cor_locked > 0):
        try:
            from scipy.stats import chi2_contingency, fisher_exact
            table = np.array([[n_err_locked, n_err_unlocked], [n_cor_locked, n_cor_unlocked]], dtype=np.int64)
            chi2, chi2_p, _, _ = chi2_contingency(table, correction=False)
            _, fisher_p_greater = fisher_exact(table, alternative='greater')
            _, fisher_p_two = fisher_exact(table, alternative='two-sided')
            lock_contingency['chi2'] = float(chi2)
            lock_contingency['chi2_p_two_sided'] = float(chi2_p)
            lock_contingency['fisher_p_err_greater'] = float(fisher_p_greater)
            lock_contingency['fisher_p_two_sided'] = float(fisher_p_two)
            pe = n_err_locked / n_err_total
            pc = n_cor_locked / n_cor_total
            lock_contingency['lock_rate_ratio'] = float(pe / pc) if pc > 0 else float('inf')
        except Exception:
            lock_contingency['chi2_p_two_sided'] = float('nan')
            lock_contingency['fisher_p_err_greater'] = float('nan')
    return {'n_err_total': float(n_err_total), 'n_cor_total': float(n_cor_total), 'lock_rate_err': len(err_locked) / n_err_total if n_err_total else float('nan'), 'lock_rate_cor': len(cor_locked) / n_cor_total if n_cor_total else float('nan'), 'lock_contingency': lock_contingency, 'tau_commit_err': _q(tau_err), 'tau_commit_cor': _q(tau_cor), 'tau_mwu_err_less_than_cor_p': tau_pl, 'tau_mwu_err_greater_than_cor_p': tau_pg, 'tau_mwu_two_sided_p': tau_p2, 'delta_err': _q(gap_err), 'delta_cor': _q(gap_cor), 'delta_mwu_err_greater_than_cor_p': gap_pg, 'delta_mwu_err_less_than_cor_p': gap_pl, 'delta_mwu_two_sided_p': gap_p2}

def compute_traces(records: List[Dict[str, Any]], probe: CommitmentProbe, conf_floor: float) -> List[CommitmentTrace]:
    traces: List[CommitmentTrace] = []
    for r in records:
        t = CommitmentTrace()
        t.step_scores = [probe.score_one(h.astype(np.float64)) for h in r['hiddens']]
        t.tau_output = r['tau_output']
        label = _trajectory_label(r, conf_floor)
        if label is None:
            t.final_was_error = None
        else:
            t.final_was_error = bool(label == 1)
        tau_commit: Optional[int] = None
        for i, s in enumerate(t.step_scores):
            if s >= probe.commit_threshold:
                tau_commit = i
                break
        t.tau_commit = tau_commit
        traces.append(t)
    return traces

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--benchmark', default='gsm8k', choices=sorted(BENCHMARK_FILES))
    parser.add_argument('--n_train', type=int, default=150)
    parser.add_argument('--n_eval', type=int, default=150)
    parser.add_argument('--layer', type=int, default=-1, help='Residual-stream layer to capture; -1 = last. When --collect_layers is set, --layer selects which of the collected layers to run Phase B/C on.')
    parser.add_argument('--collect_layers', type=str, default='', help="Comma-separated signed layer indices to capture in ONE Phase A pass (e.g. '-1,-4,-8,-16,-24'). Phase B then runs on whichever of them --layer selects, while the full multi-layer capture is kept in the cache for cheap re-sweeps.")
    parser.add_argument('--commit_threshold', type=float, default=0.75, help='Fixed commitment threshold. Ignored when --calibration_fpr > 0 (calibrated threshold replaces it in-place).')
    parser.add_argument('--calibration_fpr', type=float, default=0.0, help='If > 0, set commit_threshold to the (1 - fpr) quantile of per-trajectory max probe scores over confident-CORRECT training trajectories. Typical values: 0.05 (5% FPR), 0.10 (10% FPR). 0 disables and keeps the fixed --commit_threshold.')
    parser.add_argument('--conf_floor', type=float, default=0.6)
    parser.add_argument('--l2', type=float, default=1.0, help='L2 ridge strength on z-scored features. Was 1e-2; raised because the probe is high-dim (~1.5k) and the training set is small (~200), so weak regularisation memorises trajectories.')
    parser.add_argument('--probe_holdout_frac', type=float, default=0.2, help='Fraction of train_records held out from probe fitting and reported as eval. Set 0 to disable.')
    parser.add_argument('--probe_pca_dim', type=int, default=64, help='If >0, project hidden states to this many PCA components (fit on probe_fit slice only) before training the probe. Required when hidden_dim (typ. >= 1024) is much larger than the number of confident step samples. 0 disables PCA.')
    parser.add_argument('--max_steps', type=int, default=8)
    parser.add_argument('--phaseA_batch_size', type=int, default=1, help='Batch size for Phase A generation. >1 enables the probe=None fast path (solve_phaseA_batch). Produces residual-state records bit-compatible with the serial path; results differ only in sampling order (same temperature / seed policy).')
    parser.add_argument('--train_first_k_steps', type=int, default=0, help="If >0, train probe only on the FIRST k steps of each trajectory. Forces the probe to learn predictive 'will this end wrong' signal rather than 'error is now visible'. 0 = all steps.")
    parser.add_argument('--train_last_k_steps', type=int, default=0, help='If >0, train probe only on the LAST k steps of each trajectory. Mutually exclusive with --train_first_k_steps. 0 = all steps.')
    parser.add_argument('--out', default='results/gate1_commitment_gap.json')
    parser.add_argument('--no_cache', action='store_true', help='Force a fresh Phase A (ignore and overwrite any existing cache entry). Useful after bumping the model, prompt, or sampling hyperparams without bumping _CACHE_VERSION.')
    args = parser.parse_args()
    from loci.reasoner import LOCIReasoner
    from loci.baselines.runner_base import MC_DATASETS
    is_mc = args.benchmark in MC_DATASETS
    cache_path = _phase_a_cache_path(args)
    cache_hit = not args.no_cache and os.path.exists(cache_path)
    if cache_hit:
        print(f'[gate1] Phase A cache HIT: {cache_path}', flush=True)
        print(f'[gate1]   skipping model load + trajectory collection', flush=True)
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        train_records = cached['train_records']
        eval_records = cached['eval_records']
        meta = cached.get('meta', {})
        print(f"[gate1]   cache_version={meta.get('cache_version')} written_at={meta.get('written_at')} (n_train={len(train_records)}, n_eval={len(eval_records)})", flush=True)
    else:
        if args.no_cache:
            print(f'[gate1] Phase A cache DISABLED (--no_cache)', flush=True)
        else:
            print(f'[gate1] Phase A cache MISS: {cache_path}', flush=True)
        print(f'[gate1] loading model {args.model}', flush=True)
        model, tok = _load_model(args.model)
        if args.collect_layers:
            collect_signed = _parse_layer_list(args.collect_layers)
            if args.layer not in collect_signed:
                collect_signed = [args.layer] + collect_signed
            print(f'[gate1]   multi-layer capture: layers={collect_signed}, primary={args.layer}', flush=True)
            commitment_layer_arg: Any = collect_signed
        else:
            commitment_layer_arg = args.layer
        reasoner = LOCIReasoner(model=model, tokenizer=tok, max_steps=args.max_steps, is_multiple_choice=is_mc, commitment_layer=commitment_layer_arg, capture_hidden_states=True)
        print(f'[gate1] Phase A: collecting {args.n_train} train + {args.n_eval} eval trajectories on {args.benchmark}', flush=True)
        train_examples = load_benchmark(args.benchmark, limit=args.n_train, offset=0)
        eval_examples = load_benchmark(args.benchmark, limit=args.n_eval, offset=args.n_train)
        bs = max(1, int(args.phaseA_batch_size))
        if bs > 1:
            print(f'[gate1]   Phase A batched: batch_size={bs}', flush=True)
        train_records = collect_trajectories(reasoner, train_examples, is_mc=is_mc, batch_size=bs)
        eval_records = collect_trajectories(reasoner, eval_examples, is_mc=is_mc, batch_size=bs)
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'train_records': train_records, 'eval_records': eval_records, 'meta': {'cache_version': _CACHE_VERSION, 'model': args.model, 'benchmark': args.benchmark, 'n_train': args.n_train, 'n_eval': args.n_eval, 'max_steps': args.max_steps, 'layer': args.layer, 'collect_layers': _parse_layer_list(args.collect_layers) if args.collect_layers else None, 'is_mc': is_mc, 'written_at': time.strftime('%Y-%m-%d %H:%M:%S')}}, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[gate1] Phase A cache WROTE: {cache_path}', flush=True)

    def _select_layer_hiddens(records: List[Dict[str, Any]], layer: int) -> None:
        for r in records:
            hbl = r.get('hiddens_by_layer')
            if not hbl:
                continue
            if layer not in hbl:
                raise KeyError(f'--layer={layer} not in Phase A cache collect_layers={sorted(hbl.keys())}; rerun Phase A with --collect_layers including {layer}')
            r['hiddens'] = list(hbl[layer])
    if train_records and 'hiddens_by_layer' in train_records[0]:
        _select_layer_hiddens(train_records, args.layer)
        _select_layer_hiddens(eval_records, args.layer)
        print(f"[gate1]   multi-layer cache: running Phase B/C on layer {args.layer} (cache has layers {sorted(train_records[0]['hiddens_by_layer'].keys())})", flush=True)
    train_conf = diagnose_confidence(train_records)
    eval_conf = diagnose_confidence(eval_records)
    print(f'[gate1]   train confidence quantiles: {train_conf}', flush=True)
    print(f'[gate1]   eval  confidence quantiles: {eval_conf}', flush=True)
    effective_conf_floor = auto_pick_conf_floor(train_records, requested=args.conf_floor, min_per_class=5)
    if effective_conf_floor != args.conf_floor:
        print(f'[gate1]   conf_floor auto-adjusted {args.conf_floor:.3f} -> {effective_conf_floor:.3f} (needed >= 5 samples per class)', flush=True)
    holdout_frac = max(0.0, min(0.5, float(args.probe_holdout_frac)))
    if holdout_frac > 0.0:
        err_recs = [r for r in train_records if not r['correct']]
        cor_recs = [r for r in train_records if r['correct']]
        n_h_err = int(round(holdout_frac * len(err_recs)))
        n_h_cor = int(round(holdout_frac * len(cor_recs)))
        if len(err_recs) >= 2:
            n_h_err = max(n_h_err, 1)
        if len(cor_recs) >= 2:
            n_h_cor = max(n_h_cor, 1)
        holdout_err = err_recs[-n_h_err:] if n_h_err > 0 else []
        holdout_cor = cor_recs[-n_h_cor:] if n_h_cor > 0 else []
        probe_holdout_records = holdout_err + holdout_cor
        fit_err = err_recs[:-n_h_err] if n_h_err > 0 else err_recs
        fit_cor = cor_recs[:-n_h_cor] if n_h_cor > 0 else cor_recs
        probe_fit_records = fit_err + fit_cor
    else:
        probe_fit_records = train_records
        probe_holdout_records = []
    pca_info: Dict[str, Any] = {'enabled': False}
    if args.probe_pca_dim > 0 and probe_fit_records:
        X_fit_rows: List[np.ndarray] = []
        for r in probe_fit_records:
            for h in r['hiddens']:
                X_fit_rows.append(h.astype(np.float64))
        if X_fit_rows:
            X_fit_mat = np.stack(X_fit_rows, axis=0)
            n_rows, hidden_dim = X_fit_mat.shape
            k = int(min(args.probe_pca_dim, n_rows, hidden_dim))
            basis, pca_mean = fit_pca(X_fit_mat, k)
            Xc = X_fit_mat - pca_mean
            total_var = float(np.sum(Xc * Xc))
            proj = Xc @ basis.T
            kept_var = float(np.sum(proj * proj))
            evr = kept_var / total_var if total_var > 0 else 0.0
            pca_info = {'enabled': True, 'hidden_dim': int(hidden_dim), 'k': int(k), 'n_fit_rows': int(n_rows), 'explained_variance_ratio': float(evr)}
            print(f'[gate1]   PCA: dim {hidden_dim} -> {k} (n_fit_rows={n_rows}, evr={evr:.3f})', flush=True)
            probe_fit_records = project_records(probe_fit_records, basis, pca_mean)
            probe_holdout_records = project_records(probe_holdout_records, basis, pca_mean)
            eval_records = project_records(eval_records, basis, pca_mean)
            _pca_for_probe = (basis, pca_mean)
        else:
            _pca_for_probe = None
    else:
        _pca_for_probe = None
    print(f'[gate1] Phase B: training commitment probe (fit={len(probe_fit_records)}, holdout={len(probe_holdout_records)}, l2={args.l2})', flush=True)
    if args.train_first_k_steps > 0 and args.train_last_k_steps > 0:
        raise ValueError('--train_first_k_steps and --train_last_k_steps are mutually exclusive; set exactly one (or neither).')
    probe, probe_stats = train_probe(probe_fit_records, conf_floor=effective_conf_floor, commit_threshold=args.commit_threshold, l2=args.l2, holdout_records=probe_holdout_records, train_first_k=args.train_first_k_steps, train_last_k=args.train_last_k_steps, calibration_fpr=args.calibration_fpr)
    print(f'[gate1]   {probe_stats}', flush=True)
    print(f'[gate1] Phase C: computing commitment gaps on eval split (commit_threshold={probe.commit_threshold:.4f})', flush=True)
    traces = compute_traces(eval_records, probe, conf_floor=effective_conf_floor)
    stats = summarize_commitment_gaps(traces)
    print(f'[gate1]   {stats}', flush=True)
    temporal = temporal_distribution_report(traces)
    print(f"[gate1]   temporal lock_rate err={temporal['lock_rate_err']:.3f} cor={temporal['lock_rate_cor']:.3f} | tau_commit p50 err={temporal['tau_commit_err']['p50']} cor={temporal['tau_commit_cor']['p50']} (MWU err<cor p={temporal['tau_mwu_err_less_than_cor_p']:.4g}) | delta p50 err={temporal['delta_err']['p50']} cor={temporal['delta_cor']['p50']} (MWU err>cor p={temporal['delta_mwu_err_greater_than_cor_p']:.4g})", flush=True)
    irrev = estimate_irreversibility(eval_records, traces, conf_floor=effective_conf_floor)
    print(f'[gate1]   irreversibility {irrev}', flush=True)
    gap_diff = stats.get('median_gap_error', float('nan')) - stats.get('median_gap_correct', float('nan'))
    p_value = stats.get('mannwhitney_p', float('nan'))
    gap_passed = bool(not np.isnan(gap_diff) and gap_diff >= 2.0 and (not np.isnan(p_value)) and (p_value < 0.001))
    lock = temporal.get('lock_contingency', {})
    chi2_p = float(lock.get('chi2_p_two_sided', float('nan')))
    lock_ratio = float(lock.get('lock_rate_ratio', float('nan')))
    lock_passed = bool(not np.isnan(chi2_p) and chi2_p < 0.001 and (not np.isnan(lock_ratio)) and (lock_ratio >= 2.0))
    passed = lock_passed or gap_passed
    report = {'args': vars(args), 'effective_conf_floor': float(effective_conf_floor), 'confidence_quantiles': {'train': train_conf, 'eval': eval_conf}, 'pca': pca_info, 'probe_stats': probe_stats, 'gap_stats': stats, 'temporal_distribution': temporal, 'irreversibility': irrev, 'go_condition': {'primary': {'lock_chi2_p_two_sided': None if np.isnan(chi2_p) else chi2_p, 'lock_rate_ratio': None if np.isnan(lock_ratio) else lock_ratio, 'threshold_chi2_p': 0.001, 'threshold_lock_rate_ratio': 2.0, 'passed': lock_passed}, 'legacy_gap': {'median_gap_diff': None if np.isnan(gap_diff) else float(gap_diff), 'p_value': None if np.isnan(p_value) else float(p_value), 'threshold_gap_diff': 2.0, 'threshold_p_value': 0.001, 'passed': gap_passed}, 'passed': passed}}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    if _pca_for_probe is not None:
        probe.pca_basis, probe.pca_mean = _pca_for_probe
    probe_path = os.path.splitext(args.out)[0] + '.probe.npz'
    try:
        probe.save(probe_path)
        print(f'[gate1] wrote {probe_path}', flush=True)
    except Exception as exc:
        print(f'[gate1] probe save failed: {exc}', flush=True)
    print(f'[gate1] wrote {args.out}', flush=True)
    print(f"[gate1] primary lock-contingency: chi2_p={chi2_p:.4g} lock_ratio={lock_ratio:.3g} -> {('PASS' if lock_passed else 'fail')} | legacy gap: diff={gap_diff:.3g} p={p_value:.4g} -> {('PASS' if gap_passed else 'fail')}", flush=True)
    if passed:
        which = 'lock-contingency' if lock_passed else 'legacy-gap'
        print(f'[gate1] GO — I1 phenomenon holds ({which}), proceed to Gate 2')
        return 0
    print('[gate1] NO-GO — neither lock-contingency nor gap criterion met')
    return 1
if __name__ == '__main__':
    sys.exit(main())
