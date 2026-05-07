from __future__ import annotations
import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loci.commitment import CommitmentHeadIdentifier, HeadScore
from loci.patching import HeadPatcher, PatchTrajectory
from loci.probe import CommitmentProbe
from scripts.train_probe import BENCHMARK_FILES, _load_model, collect_trajectories, load_benchmark

def _introspect_heads_layers(model) -> Tuple[int, int]:
    from loci.hooks import resolve_attn
    base = getattr(model, 'model', model)
    if hasattr(base, 'layers'):
        layers = base.layers
    elif hasattr(base, 'decoder') and hasattr(base.decoder, 'layers'):
        layers = base.decoder.layers
    else:
        raise AttributeError('cannot locate decoder layers for head introspection')
    n_layers = len(layers)
    first_attn, _ = resolve_attn(layers[0])
    num_heads = getattr(first_attn, 'num_heads', getattr(first_attn, 'num_attention_heads', None))
    if num_heads is None:
        cfg = getattr(model, 'config', None)
        num_heads = getattr(cfg, 'num_attention_heads', None)
    if num_heads is None:
        raise AttributeError('cannot read num_heads from attention submodule or model config')
    return (int(n_layers), int(num_heads))

def _score_and_filter(records: List[Dict[str, Any]], probe: CommitmentProbe, conf_floor: float) -> List[PatchTrajectory]:
    out: List[PatchTrajectory] = []
    for r in records:
        if r['confidence'] < conf_floor:
            continue
        if r['correct']:
            continue
        hiddens = r.get('hiddens', [])
        if not hiddens:
            continue
        tau_commit: Optional[int] = None
        for s, h in enumerate(hiddens):
            score = probe.score_one(np.asarray(h, dtype=np.float64))
            if score >= probe.commit_threshold:
                tau_commit = s
                break
        if tau_commit is None:
            continue
        steps = r.get('raw_outputs') or []
        if steps and tau_commit < len(steps):
            prefix = '\n'.join(steps[:tau_commit + 1])
        else:
            prefix = ''
        n_remaining = max(1, r.get('n_steps', len(hiddens)) - tau_commit - 1)
        out.append(PatchTrajectory(question=r['question'], prefix_text=prefix, gold=r['gold'], is_multiple_choice=bool(r.get('is_mc', False)), original_answer=str(r.get('predicted', '')), n_remaining_steps=int(n_remaining), tau_commit=int(tau_commit), meta={'confidence': r['confidence'], 'n_steps': r.get('n_steps', 0)}))
    return out

def summarise_heads(scores: List[HeadScore], top_k: int) -> Dict[str, float]:
    flips = np.asarray([s.flip_rate for s in scores], dtype=np.float64)
    top = flips[:top_k] if top_k > 0 else flips[:0]
    return {'n_heads_scored': int(flips.size), 'mean_flip_rate': float(flips.mean()) if flips.size else float('nan'), 'max_flip_rate': float(flips.max()) if flips.size else float('nan'), 'top_k': int(top_k), 'top_k_mean_flip_rate': float(top.mean()) if top.size else float('nan')}

def _introspect_head_dim(model, layers, layer_range: Tuple[int, int]) -> int:
    from loci.hooks import resolve_attn
    attn, _ = resolve_attn(layers[layer_range[0]])
    hd = getattr(attn, 'head_dim', None)
    if hd is not None:
        return int(hd)
    nh = getattr(attn, 'num_heads', getattr(attn, 'num_attention_heads', None))
    if nh is None:
        cfg = getattr(model, 'config', None)
        nh = getattr(cfg, 'num_attention_heads', None)
    W = attn.o_proj.weight
    return int(W.shape[1] // int(nh))

def compute_head_attributions(model, patcher: HeadPatcher, probe: CommitmentProbe, layer_range: Tuple[int, int], n_heads: int, token_position: str='last') -> np.ndarray:
    import torch
    from loci.hooks import resolve_attn
    from loci.hooks import probe_to_raw_direction
    base = getattr(model, 'model', model)
    if hasattr(base, 'layers'):
        layers = base.layers
    elif hasattr(base, 'decoder') and hasattr(base.decoder, 'layers'):
        layers = base.decoder.layers
    else:
        raise AttributeError('cannot locate decoder layers')
    lo, hi = layer_range
    n_layers_range = hi - lo
    head_dim = _introspect_head_dim(model, layers, layer_range)
    w_raw = probe_to_raw_direction(probe)
    w_hat = torch.as_tensor(w_raw, dtype=torch.float32).flatten()
    w_norm = float(w_hat.norm().item())
    if not np.isfinite(w_norm) or w_norm < 1e-10:
        raise ValueError(f'probe direction has zero norm ({w_norm})')
    w_hat = w_hat / w_norm
    p0 = next(model.parameters())
    w_hat = w_hat.to(device=p0.device, dtype=p0.dtype)
    captured: Dict[int, 'torch.Tensor'] = {}

    def make_hook(li: int):

        def pre_hook(module, inputs):
            x = inputs[0]
            captured[li] = x.detach()
        return pre_hook
    handles = []
    o_projs: Dict[int, 'torch.nn.Module'] = {}
    for li in range(lo, hi):
        attn, _ = resolve_attn(layers[li])
        o_projs[li] = attn.o_proj
        handles.append(attn.o_proj.register_forward_pre_hook(make_hook(li)))
    n_traj = len(patcher.trajectories)
    sum_attr = np.zeros((n_layers_range, n_heads), dtype=np.float64)
    counts = np.zeros(n_layers_range, dtype=np.int64)
    try:
        for traj_idx in range(n_traj):
            cached = patcher._get_cached_inputs(traj_idx)
            input_ids = cached['input_ids']
            attn_mask = cached['attention_mask']
            captured.clear()
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask)
            for li in range(lo, hi):
                x = captured.get(li)
                if x is None:
                    continue
                W = o_projs[li].weight
                if token_position == 'last':
                    x_t = x[:, -1, :]
                else:
                    x_t = x[:, -1, :]
                B = x_t.shape[0]
                x_heads = x_t.view(B, n_heads, head_dim)
                W_blk = W.view(-1, n_heads, head_dim).permute(1, 0, 2)
                w_proj = torch.einsum('d,hdc->hc', w_hat, W_blk)
                attr = torch.einsum('bhc,hc->bh', x_heads, w_proj)
                row = li - lo
                sum_attr[row] += attr.float().mean(dim=0).cpu().numpy()
                counts[row] += 1
    finally:
        for h in handles:
            h.remove()
    mean_attr = np.zeros_like(sum_attr)
    for row in range(n_layers_range):
        if counts[row] > 0:
            mean_attr[row] = sum_attr[row] / float(counts[row])
    return mean_attr

def attribution_ranking(mean_attr: np.ndarray, layer_range: Tuple[int, int]) -> List[Tuple[int, int, float]]:
    lo, _ = layer_range
    entries: List[Tuple[int, int, float]] = []
    n_layers_range, n_heads = mean_attr.shape
    for row in range(n_layers_range):
        for h in range(n_heads):
            entries.append((lo + row, h, float(mean_attr[row, h])))
    entries.sort(key=lambda x: abs(x[2]), reverse=True)
    return entries

def jaccard_topk(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> float:
    sa, sb = (set(a), set(b))
    if not sa and (not sb):
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def run_stability_split(scores_full: List[HeadScore], ident: CommitmentHeadIdentifier, patcher: HeadPatcher, split_ids: List[int], layer_range: Tuple[int, int]) -> Dict[str, float]:
    if not split_ids:
        return {'jaccard': float('nan'), 'n_split_trajectories': 0}
    ident_b = CommitmentHeadIdentifier(num_layers=ident.num_layers, num_heads=ident.num_heads, budget_fraction=ident.budget_fraction)
    scores_b = ident_b.score_all(split_ids, patcher.patch_fn, layer_range=layer_range)
    jac = ident.jaccard_stability(scores_b)
    return {'jaccard': float(jac), 'n_split_trajectories': int(len(split_ids)), 'h_commit_split': [{'layer': s.layer, 'head': s.head, 'flip_rate': s.flip_rate} for s in ident_b.select_h_commit()]}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3.5-9B')
    parser.add_argument('--benchmark', default='gsm8k', choices=sorted(BENCHMARK_FILES))
    parser.add_argument('--probe', required=True, help='Path to trained Gate-1 probe (.npz).')
    parser.add_argument('--n_collect', type=int, default=60, help='Number of problems to run through the reasoner to harvest confident-error trajectories.')
    parser.add_argument('--max_trajectories', type=int, default=20, help='Cap on confident-error trajectories fed into the patcher. Bigger = slower but more stable.')
    parser.add_argument('--max_replay_steps', type=int, default=4, help='Per-trajectory step budget for the replay.')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Per-step token budget for the replay.')
    parser.add_argument('--budget_fraction', type=float, default=0.05, help='Top fraction of heads to label H_commit.')
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--commit_threshold', type=float, default=None, help="Override probe's stored (calibrated) threshold. Omit to preserve whatever threshold was baked into the .npz at train time.")
    parser.add_argument('--conf_floor', type=float, default=0.4)
    parser.add_argument('--max_steps', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Data offset to avoid overlap with Gate 1 train set. Set to gate1_n_train to guarantee no leakage.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--layer_range', type=str, default=None, help="Override layer sweep, e.g. '10,14'. Default = middle third of the stack.")
    parser.add_argument('--collect_batch_size', type=int, default=1, help='Batch size for the Phase A trajectory collection via reasoner.solve_phaseA_batch(). >1 means probe-less fast path — big GPU util win on small models. Default 1 (legacy per-example).')
    parser.add_argument('--phase_d', action='store_true', help='Run Phase D attribution: per-head <h_out, w_commit> at the commitment-moment token, plus Jaccard against the Phase B flip-rate top-K ranking. Cheap (one forward pass per trajectory) and adds the necessary-condition complement to the sufficient-condition flip-rate ranking.')
    parser.add_argument('--out', default='results/gate2_commitment_heads.json')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    from loci.reasoner import LOCIReasoner
    from loci.baselines.runner_base import MC_DATASETS
    is_mc = args.benchmark in MC_DATASETS
    print(f'[gate2] loading probe {args.probe}', flush=True)
    probe = CommitmentProbe.load(args.probe)
    if args.commit_threshold is not None:
        probe.commit_threshold = float(args.commit_threshold)
        print(f'[gate2]   threshold override (CLI): {probe.commit_threshold:.4f}', flush=True)
    else:
        print(f'[gate2]   threshold from probe (calibrated): {probe.commit_threshold:.4f}', flush=True)
    print(f'[gate2] loading model {args.model}', flush=True)
    model, tok = _load_model(args.model)
    n_layers, n_heads = _introspect_heads_layers(model)
    print(f'[gate2]   architecture: {n_layers} layers x {n_heads} heads', flush=True)
    reasoner = LOCIReasoner(model=model, tokenizer=tok, max_steps=args.max_steps, is_multiple_choice=is_mc, commitment_layer=args.layer, capture_hidden_states=True)
    bs = max(1, int(args.collect_batch_size))
    print(f'[gate2] Phase A: harvesting confident-error trajectories (n_collect={args.n_collect}, batch_size={bs})', flush=True)
    examples = load_benchmark(args.benchmark, limit=args.n_collect, offset=args.offset)
    records = collect_trajectories(reasoner, examples, is_mc=is_mc, batch_size=bs)
    for r in records:
        r['is_mc'] = is_mc
    patch_trajs = _score_and_filter(records, probe, args.conf_floor)
    print(f'[gate2]   confident-error trajectories with tau_commit set: {len(patch_trajs)}', flush=True)
    if len(patch_trajs) < 2:
        print('[gate2] NO-GO — not enough confident-error trajectories to patch; lower commit_threshold or increase n_collect', flush=True)
        report = {'args': vars(args), 'n_confident_errors': len(patch_trajs), 'passed': False, 'reason': 'too few confident-error trajectories'}
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        return 1
    if len(patch_trajs) > args.max_trajectories:
        patch_trajs = patch_trajs[:args.max_trajectories]
    print(f'[gate2]   patching on {len(patch_trajs)} trajectories', flush=True)
    patcher = HeadPatcher(model=model, tokenizer=tok, trajectories=patch_trajs, max_new_tokens=args.max_new_tokens, max_replay_steps=args.max_replay_steps)
    clean_stats = patcher.clean_stats()
    print(f'[gate2]   clean replay: {clean_stats}', flush=True)
    ident = CommitmentHeadIdentifier(num_layers=n_layers, num_heads=n_heads, budget_fraction=float(args.budget_fraction))
    if args.layer_range:
        lo_str, hi_str = args.layer_range.split(',')
        lo, hi = (int(lo_str), int(hi_str))
    else:
        lo = n_layers // 3
        hi = 2 * n_layers // 3 + 1
    layer_range = (lo, hi)
    print(f'[gate2] Phase B: scoring heads over layer range {layer_range}', flush=True)
    all_ids = list(range(len(patch_trajs)))
    scores = ident.score_all(all_ids, patcher.patch_fn, layer_range=layer_range)
    h_commit = ident.select_h_commit()
    top_k = max(1, int(round(ident.budget_fraction * n_layers * n_heads)))
    top_k = min(top_k, len(scores))
    full_summary = summarise_heads(scores, top_k)
    print(f'[gate2]   full ranking: {full_summary}', flush=True)
    print(f'[gate2]   H_commit ({len(h_commit)} heads): {[(s.layer, s.head, round(s.flip_rate, 3)) for s in h_commit]}', flush=True)
    if len(all_ids) >= 4:
        mid = len(all_ids) // 2
        split_ids = all_ids[mid:]
    else:
        split_ids = all_ids
    print(f'[gate2] Phase C: T3 stability via disjoint split (n={len(split_ids)})', flush=True)
    stability = run_stability_split(scores, ident, patcher, split_ids, layer_range=layer_range)
    print(f'[gate2]   stability: {stability}', flush=True)
    phase_d_report: Optional[Dict[str, Any]] = None
    if args.phase_d:
        print(f'[gate2] Phase D: attribution on {len(patch_trajs)} trajectories (layer range {layer_range})', flush=True)
        mean_attr = compute_head_attributions(model=model, patcher=patcher, probe=probe, layer_range=layer_range, n_heads=n_heads, token_position='last')
        attr_ranked = attribution_ranking(mean_attr, layer_range)
        top_k_attr = min(top_k, len(attr_ranked))
        h_attr = [(l, h) for l, h, _ in attr_ranked[:top_k_attr]]
        h_flip = [(s.layer, s.head) for s in scores[:top_k_attr]]
        jac_attr_flip = jaccard_topk(h_attr, h_flip)
        h_commit_set = [(s.layer, s.head) for s in h_commit]
        jac_attr_commit = jaccard_topk(h_attr, h_commit_set)
        phase_d_report = {'token_position': 'last', 'top_k': int(top_k_attr), 'h_attr_topk': [{'layer': l, 'head': h, 'attribution': a} for l, h, a in attr_ranked[:top_k_attr]], 'jaccard_attribution_vs_flip': float(jac_attr_flip), 'jaccard_attribution_vs_h_commit': float(jac_attr_commit), 'attribution_abs_mean': float(np.mean(np.abs(mean_attr))), 'attribution_abs_max': float(np.max(np.abs(mean_attr)))}
        print(f'[gate2]   Phase D: Jaccard(attr, flip_top{top_k_attr})={jac_attr_flip:.3f} | Jaccard(attr, H_commit)={jac_attr_commit:.3f}', flush=True)
        print(f'[gate2]   top attribution heads: {[(l, h, round(a, 4)) for l, h, a in attr_ranked[:min(5, top_k_attr)]]}', flush=True)
    top_mean = full_summary.get('top_k_mean_flip_rate', float('nan'))
    jac = stability.get('jaccard', float('nan'))
    passed = bool(not np.isnan(top_mean) and top_mean >= 0.15 and (not np.isnan(jac)) and (jac >= 0.4))
    report = {'args': vars(args), 'architecture': {'num_layers': n_layers, 'num_heads': n_heads}, 'layer_range': list(layer_range), 'n_confident_errors': len(patch_trajs), 'clean_replay': clean_stats, 'full_summary': full_summary, 'h_commit': [{'layer': s.layer, 'head': s.head, 'flip_rate': s.flip_rate} for s in h_commit], 'stability': stability, 'phase_d': phase_d_report, 'go_condition': {'top_k_mean_flip_rate': top_mean, 'jaccard': jac, 'threshold_flip_rate': 0.15, 'threshold_jaccard': 0.4, 'passed': passed}}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    spec_path = os.path.splitext(args.out)[0] + '.h_commit.json'
    with open(spec_path, 'w', encoding='utf-8') as f:
        json.dump([[s.layer, s.head] for s in h_commit], f)
    print(f'[gate2] wrote {args.out}', flush=True)
    print(f'[gate2] wrote {spec_path}', flush=True)
    if passed:
        print('[gate2] GO — I2 ranking is stable and selective')
        return 0
    print('[gate2] NO-GO — H_commit too weak or unstable for the oral narrative')
    return 1
if __name__ == '__main__':
    sys.exit(main())
