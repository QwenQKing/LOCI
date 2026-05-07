from __future__ import annotations
import argparse
import inspect
import json
import os
import random
import sys
import time
from typing import Any, Dict, List
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loci.baselines import BaselineCase, CaseWriter, already_done_ids, DEFAULT_FAIR_CONFIG, FairConfig, get_registry
LOCI_KWARGS: Dict[str, Any] = {}
LOCI_PRESETS: Dict[str, Dict[str, float]] = {'conservative': {'commit_threshold': 0.55, 'target_residual': 0.1, 'max_window': 4}, 'aggressive': {'commit_threshold': 0.5, 'target_residual': 0.08, 'max_window': 5}}
UNIFIED_DIR_REL = os.path.join('data', 'unified')

def load_dataset(name: str, root: str) -> List[Dict[str, Any]]:
    path = os.path.join(root, UNIFIED_DIR_REL, f'{name}.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'unified dataset not found: {path} — run data/normalize.py first')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_backbone(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f'[suite] loading model {model_path}', flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    attn_impl = os.environ.get('LOCI_ATTN_IMPL', '').strip()
    if not attn_impl:
        try:
            import flash_attn
            attn_impl = 'flash_attention_2'
        except Exception:
            attn_impl = 'sdpa'
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True, attn_implementation=attn_impl)
    except (TypeError, ValueError) as e:
        print(f'[suite] attn_impl={attn_impl} rejected ({e}); retrying without attn_implementation', flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto', trust_remote_code=True)
    print(f'[suite] loaded with attn_implementation={attn_impl}', flush=True)
    model.eval()
    want_static = os.environ.get('LOCI_STATIC_CACHE', '1') == '1'
    want_compile = os.environ.get('LOCI_TORCH_COMPILE', '0') == '1'
    if want_static:
        try:
            model.generation_config.cache_implementation = 'static'
            print('[suite] generation_config.cache_implementation=static', flush=True)
        except Exception as e:
            print(f'[suite] static cache rejected: {e}', flush=True)
    if want_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
            print('[suite] torch.compile(mode=reduce-overhead) applied', flush=True)
        except Exception as e:
            print(f'[suite] torch.compile failed: {e}', flush=True)
    return (model, tok)

def _model_tag(model_path: str) -> str:
    base = os.path.basename(model_path.rstrip('/\\'))
    return base.replace('___', '_')

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def run_one(method_name: str, runner_cls, examples: List[Dict[str, Any]], out_dir: str, dataset_name: str, model_tag: str, seed: int, limit: int, model, tokenizer, config: FairConfig, batch_size: int=1, shard_id: int=0, num_shards: int=1) -> Dict[str, Any]:
    out_subdir = os.path.join(out_dir, model_tag, dataset_name)
    os.makedirs(out_subdir, exist_ok=True)
    shard_suffix = f'.shard{shard_id}of{num_shards}' if num_shards > 1 else ''
    cases_path = os.path.join(out_subdir, f'{method_name}.seed{seed}{shard_suffix}.cases.jsonl')
    summary_path = os.path.join(out_subdir, f'{method_name}.seed{seed}{shard_suffix}.summary.json')
    if num_shards > 1:
        examples = [ex for i, ex in enumerate(examples) if i % num_shards == shard_id]
        print(f'  [shard {shard_id + 1}/{num_shards}] {len(examples)} examples after sharding', flush=True)
    done = already_done_ids(cases_path)
    if done:
        print(f'  [{method_name}/{dataset_name}/seed{seed}] resuming ({len(done)} already done)', flush=True)
    extra_kwargs: Dict[str, Any] = {}
    cls_init = inspect.signature(runner_cls.__init__).parameters
    for k in ('commitment_layer', 'commit_threshold', 'max_window', 'target_residual', 'fixed_k', 'probe_path', 'h_commit_path', 'vector_path', 'layer', 'coef', 'layer_start', 'layer_end'):
        if k in cls_init and k in LOCI_KWARGS:
            extra_kwargs[k] = LOCI_KWARGS[k]
    runner = runner_cls(model=model, tokenizer=tokenizer, config=config, model_name=model_tag, **extra_kwargs)
    n_correct = 0
    n_total = 0
    sum_tokens_policy = 0
    sum_tokens_critic = 0
    sum_latency = 0.0
    sum_calls = 0
    bs = max(1, int(batch_size))
    with CaseWriter(cases_path) as writer:
        slice_examples = examples[:limit] if limit and limit > 0 else examples
        todo = []
        for i, ex in enumerate(slice_examples):
            ex_id = str(ex.get('id', f'{dataset_name}/{i}'))
            if ex_id in done:
                continue
            todo.append((i, ex))

        def _prompt_len(ex):
            return len(str(ex.get('question') or ex.get('prompt') or ''))
        todo.sort(key=lambda ie: _prompt_len(ie[1]), reverse=True)
        import torch as _torch
        cur_bs = bs
        cursor = 0
        total_n = len(slice_examples)
        while cursor < len(todo):
            take = min(cur_bs, len(todo) - cursor)
            chunk = todo[cursor:cursor + take]
            try:
                t0 = time.time()
                if take == 1:
                    cases_chunk = [runner.solve_and_pack(chunk[0][1])]
                else:
                    cases_chunk = runner.solve_and_pack_batch([ex for _, ex in chunk])
                batch_dt = time.time() - t0
            except _torch.cuda.OutOfMemoryError:
                _torch.cuda.empty_cache()
                if cur_bs <= 1:
                    print(f'  [OOM at bs=1] cursor={cursor} — cannot shrink further, re-raising', flush=True)
                    raise
                new_bs = max(1, cur_bs // 2)
                print(f'  [OOM] bs {cur_bs} -> {new_bs} (cursor={cursor}, take={take}); cache cleared, retrying', flush=True)
                cur_bs = new_bs
                continue
            batch_tok = sum((int(c.n_output_tokens) for c in cases_chunk))
            for (ci, _ex), case in zip(chunk, cases_chunk):
                writer.write(case)
                n_total += 1
                n_correct += int(case.correct)
                sum_tokens_policy += int(case.n_policy_output_tokens)
                sum_tokens_critic += int(case.n_critic_output_tokens)
                sum_latency += float(case.latency_s)
                sum_calls += int(case.n_calls)
                err_str = f' err={case.error[:120]}' if case.error else ''
                print(f'  [{method_name}/{dataset_name}/{ci + 1}/{total_n}] correct={case.correct} tok={case.n_policy_output_tokens} calls={case.n_calls} dt={case.latency_s:.1f}s{err_str}', flush=True)
            if len(chunk) > 1 and batch_tok > 0:
                print(f'    └─ batch={len(chunk)} wall={batch_dt:.1f}s eff={batch_dt * 1000 / batch_tok:.1f} ms/tok', flush=True)
            cursor += take
            try:
                import gc as _gc
                _gc.collect()
                _torch.cuda.empty_cache()
                _torch.cuda.ipc_collect()
            except Exception:
                pass
    all_lines = list(_iter_jsonl(cases_path))
    n_all = len(all_lines)
    n_correct_all = sum((int(c.get('correct', False)) for c in all_lines))
    summary = {'method': method_name, 'dataset': dataset_name, 'model': model_tag, 'seed': seed, 'n_examples': n_all, 'accuracy': n_correct_all / n_all if n_all else float('nan'), 'n_correct': n_correct_all, 'mean_n_policy_output_tokens': float(np.mean([c.get('n_policy_output_tokens', 0) for c in all_lines])) if all_lines else 0.0, 'mean_n_critic_output_tokens': float(np.mean([c.get('n_critic_output_tokens', 0) for c in all_lines])) if all_lines else 0.0, 'mean_latency_s': float(np.mean([c.get('latency_s', 0.0) for c in all_lines])) if all_lines else 0.0, 'mean_n_calls': float(np.mean([c.get('n_calls', 0) for c in all_lines])) if all_lines else 0.0, 'config': config.to_dict()}
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return summary

def _iter_jsonl(path: str):
    if not os.path.isfile(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path or HF id of the policy backbone.')
    parser.add_argument('--datasets', required=True, help='Comma-separated dataset names under data/unified/.')
    parser.add_argument('--methods', required=True, help='Comma-separated baseline names from the registry.')
    parser.add_argument('--seeds', default='42', help='Comma-separated seeds.')
    parser.add_argument('--limit', type=int, default=0, help='Max examples per dataset; 0 = full set.')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--n_candidates', type=int, default=None)
    parser.add_argument('--loci_layer', type=int, default=None, help='commitment_layer override (EXP-5b probe layer sweep)')
    parser.add_argument('--loci_threshold', type=float, default=None, help='commit_threshold override (EXP-5c)')
    parser.add_argument('--loci_max_window', type=int, default=None, help='max_window override (EXP-5a k* sensitivity)')
    parser.add_argument('--loci_target_residual', type=float, default=None)
    parser.add_argument('--loci_fixed_k', type=int, default=None, help='fixed k for loci_fixed_window arm (Ablation Study)')
    parser.add_argument('--loci_probe_path', type=str, default=None, help='override Gate-1 probe.npz path')
    parser.add_argument('--loci_h_commit_path', type=str, default=None, help='override Gate-2 h_commit.json path')
    parser.add_argument('--loci_preset', choices=['conservative', 'aggressive'], default=None, help='Expand into (commit_threshold, target_residual, max_window). conservative = (0.55, 0.10, 4) — current defaults. aggressive = (0.50, 0.08, 5) — more commitment breaks, longer ablation window, used for the paper headline table. Explicit --loci_threshold / --loci_max_window / --loci_target_residual flags override the preset.')
    parser.add_argument('--caa_vector_path', type=str, default=None, help='path to caa_vectors_<tag>.npz from calibrate_caa_repe.py')
    parser.add_argument('--caa_layer', type=int, default=None, help='single layer index to inject CAA steering vector')
    parser.add_argument('--caa_coef', type=float, default=2.0, help='CAA steering coefficient (default 2.0)')
    parser.add_argument('--repe_vector_path', type=str, default=None, help='path to repe_vectors_<tag>.npz from calibrate_caa_repe.py')
    parser.add_argument('--repe_layer_start', type=int, default=None, help='lowest layer in RepE injection range (inclusive)')
    parser.add_argument('--repe_layer_end', type=int, default=None, help='highest layer in RepE injection range (exclusive)')
    parser.add_argument('--repe_coef', type=float, default=1.0, help='RepE coefficient (default 1.0; honesty paper used 8.0 — too aggressive for math)')
    parser.add_argument('--out_dir', required=True, help='Output root (e.g. res/baseline_suite_smoke).')
    parser.add_argument('--batch_size', type=int, default=8, help='Generation batch size. >1 packs N examples into one generate() call for runners that override solve_batch (zero_shot, cot, bon, prm_bon). Multi-step runners fall back to per-example regardless of this flag.')
    parser.add_argument('--shard_id', type=int, default=0, help='Shard index in [0, num_shards). Each shard processes examples where idx %% num_shards == shard_id. Output filename gets a .shard<i>of<N> suffix when num_shards > 1.')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards (1 = no sharding). Useful for splitting a single (method, seed) across multiple GPUs.')
    args = parser.parse_args()
    if args.num_shards < 1 or args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise SystemExit(f'invalid shard config: shard_id={args.shard_id} num_shards={args.num_shards}; require 0 <= shard_id < num_shards')
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    registry = get_registry()
    bad = [m for m in methods if m not in registry]
    if bad:
        raise SystemExit(f'unknown methods: {bad}; available={list(registry)}')
    cfg_kwargs = {}
    if args.temperature is not None:
        cfg_kwargs['temperature'] = args.temperature
    if args.max_new_tokens is not None:
        cfg_kwargs['max_new_tokens'] = args.max_new_tokens
    if args.max_steps is not None:
        cfg_kwargs['max_steps'] = args.max_steps
    if args.n_candidates is not None:
        cfg_kwargs['n_candidates'] = args.n_candidates
    config = DEFAULT_FAIR_CONFIG.with_overrides(**cfg_kwargs) if cfg_kwargs else DEFAULT_FAIR_CONFIG
    global LOCI_KWARGS
    LOCI_KWARGS = {}
    if args.loci_preset is not None:
        preset = LOCI_PRESETS[args.loci_preset]
        LOCI_KWARGS['commit_threshold'] = preset['commit_threshold']
        LOCI_KWARGS['target_residual'] = preset['target_residual']
        LOCI_KWARGS['max_window'] = preset['max_window']
    if args.loci_layer is not None:
        LOCI_KWARGS['commitment_layer'] = args.loci_layer
    if args.loci_threshold is not None:
        LOCI_KWARGS['commit_threshold'] = args.loci_threshold
    if args.loci_max_window is not None:
        LOCI_KWARGS['max_window'] = args.loci_max_window
    if args.loci_target_residual is not None:
        LOCI_KWARGS['target_residual'] = args.loci_target_residual
    if args.loci_fixed_k is not None:
        LOCI_KWARGS['fixed_k'] = args.loci_fixed_k
    if args.loci_probe_path is not None:
        LOCI_KWARGS['probe_path'] = args.loci_probe_path
    if args.loci_h_commit_path is not None:
        LOCI_KWARGS['h_commit_path'] = args.loci_h_commit_path
    if args.caa_vector_path is not None:
        LOCI_KWARGS['vector_path'] = args.caa_vector_path
    if args.caa_layer is not None:
        LOCI_KWARGS['layer'] = args.caa_layer
    if args.caa_coef is not None and any((m == 'caa' for m in methods)):
        LOCI_KWARGS['coef'] = args.caa_coef
    if args.repe_vector_path is not None:
        LOCI_KWARGS['vector_path'] = args.repe_vector_path
    if args.repe_layer_start is not None:
        LOCI_KWARGS['layer_start'] = args.repe_layer_start
    if args.repe_layer_end is not None:
        LOCI_KWARGS['layer_end'] = args.repe_layer_end
    if args.repe_coef is not None and any((m == 'repe' for m in methods)):
        LOCI_KWARGS['coef'] = args.repe_coef
    print('=' * 60)
    print('HELM / LOCI — baseline suite')
    print(f'  model:     {args.model}')
    print(f'  datasets:  {datasets}')
    print(f'  methods:   {methods}')
    print(f'  seeds:     {seeds}')
    print(f'  limit:     {args.limit}')
    print(f'  out_dir:   {args.out_dir}')
    if args.num_shards > 1:
        print(f'  shard:     {args.shard_id + 1}/{args.num_shards}')
    print(f'  config:    {config.to_dict()}')
    if args.loci_preset is not None:
        print(f'  loci_preset: {args.loci_preset} -> {LOCI_PRESETS[args.loci_preset]}')
    if LOCI_KWARGS:
        print(f'  loci_kwargs (effective): {LOCI_KWARGS}')
    print('=' * 60)
    model_tag = _model_tag(args.model)
    model, tokenizer = load_backbone(args.model)
    all_summaries: List[Dict[str, Any]] = []
    t_start = time.time()
    for ds in datasets:
        examples = load_dataset(ds, here)
        for seed in seeds:
            for method in methods:
                _seed_everything(seed)
                runner_cls = registry[method]
                print(f'\n--- {method} × {ds} × seed{seed} ---', flush=True)
                try:
                    import gc
                    import torch as _torch
                    gc.collect()
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                        _torch.cuda.ipc_collect()
                except Exception as e:
                    print(f'  [cleanup] VRAM reset skipped: {e}', flush=True)
                summary = run_one(method_name=method, runner_cls=runner_cls, examples=examples, out_dir=args.out_dir, dataset_name=ds, model_tag=model_tag, seed=seed, limit=args.limit, model=model, tokenizer=tokenizer, config=config, batch_size=args.batch_size, shard_id=args.shard_id, num_shards=args.num_shards)
                all_summaries.append(summary)
                print(f"  -> acc={summary['accuracy']:.3f} tok={summary['mean_n_policy_output_tokens']:.0f} lat={summary['mean_latency_s']:.1f}s", flush=True)
    suite_path = os.path.join(args.out_dir, 'suite_summary.json')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(suite_path, 'w', encoding='utf-8') as f:
        json.dump({'suite_runtime_s': time.time() - t_start, 'model': args.model, 'datasets': datasets, 'methods': methods, 'seeds': seeds, 'limit': args.limit, 'config': config.to_dict(), 'summaries': all_summaries}, f, indent=2)
    print(f'\nDONE in {time.time() - t_start:.1f}s; suite summary -> {suite_path}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
