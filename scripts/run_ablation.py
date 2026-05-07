from __future__ import annotations
import argparse
import os
import subprocess
import sys
ABLATION_METHODS = ['loci_base', 'loci_probe_only', 'loci_random_heads', 'loci_fixed_window', 'loci']

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--n_candidates', type=int, default=None)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--probe_path', default=None)
    parser.add_argument('--h_commit_path', default=None)
    parser.add_argument('--fixed_k', type=int, default=3, help='constant k for loci_fixed_window arm')
    parser.add_argument('--loci_preset', choices=['conservative', 'aggressive'], default=None, help='Forwarded to run_baseline_suite.py; keeps the 5 ablation arms aligned with whatever knob bundle the headline table used.')
    args = parser.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runner_script = os.path.join(here, 'experiments', 'run_baseline_suite.py')
    py = sys.executable
    cmd = [py, runner_script, '--model', args.model, '--datasets', args.datasets, '--methods', ','.join(ABLATION_METHODS), '--seeds', args.seeds, '--limit', str(args.limit), '--out_dir', args.out_dir, '--loci_fixed_k', str(args.fixed_k)]
    if args.max_steps is not None:
        cmd += ['--max_steps', str(args.max_steps)]
    if args.max_new_tokens is not None:
        cmd += ['--max_new_tokens', str(args.max_new_tokens)]
    if args.n_candidates is not None:
        cmd += ['--n_candidates', str(args.n_candidates)]
    if args.probe_path is not None:
        cmd += ['--loci_probe_path', args.probe_path]
    if args.h_commit_path is not None:
        cmd += ['--loci_h_commit_path', args.h_commit_path]
    if args.loci_preset is not None:
        cmd += ['--loci_preset', args.loci_preset]
    print('=' * 70)
    print(f'Ablation Study ABLATION — 5 LOCI arms × {args.datasets}')
    print('=' * 70)
    return subprocess.call(cmd)
if __name__ == '__main__':
    sys.exit(main())
