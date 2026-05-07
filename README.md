# LOCI

**LOCI**: **L**ock-in **O**bservation, **C**ausal localization, and **I**ntervention —
inference-time reasoning-error repair for frozen LLMs via residual-stream
intervention.

LOCI detects, localizes, and repairs reasoning *lock-in* (committed errors that
form in the residual stream multiple steps before they surface as tokens),
within a single inference trajectory and without modifying model weights.

## Method overview

Three components, trained / built once and applied at inference time:

| Component | What it does | Script |
|---|---|---|
| **I1** Commitment trajectory probe | Linear probe over the residual stream that fires when a confident error commits | `scripts/train_probe.py` |
| **I2** Per-query causal head set $\mathcal{H}_{\mathrm{commit}}$ | Commitment-guided cross-step soft ablation localizes the sparse head set causally driving lock-in | `scripts/localize_heads.py` |
| **I3** Donor-bank residual repair | Retrieval-based residual patching from correct-trajectory donors within a closed-form optimal window $w^\star$ | `scripts/intervention.py` |

## Installation

```bash
git clone <repo>
cd LOCI
pip install -r requirements.txt
```

## Folder layout

```
LOCI/
├── loci/                      # main package
│   ├── reasoner.py            # commitment-aware step-level reasoner
│   ├── probe.py               # I1 — linear residual-stream probe
│   ├── commitment.py          # commit-step / threshold-crossing logic
│   ├── patching.py            # I3 — residual patch ops
│   ├── hooks.py               # forward-hook utilities
│   ├── metrics.py             # accuracy / token / answer grading helpers
│   └── baselines/             # baseline runners (zero-shot / CoT / BoN / …)
│       └── loci_runner.py     # LOCI registered as a baseline (5 arms)
├── scripts/
│   ├── train_probe.py         # Gate 1 — fit commitment trajectory probe
│   ├── localize_heads.py      # Gate 2 — identify H_commit per query
│   ├── intervention.py        # Gate 3 — end-to-end LOCI evaluation
│   ├── run_baselines.py       # run any registered baseline (incl. LOCI)
│   ├── run_ablation.py        # 5-arm LOCI ablation study
│   └── aggregate.py           # aggregate cases.jsonl → metrics
├── data/                      # datasets (you provide)
├── examples/
├── requirements.txt
└── LICENSE
```

## Quick start

### 1. Train the commitment trajectory probe (Gate 1)

```bash
python scripts/train_probe.py \
    --model /path/to/Qwen3-4B \
    --benchmark sample_2502 \
    --layer 16 \
    --calibration_fpr 0.2 \
    --out results/probe_qwen3_4b.npz
```

### 2. Localize the causal head set $\mathcal{H}_{\mathrm{commit}}$ (Gate 2)

```bash
python scripts/localize_heads.py \
    --model /path/to/Qwen3-4B \
    --probe results/probe_qwen3_4b.npz \
    --benchmark sample_2502 \
    --layer_range 8 24 \
    --budget_fraction 0.05 \
    --out results/h_commit_qwen3_4b.json
```

### 3. End-to-end LOCI evaluation (Gate 3)

```bash
python scripts/intervention.py \
    --model /path/to/Qwen3-4B \
    --benchmark sample_2502 \
    --probe results/probe_qwen3_4b.npz \
    --h_commit results/h_commit_qwen3_4b.json \
    --retrieval_index results/donor_bank.npz \
    --intervention_mode retrieval_patch \
    --n_eval 200 --max_steps 8 --max_window 6 \
    --out results/loci_eval.json
```

By default Gate 3 evaluates **all three arms** (`arm_base`, `arm_capture`,
`arm_break`); use `--skip_arms` only if you want to omit one explicitly.

### 4. Run baselines for comparison

```bash
python scripts/run_baselines.py \
    --model /path/to/Qwen3-4B \
    --datasets gsm8k math_500 olympiadbench gpqa_diamond \
    --methods zero_shot cot bon prm_bon math_shepherd \
              snell_optimal_tts phi_decoding loci \
    --seeds 42 0 1 \
    --batch_size 16 \
    --loci_probe_path results/probe_qwen3_4b.npz \
    --loci_h_commit_path results/h_commit_qwen3_4b.json \
    --out_dir res/main_run
```

### 5. Ablation study

```bash
python scripts/run_ablation.py \
    --model /path/to/Qwen3-4B \
    --datasets sample_biased_1280 \
    --probe results/probe_qwen3_4b.npz \
    --h_commit results/h_commit_qwen3_4b.json \
    --out_dir res/ablation
```

### 6. Aggregate results

```bash
python scripts/aggregate.py \
    res/main_run/Qwen3-4B/*/cases.jsonl \
    --out results/main_summary.json
```

## Citation

```bibtex
@article{loci2026,
  title   = {LOCI: Reasoning Errors Are Decided Before They Are Spoken},
  author  = {...},
  year    = {2026},
}
```

## License

MIT — see `LICENSE`.
