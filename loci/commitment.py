from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from loci.probe import CommitmentProbe

@dataclass
class CommitmentTrace:
    step_scores: List[float] = field(default_factory=list)
    tau_commit: Optional[int] = None
    tau_output: Optional[int] = None
    final_was_error: Optional[bool] = None

    @property
    def commitment_gap(self) -> Optional[int]:
        if self.tau_commit is None or self.tau_output is None:
            return None
        return max(0, self.tau_output - self.tau_commit)

class CommitmentGapTracker:

    def __init__(self, probe: CommitmentProbe):
        self.probe = probe
        self.trace = CommitmentTrace()
        self._tau_commit_locked = False

    def record_step(self, step_idx: int, hidden_state: np.ndarray) -> float:
        assert step_idx >= 0, f'step_idx must be non-negative, got {step_idx}'
        h = np.asarray(hidden_state, dtype=np.float64).reshape(-1)
        score = self.probe.score_one(h)
        self.trace.step_scores.append(score)
        if not self._tau_commit_locked and score >= self.probe.commit_threshold:
            self.trace.tau_commit = step_idx
            self._tau_commit_locked = True
        return score

    def current_score(self) -> float:
        return self.trace.step_scores[-1] if self.trace.step_scores else 0.0

    def is_currently_committed(self) -> bool:
        return self.current_score() >= self.probe.commit_threshold

    def finalize(self, tau_output: int, was_error: bool) -> CommitmentTrace:
        self.trace.tau_output = int(tau_output)
        self.trace.final_was_error = bool(was_error)
        return self.trace

    def reset(self) -> None:
        self.trace = CommitmentTrace()
        self._tau_commit_locked = False

def summarize_commitment_gaps(traces: Sequence[CommitmentTrace]) -> Dict[str, float]:
    err_total = [t for t in traces if t.final_was_error is True]
    cor_total = [t for t in traces if t.final_was_error is False]
    err = [t.commitment_gap for t in err_total if t.commitment_gap is not None]
    cor = [t.commitment_gap for t in cor_total if t.commitment_gap is not None]
    out: Dict[str, float] = {'n_error_total': float(len(err_total)), 'n_correct_total': float(len(cor_total)), 'n_error': float(len(err)), 'n_correct': float(len(cor)), 'frac_locked_error': len(err) / len(err_total) if err_total else float('nan'), 'frac_locked_correct': len(cor) / len(cor_total) if cor_total else float('nan'), 'median_gap_error': float(np.median(err)) if err else float('nan'), 'median_gap_correct': float(np.median(cor)) if cor else float('nan')}
    if err and cor:
        try:
            from scipy.stats import mannwhitneyu
            u, p = mannwhitneyu(err, cor, alternative='greater')
            out['mannwhitney_u'] = float(u)
            out['mannwhitney_p'] = float(p)
            _, p_less = mannwhitneyu(err, cor, alternative='less')
            out['mannwhitney_p_reversed'] = float(p_less)
            _, p_two = mannwhitneyu(err, cor, alternative='two-sided')
            out['mannwhitney_p_two_sided'] = float(p_two)
        except Exception:
            combined = sorted([(v, 0) for v in err] + [(v, 1) for v in cor])
            ranks_err = sum((i + 1 for i, (_, g) in enumerate(combined) if g == 0))
            n1, n2 = (len(err), len(cor))
            u = ranks_err - n1 * (n1 + 1) / 2
            out['mannwhitney_u'] = float(u)
            out['mannwhitney_p'] = float('nan')
            out['mannwhitney_p_reversed'] = float('nan')
            out['mannwhitney_p_two_sided'] = float('nan')
    return out
PatchFn = Callable[[int, int, int], float]
'\nSignature of an externally supplied patching function.\n\nArgs:\n    trajectory_id: which confident-error trajectory to patch\n    layer_idx:     which transformer block\n    head_idx:      which attention head inside that block\n\nReturns:\n    flip_rate: probability that the final answer flips from wrong to\n               right when the specified (layer, head) carries its\n               clean-run state across the step boundary in place of\n               the confident-error run state.\n\nThe ``CommitmentHeadIdentifier`` stays agnostic to how the flip rate is\ncomputed; heavy torch + patching code lives in ``loci.patching``.\n'

@dataclass
class HeadScore:
    layer: int
    head: int
    flip_rate: float

class CommitmentHeadIdentifier:

    def __init__(self, num_layers: int, num_heads: int, budget_fraction: float=0.05):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.budget_fraction = budget_fraction
        self.scores: List[HeadScore] = []

    def score_all(self, trajectory_ids: Sequence[int], patch_fn: PatchFn, layer_range: Optional[Tuple[int, int]]=None) -> List[HeadScore]:
        if layer_range is None:
            lo = self.num_layers // 3
            hi = 2 * self.num_layers // 3 + 1
            layer_range = (lo, hi)
        lo, hi = layer_range
        owner = getattr(patch_fn, '__self__', None)
        batch_fn = getattr(owner, 'patch_fn_batch', None) if owner else None
        traj_ids_list = list(trajectory_ids)
        print(f"[score_all] layers=[{lo},{hi}) heads={self.num_heads} n_traj={len(traj_ids_list)} path={('batched' if batch_fn is not None else 'per-sample')}", flush=True)
        scored: List[HeadScore] = []
        for li in range(lo, hi):
            for hi_ in range(self.num_heads):
                if batch_fn is not None:
                    rates = batch_fn(traj_ids_list, li, hi_)
                else:
                    rates = [patch_fn(tid, li, hi_) for tid in traj_ids_list]
                avg = float(np.mean(rates)) if rates else 0.0
                scored.append(HeadScore(layer=li, head=hi_, flip_rate=avg))
        scored.sort(key=lambda s: s.flip_rate, reverse=True)
        self.scores = scored
        return scored

    def select_h_commit(self) -> List[HeadScore]:
        if not self.scores:
            return []
        total = self.num_layers * self.num_heads
        k = max(1, int(round(self.budget_fraction * total)))
        useful = [s for s in self.scores if s.flip_rate > 0.0]
        return useful[:k]

    def jaccard_stability(self, other_scores: List[HeadScore]) -> float:
        a = {(s.layer, s.head) for s in self.select_h_commit()}
        b_ident = self.__class__(self.num_layers, self.num_heads, self.budget_fraction)
        b_ident.scores = sorted(other_scores, key=lambda s: s.flip_rate, reverse=True)
        b = {(s.layer, s.head) for s in b_ident.select_h_commit()}
        if not a and (not b):
            return 1.0
        return len(a & b) / max(1, len(a | b))

@dataclass
class BreakingDecision:
    intervene: bool
    head_spec: List[Tuple[int, int]]
    window_k: int
    predicted_residual: float

class CommitmentBreakingPolicy:

    def __init__(self, head_spec: Sequence[Tuple[int, int]], target_residual: float=0.1, min_window: int=1, max_window: int=6, detect_threshold: Optional[float]=None):
        self.head_spec = [(int(l), int(h)) for l, h in head_spec]
        self.target_residual = float(target_residual)
        self.min_window = int(min_window)
        self.max_window = int(max_window)
        self.detect_threshold = detect_threshold
        self._prev_score: Optional[float] = None
        self._t_num: float = 0.0
        self._t_den: float = 0.0
        self._n_samples: int = 0
        self.n_triggered: int = 0
        self.n_skipped: int = 0

    def observe(self, score: float, was_ablated: bool=False) -> None:
        if was_ablated:
            self._prev_score = None
            return
        if self._prev_score is not None:
            self._t_num += self._prev_score * score
            self._t_den += self._prev_score * self._prev_score
            self._n_samples += 1
        self._prev_score = score

    @property
    def transition_operator(self) -> float:
        if self._t_den <= 1e-12 or self._n_samples < 2:
            return 1.0
        return float(self._t_num / self._t_den)

    @property
    def spectral_radius(self) -> float:
        t = self.transition_operator
        if t <= 0.0:
            return 0.0
        return float(min(t, 1.0))

    def optimal_window(self, current_score: float) -> int:
        s0 = max(float(current_score), 1e-06)
        delta = self.target_residual
        if s0 <= delta:
            return self.min_window
        rho = self.spectral_radius
        if rho <= 1e-06 or abs(np.log(max(rho, 1e-12))) < 0.02:
            return self.max_window
        k = int(np.ceil(np.log(delta / s0) / np.log(rho)))
        return int(np.clip(k, self.min_window, self.max_window))

    def decide(self, current_score: float, probe_threshold: float) -> BreakingDecision:
        thr = self.detect_threshold if self.detect_threshold is not None else probe_threshold
        if current_score < thr:
            self.n_skipped += 1
            return BreakingDecision(intervene=False, head_spec=[], window_k=0, predicted_residual=current_score)
        if not self.head_spec:
            self.n_skipped += 1
            return BreakingDecision(intervene=False, head_spec=[], window_k=0, predicted_residual=current_score)
        k = self.optimal_window(current_score)
        self.n_triggered += 1
        residual = current_score * self.spectral_radius ** k
        return BreakingDecision(intervene=True, head_spec=list(self.head_spec), window_k=k, predicted_residual=residual)

    def regret_bound(self, epsilon: float, p_detect: float, alpha_base: float) -> float:
        if np.isnan(epsilon) or np.isnan(p_detect) or np.isnan(alpha_base):
            return 0.0
        eps = float(np.clip(epsilon, 0.0, 1.0))
        p = float(np.clip(p_detect, 0.0, 1.0))
        a = float(np.clip(alpha_base, 0.0, 1.0))
        return float((1.0 - eps) * p * (1.0 - a))

    def reset(self) -> None:
        self._prev_score = None
        self._t_num = 0.0
        self._t_den = 0.0
        self._n_samples = 0
        self.n_triggered = 0
        self.n_skipped = 0
