from __future__ import annotations
import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Sequence, Optional
import numpy as np
_LATEX_WRAPPERS = [('\\\\left', ''), ('\\\\right', ''), ('\\\\!', ''), ('\\\\,', ''), ('\\\\;', ''), ('\\\\:', ''), ('\\\\quad', ''), ('\\\\qquad', ''), ('\\\\displaystyle', ''), ('\\\\textstyle', ''), ('\\\\text\\s*\\{([^{}]*)\\}', '\\1'), ('\\\\mathrm\\s*\\{([^{}]*)\\}', '\\1'), ('\\\\mathbf\\s*\\{([^{}]*)\\}', '\\1'), ('\\\\operatorname\\s*\\{([^{}]*)\\}', '\\1'), ('\\\\dfrac', '\\\\frac'), ('\\\\tfrac', '\\\\frac'), ('−', '-')]

def _extract_boxed(s: str) -> str:
    idx = s.rfind('\\boxed{')
    if idx < 0:
        return s
    start = idx + len('\\boxed{')
    depth = 1
    i = start
    while i < len(s) and depth > 0:
        c = s[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[start:i]
        i += 1
    return s[start:]

def _strip_frac(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub('\\\\frac\\s*\\{([^{}]+)\\}\\s*\\{([^{}]+)\\}', '(\\1)/(\\2)', s)
    return s

def _strip_sqrt(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub('\\\\sqrt\\s*\\{([^{}]+)\\}', 'sqrt(\\1)', s)
    return s

def normalize_answer(s: Optional[str]) -> str:
    if s is None:
        return ''
    s = str(s).strip()
    s = s.strip('$').strip()
    if '\\boxed{' in s:
        s = _extract_boxed(s)
    for pat, repl in _LATEX_WRAPPERS:
        s = re.sub(pat, repl, s)
    s = _strip_frac(s)
    s = _strip_sqrt(s)
    s = s.lower()
    s = s.replace('$', '').replace(' ', '').replace('\t', '')
    s = re.sub('(?<=\\d),(?=\\d{3}\\b)', '', s)
    s = s.rstrip('.,;:')
    if s.startswith('{') and s.endswith('}'):
        depth = 0
        balanced = True
        for i, c in enumerate(s):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and i < len(s) - 1:
                    balanced = False
                    break
        if balanced:
            s = s[1:-1]
    return s

def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

def _sympy_equal(a: str, b: str) -> bool:
    try:
        import sympy
        from sympy import sympify, simplify, nsimplify, Basic
    except Exception:
        return False
    try:
        ea = sympify(a, rational=True, evaluate=True)
        eb = sympify(b, rational=True, evaluate=True)
    except Exception:
        return False
    if not isinstance(ea, Basic) or not isinstance(eb, Basic):
        return False
    try:
        if ea == eb:
            return True
        diff = simplify(ea - eb)
        if diff == 0:
            return True
        try:
            return abs(float(diff.evalf())) < 1e-06
        except Exception:
            return False
    except Exception:
        return False

def _tuple_equal(a: str, b: str) -> bool:
    if not (a.startswith('(') and a.endswith(')') and b.startswith('(') and b.endswith(')')):
        return False
    a_parts = [p.strip() for p in a[1:-1].split(',')]
    b_parts = [p.strip() for p in b[1:-1].split(',')]
    if len(a_parts) != len(b_parts):
        return False
    for pa, pb in zip(a_parts, b_parts):
        if pa == pb:
            continue
        fa, fb = (_to_float(pa), _to_float(pb))
        if fa is not None and fb is not None:
            if abs(fa - fb) < 1e-06:
                continue
            return False
        if _sympy_equal(pa, pb):
            continue
        return False
    return True
_MC_LETTER_TOKEN = re.compile('(?:^|[\\s(\\[,.:;])([a-eA-E])(?=$|[\\s)\\].,:;])')

def _extract_mc_letter(s: str) -> Optional[str]:
    matches = _MC_LETTER_TOKEN.findall(s)
    if matches:
        return matches[-1].upper()
    for c in reversed(s):
        if c.upper() in 'ABCDE':
            return c.upper()
    return None

def grade(predicted: Optional[str], target: Optional[str], is_multiple_choice: bool=False) -> bool:
    if predicted is None or target is None:
        return False
    p_raw = str(predicted)
    t_raw = str(target)
    p = normalize_answer(p_raw)
    t = normalize_answer(t_raw)
    if not p or not t:
        return False
    if is_multiple_choice:
        pl = _extract_mc_letter(p_raw) or _extract_mc_letter(p)
        tl = _extract_mc_letter(t_raw) or _extract_mc_letter(t)
        if pl and tl:
            return pl == tl
        return p == t
    if p == t:
        return True
    fp, ft = (_to_float(p), _to_float(t))
    if fp is not None and ft is not None:
        return abs(fp - ft) < 1e-06
    if _tuple_equal(p, t):
        return True
    if _sympy_equal(p, t):
        return True
    longer, shorter = (p, t) if len(p) >= len(t) else (t, p)
    if shorter and shorter in longer and (len(longer) <= max(len(shorter) * 2, len(shorter) + 4)):
        return True
    return False

def pass_at_1(predictions: Sequence[str], targets: Sequence[str], is_mc: bool=False) -> float:
    n = len(predictions)
    if n == 0:
        return 0.0
    return sum((grade(p, t, is_mc) for p, t in zip(predictions, targets))) / n

def avg_at_k(samples: Sequence[Sequence[str]], targets: Sequence[str], is_mc: bool=False) -> float:
    if not samples:
        return 0.0
    per_ex = []
    for sample_list, t in zip(samples, targets):
        if not sample_list:
            per_ex.append(0.0)
            continue
        correct = sum((grade(s, t, is_mc) for s in sample_list))
        per_ex.append(correct / len(sample_list))
    return float(np.mean(per_ex))

def cons_at_k(samples: Sequence[Sequence[str]], targets: Sequence[str], is_mc: bool=False) -> float:
    if not samples:
        return 0.0
    correct = 0
    for sample_list, t in zip(samples, targets):
        if not sample_list:
            continue
        norm = [normalize_answer(s) for s in sample_list if s]
        if not norm:
            continue
        top_norm = Counter(norm).most_common(1)[0][0]
        voted = next((s for s in sample_list if normalize_answer(s) == top_norm), sample_list[0])
        if grade(voted, t, is_mc):
            correct += 1
    return correct / len(samples)

def pass_at_k(samples: Sequence[Sequence[str]], targets: Sequence[str], is_mc: bool=False) -> float:
    if not samples:
        return 0.0
    hits = 0
    for sample_list, t in zip(samples, targets):
        if any((grade(s, t, is_mc) for s in sample_list)):
            hits += 1
    return hits / len(samples)

def accuracy_by_slice(correctnesses: Sequence[bool], slice_keys: Sequence[Optional[str]]) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[bool]] = {}
    for c, k in zip(correctnesses, slice_keys):
        if k is None:
            continue
        groups.setdefault(str(k), []).append(bool(c))
    return {k: {'accuracy': float(sum(v) / len(v)), 'n': len(v)} for k, v in groups.items()}

def expected_calibration_error(confidences: Sequence[float], correctnesses: Sequence[bool], n_bins: int=15) -> float:
    if len(confidences) == 0:
        return 0.0
    confidences = np.asarray(confidences, dtype=float)
    correctnesses = np.asarray(correctnesses, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = (bin_edges[i], bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf = float(confidences[mask].mean())
        acc = float(correctnesses[mask].mean())
        ece += mask.sum() / n * abs(avg_conf - acc)
    return float(ece)

def brier_score(confidences: Sequence[float], correctnesses: Sequence[bool]) -> float:
    if not confidences:
        return 0.0
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correctnesses, dtype=float)
    return float(np.mean((c - y) ** 2))

def adaptive_ece(confidences: Sequence[float], correctnesses: Sequence[bool], n_bins: int=15) -> float:
    if len(confidences) == 0:
        return 0.0
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correctnesses, dtype=float)
    n = len(c)
    order = np.argsort(c)
    c_sorted = c[order]
    y_sorted = y[order]
    eff_bins = max(min(n_bins, n), 1)
    bin_size = max(n // eff_bins, 1)
    ece = 0.0
    for b in range(eff_bins):
        lo = b * bin_size
        hi = (b + 1) * bin_size if b < eff_bins - 1 else n
        if lo >= n or hi <= lo:
            continue
        avg_conf = float(c_sorted[lo:hi].mean())
        acc = float(y_sorted[lo:hi].mean())
        ece += (hi - lo) / n * abs(avg_conf - acc)
    return float(ece)

def maximum_calibration_error(confidences: Sequence[float], correctnesses: Sequence[bool], n_bins: int=15) -> float:
    if len(confidences) == 0:
        return 0.0
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correctnesses, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        lo, hi = (bin_edges[i], bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (c >= lo) & (c <= hi)
        else:
            mask = (c >= lo) & (c < hi)
        if mask.sum() == 0:
            continue
        gap = abs(float(c[mask].mean()) - float(y[mask].mean()))
        mce = max(mce, gap)
    return float(mce)

def nll_binary(confidences: Sequence[float], correctnesses: Sequence[bool]) -> float:
    if len(confidences) == 0:
        return 0.0
    c = np.clip(np.asarray(confidences, dtype=float), 1e-07, 1.0 - 1e-07)
    y = np.asarray(correctnesses, dtype=float)
    return float(-np.mean(y * np.log(c) + (1.0 - y) * np.log(1.0 - c)))

def auroc(confidences: Sequence[float], correctnesses: Sequence[bool]) -> float:
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correctnesses, dtype=bool)
    pos = c[y]
    neg = c[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    rank_pos = ranks[:len(pos)]
    auc = (rank_pos.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return float(auc)

def detection_recall_at_fpr(scores: Sequence[float], labels: Sequence[bool], target_fpr: float=0.05) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(s) == 0 or y.sum() == 0 or (~y).sum() == 0:
        return 0.0
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    fpr = fp / (~y).sum()
    tpr = tp / y.sum()
    valid = fpr <= target_fpr
    return float(tpr[valid].max()) if valid.any() else 0.0

def roc_curve(scores: Sequence[float], labels: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(s) == 0:
        return (np.array([0, 1]), np.array([0, 1]))
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.concatenate([[0], np.cumsum(y_sorted)])
    fp = np.concatenate([[0], np.cumsum(~y_sorted)])
    tpr = tp / max(y.sum(), 1)
    fpr = fp / max((~y).sum(), 1)
    return (fpr, tpr)

def fpr_at_tpr(scores: Sequence[float], labels: Sequence[bool], target_tpr: float=0.95) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(s) == 0 or y.sum() == 0 or (~y).sum() == 0:
        return 1.0
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = tp / y.sum()
    fpr = fp / (~y).sum()
    valid = tpr >= target_tpr
    return float(fpr[valid].min()) if valid.any() else 1.0

def youden_j(scores: Sequence[float], labels: Sequence[bool]) -> Dict[str, float]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(s) == 0 or y.sum() == 0 or (~y).sum() == 0:
        return {'j': 0.0, 'threshold': 0.5, 'tpr': 0.0, 'fpr': 0.0}
    order = np.argsort(-s)
    s_sorted = s[order]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = tp / y.sum()
    fpr = fp / (~y).sum()
    j = tpr - fpr
    best = int(np.argmax(j))
    return {'j': float(j[best]), 'threshold': float(s_sorted[best]), 'tpr': float(tpr[best]), 'fpr': float(fpr[best])}

def auprc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(s) == 0 or y.sum() == 0:
        return float('nan')
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))

def pr_curve(scores: Sequence[float], labels: Sequence[bool]) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if len(s) == 0 or y.sum() == 0:
        return (np.array([0, 1]), np.array([1, 0]))
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    return (recall, precision)

def efficiency_summary(total_tokens: int, total_calls: int, wall_clock_s: float, n_examples: int, n_interventions: int=0, peak_vram_gb: float=0.0, n_correct: int=0, latencies: Optional[Sequence[float]]=None) -> Dict[str, float]:
    n = max(n_examples, 1)
    out = {'tokens_per_ex': total_tokens / n, 'calls_per_ex': total_calls / n, 'seconds_per_ex': wall_clock_s / n, 'total_tokens': int(total_tokens), 'total_calls': int(total_calls), 'wall_clock_s': float(wall_clock_s), 'intervention_rate': n_interventions / n if n else 0.0, 'peak_vram_gb': float(peak_vram_gb)}
    if n_correct > 0:
        out['tokens_per_correct'] = total_tokens / n_correct
        out['seconds_per_correct'] = wall_clock_s / n_correct
    else:
        out['tokens_per_correct'] = float('inf')
        out['seconds_per_correct'] = float('inf')
    if latencies is not None and len(latencies) > 0:
        lat = np.asarray(latencies, dtype=float)
        out['latency_p50'] = float(np.percentile(lat, 50))
        out['latency_p95'] = float(np.percentile(lat, 95))
        out['latency_p99'] = float(np.percentile(lat, 99))
        out['latency_max'] = float(lat.max())
    else:
        out['latency_p50'] = out['latency_p95'] = 0.0
        out['latency_p99'] = out['latency_max'] = 0.0
    return out

def cost_adjusted_accuracy(accuracy: float, tokens_per_ex: float, lam: float=0.0001) -> float:
    return float(accuracy - lam * tokens_per_ex)

def confident_error_rate(confidences: Sequence[float], correctnesses: Sequence[bool], threshold: float=0.8) -> float:
    if len(confidences) == 0:
        return 0.0
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(correctnesses, dtype=bool)
    confident_wrong = (c > threshold) & ~y
    return float(confident_wrong.sum() / len(c))

def step_level_err_auroc(step_records_list: Sequence[Sequence[Dict[str, float]]], correctnesses: Sequence[bool]) -> Dict[str, float]:
    scores: List[float] = []
    labels: List[bool] = []
    for steps, correct in zip(step_records_list, correctnesses):
        if not steps:
            continue
        is_error = not bool(correct)
        for s in steps:
            ent = float(s.get('entropy', 0.0) or 0.0)
            scores.append(ent)
            labels.append(is_error)
    if not scores or sum(labels) == 0 or sum(labels) == len(labels):
        return {'auroc': float('nan'), 'auprc': float('nan'), 'n_steps': len(scores)}
    return {'auroc': auroc(scores, labels), 'auprc': auprc(scores, labels), 'n_steps': len(scores)}

def reasoning_chain_stats(step_records_list: Sequence[Sequence[Dict[str, float]]], got_answer_flags: Sequence[bool]) -> Dict[str, float]:
    n_examples = max(len(step_records_list), 1)
    total_steps = 0
    total_tokens_in_steps = 0
    step_entropies: List[float] = []
    per_ex_max_ent: List[float] = []
    n_interventions = 0
    for steps in step_records_list:
        if not steps:
            per_ex_max_ent.append(0.0)
            continue
        total_steps += len(steps)
        ents = []
        for s in steps:
            total_tokens_in_steps += int(s.get('token_count', 0) or 0)
            e = float(s.get('entropy', 0.0) or 0.0)
            step_entropies.append(e)
            ents.append(e)
            if bool(s.get('intervened', False)):
                n_interventions += 1
        per_ex_max_ent.append(max(ents) if ents else 0.0)
    avg_steps = total_steps / n_examples
    avg_step_tokens = total_tokens_in_steps / total_steps if total_steps else 0.0
    avg_step_ent = float(np.mean(step_entropies)) if step_entropies else 0.0
    max_step_ent = float(np.mean(per_ex_max_ent)) if per_ex_max_ent else 0.0
    early_exit = float(sum((bool(g) for g in got_answer_flags)) / n_examples) if got_answer_flags else 0.0
    inter_rate_step = n_interventions / total_steps if total_steps else 0.0
    return {'avg_steps_per_traj': float(avg_steps), 'avg_step_tokens': float(avg_step_tokens), 'avg_step_entropy': avg_step_ent, 'max_step_entropy_mean': max_step_ent, 'early_exit_rate': early_exit, 'intervention_rate_stepwise': float(inter_rate_step), 'total_steps': int(total_steps), 'total_step_interventions': int(n_interventions)}

def intervention_diagnostics(predictions: Sequence[Dict[str, object]]) -> Dict[str, float]:
    n = len(predictions)
    if n == 0:
        return {'intervention_precision': 0.0, 'intervention_recall': 0.0, 'self_correction_rate': 0.0}
    flipped_total = 0
    saved_by_intervention = 0
    wrong_intervened = 0
    wrong_total = 0
    for p in predictions:
        if not bool(p.get('correct', False)):
            wrong_total += 1
            if int(p.get('n_interventions', 0) or 0) > 0:
                wrong_intervened += 1
        if bool(p.get('answer_flipped_by_intervention', False)):
            flipped_total += 1
            if bool(p.get('correct', False)):
                saved_by_intervention += 1
    precision = saved_by_intervention / flipped_total if flipped_total else 0.0
    recall = wrong_intervened / wrong_total if wrong_total else 0.0
    return {'intervention_precision': float(precision), 'intervention_recall': float(recall), 'self_correction_rate': float(flipped_total / n)}

def prediction_mse_by_horizon(predictions: Sequence[Sequence[float]], actuals: Sequence[Sequence[float]]) -> Dict[int, float]:
    horizons: Dict[int, List[float]] = {}
    for pred_traj, act_traj in zip(predictions, actuals):
        for k, (p, a) in enumerate(zip(pred_traj, act_traj), start=1):
            horizons.setdefault(k, []).append((p - a) ** 2)
    return {k: float(np.mean(v)) for k, v in horizons.items() if v}

def fit_exponential_decay(horizon_mse: Dict[int, float]) -> Tuple[float, float]:
    if len(horizon_mse) < 2:
        return (1.0, 0.9)
    ks = np.array(sorted(horizon_mse.keys()), dtype=float)
    mses = np.array([horizon_mse[int(k)] for k in ks], dtype=float)
    valid = mses > 1e-10
    if valid.sum() < 2:
        return (1.0, 0.9)
    log_mse = np.log(mses[valid])
    slope, intercept = np.polyfit(ks[valid], log_mse, 1)
    return (float(np.exp(intercept)), float(np.exp(slope)))
