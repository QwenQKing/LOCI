from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out

@dataclass
class CommitmentProbe:
    dim: int
    w: np.ndarray = field(default=None, repr=False)
    b: float = 0.0
    l2: float = 0.01
    commit_threshold: float = 0.75
    mean: np.ndarray = field(default=None, repr=False)
    std: np.ndarray = field(default=None, repr=False)
    pca_basis: Optional[np.ndarray] = field(default=None, repr=False)
    pca_mean: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.zeros(self.dim, dtype=np.float64)
        if self.mean is None:
            self.mean = np.zeros(self.dim, dtype=np.float64)
        if self.std is None:
            self.std = np.ones(self.dim, dtype=np.float64)

    def _maybe_project(self, X: np.ndarray) -> np.ndarray:
        if self.pca_basis is None or self.pca_mean is None:
            return X
        hidden_dim = int(self.pca_basis.shape[1])
        if X.shape[-1] == hidden_dim:
            return (X - self.pca_mean) @ self.pca_basis.T
        return X

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int=50, tol: float=1e-05, class_weight: Optional[str]='balanced') -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n, d = X.shape
        assert d == self.dim, f'expected dim {self.dim}, got {d}'
        if class_weight == 'balanced':
            n_pos = float(np.sum(y == 1))
            n_neg = float(np.sum(y == 0))
            if n_pos > 0 and n_neg > 0:
                w_pos = 0.5 * n / n_pos
                w_neg = 0.5 * n / n_neg
            else:
                w_pos = w_neg = 1.0
            sample_w = np.where(y == 1, w_pos, w_neg)
        else:
            sample_w = np.ones(n, dtype=np.float64)
        sw_sum = float(sample_w.sum()) + 1e-12
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std = np.where(self.std < 1e-06, 1.0, self.std)
        Xs = self._standardize(X)
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        for _ in range(max_iter):
            z = Xs @ w + b
            p = _sigmoid(z)
            resid = (p - y) * sample_w
            grad_w = Xs.T @ resid / sw_sum + self.l2 * w
            grad_b = float(resid.sum() / sw_sum)
            s = (p * (1.0 - p) + 1e-08) * sample_w
            H = Xs.T * s @ Xs / sw_sum + self.l2 * np.eye(d)
            try:
                delta = np.linalg.solve(H, grad_w)
            except np.linalg.LinAlgError:
                delta = grad_w
            w_new = w - delta
            b_new = b - grad_b / (float(s.sum() / sw_sum) + 1e-08)
            if np.linalg.norm(w_new - w) < tol and abs(b_new - b) < tol:
                w, b = (w_new, b_new)
                break
            w, b = (w_new, b_new)
        self.w, self.b = (w, float(b))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._maybe_project(X)
        Xs = self._standardize(X)
        logits = Xs @ self.w + self.b
        logits = np.clip(logits, -20.0, 20.0)
        return _sigmoid(logits)

    def score_one(self, h: np.ndarray) -> float:
        return float(self.predict_proba(h.reshape(1, -1))[0])

    def is_committed(self, h: np.ndarray, threshold: Optional[float]=None) -> bool:
        thr = self.commit_threshold if threshold is None else threshold
        return self.score_one(h) >= thr

    def save(self, path: str) -> None:
        payload: Dict[str, Any] = {'w': self.w, 'b': np.array([self.b]), 'commit_threshold': np.array([self.commit_threshold]), 'mean': self.mean, 'std': self.std}
        if self.pca_basis is not None and self.pca_mean is not None:
            payload['pca_basis'] = self.pca_basis
            payload['pca_mean'] = self.pca_mean
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str) -> 'CommitmentProbe':
        data = np.load(path)
        w = data['w']
        b = float(data['b'][0])
        thr = float(data['commit_threshold'][0])
        p = cls(dim=w.shape[0], commit_threshold=thr)
        p.w = w
        p.b = b
        if 'mean' in data.files:
            p.mean = data['mean']
        if 'std' in data.files:
            p.std = data['std']
        if 'pca_basis' in data.files and 'pca_mean' in data.files:
            p.pca_basis = data['pca_basis']
            p.pca_mean = data['pca_mean']
        return p
