from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

def _resolve_layer(model, layer_idx: int):
    base = getattr(model, 'model', model)
    if hasattr(base, 'layers'):
        return base.layers[layer_idx]
    if hasattr(base, 'decoder') and hasattr(base.decoder, 'layers'):
        return base.decoder.layers[layer_idx]
    raise AttributeError(f'cannot locate decoder layers on {type(model).__name__}')

def resolve_attn(layer):
    for name in ('self_attn', 'self_attention', 'attention', 'attn'):
        mod = getattr(layer, name, None)
        if mod is not None and hasattr(mod, 'o_proj'):
            return (mod, name)
    for name, child in layer.named_children():
        if hasattr(child, 'o_proj'):
            return (child, name)
    children = list(layer._modules.keys())
    raise AttributeError(f'no attention submodule with .o_proj on {type(layer).__name__}; children: {children}')

class HiddenStateCapture:

    def __init__(self, model, layer_idx: int):
        self.layer = _resolve_layer(model, layer_idx)
        self.layer_idx = layer_idx
        self._last_state_gpu: Optional[torch.Tensor] = None
        self._handle = None

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.ndim == 3:
            last_pos = hs[:, -1, :]
        elif hs.ndim == 2:
            last_pos = hs[-1:, :]
        elif hs.ndim == 1:
            last_pos = hs.unsqueeze(0)
        else:
            raise ValueError(f'Unexpected hidden state dimensions: {hs.shape}')
        self._last_state_gpu = last_pos.detach()
        return output

    @property
    def last_state(self) -> Optional[np.ndarray]:
        t = self._last_state_gpu
        if t is None:
            return None
        if t.ndim == 1:
            t = t.unsqueeze(0)
        arr = t.float().cpu().numpy()
        if arr.ndim == 2 and arr.shape[0] > 1:
            arr = arr[:1]
        return arr

    @property
    def last_state_batch(self) -> Optional[np.ndarray]:
        t = self._last_state_gpu
        if t is None:
            return None
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t.float().cpu().numpy()

    def __enter__(self) -> 'HiddenStateCapture':
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def reset(self) -> None:
        self._last_state_gpu = None

class MultiLayerHiddenStateCapture:

    def __init__(self, model, layer_indices: Sequence[int]):
        base = getattr(model, 'model', model)
        if hasattr(base, 'layers'):
            n_layers = len(base.layers)
        elif hasattr(base, 'decoder') and hasattr(base.decoder, 'layers'):
            n_layers = len(base.decoder.layers)
        else:
            raise AttributeError(f'cannot resolve layers on {type(model).__name__}')
        resolved: List[int] = []
        for li in layer_indices:
            idx = int(li)
            if idx < 0:
                idx = n_layers + idx
            if not 0 <= idx < n_layers:
                raise ValueError(f'layer_indices={list(layer_indices)} contains out-of-range {li} (resolved {idx}) for {n_layers}-layer model')
            resolved.append(idx)
        seen: set = set()
        ordered: List[int] = []
        for i in resolved:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        self.layer_indices: List[int] = ordered
        self._captures: Dict[int, HiddenStateCapture] = {i: HiddenStateCapture(model, layer_idx=i) for i in ordered}

    @property
    def last_state(self) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for i, cap in self._captures.items():
            hs = cap.last_state
            if hs is not None:
                out[i] = hs
        return out

    @property
    def last_state_batch(self) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for i, cap in self._captures.items():
            hs = cap.last_state_batch
            if hs is not None:
                out[i] = hs
        return out

    def __enter__(self) -> 'MultiLayerHiddenStateCapture':
        for cap in self._captures.values():
            cap.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for cap in self._captures.values():
            cap.__exit__(exc_type, exc, tb)

    def reset(self) -> None:
        for cap in self._captures.values():
            cap.reset()

class AttentionHeadAblator:

    def __init__(self, model, layer_idx: int, head_indices: Iterable[int], num_heads: Optional[int]=None):
        self.layer = _resolve_layer(model, layer_idx)
        self.attn, self._attn_attr = resolve_attn(self.layer)
        self.o_proj = self.attn.o_proj
        self.layer_idx = layer_idx
        self.heads = sorted(set((int(h) for h in head_indices)))
        if num_heads is None:
            num_heads = getattr(self.attn, 'num_heads', getattr(self.attn, 'num_attention_heads', None))
        if num_heads is None:
            cfg = getattr(model, 'config', None)
            num_heads = getattr(cfg, 'num_attention_heads', None)
        if num_heads is None:
            raise AttributeError('cannot determine num_heads on self_attn; pass num_heads=')
        self.num_heads = int(num_heads)
        self._handle = None

    def _pre_hook(self, module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if x is None:
            return inputs
        x = x.clone()
        *lead, d = x.shape
        head_dim = d // self.num_heads
        v = x.view(*lead, self.num_heads, head_dim)
        for h in self.heads:
            if 0 <= h < self.num_heads:
                v[..., h, :] = 0.0
        x = v.view(*lead, d)
        return (x,) + tuple(inputs[1:])

    def __enter__(self) -> 'AttentionHeadAblator':
        self._handle = self.o_proj.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

class WeightPatchAblator:

    def __init__(self, model, layer_idx: int, head_indices: Iterable[int], num_heads: Optional[int]=None):
        self.layer = _resolve_layer(model, layer_idx)
        self.attn, self._attn_attr = resolve_attn(self.layer)
        self.o_proj = self.attn.o_proj
        self.layer_idx = layer_idx
        self.heads = sorted(set((int(h) for h in head_indices)))
        if num_heads is None:
            num_heads = getattr(self.attn, 'num_heads', getattr(self.attn, 'num_attention_heads', None))
        if num_heads is None:
            cfg = getattr(model, 'config', None)
            num_heads = getattr(cfg, 'num_attention_heads', None)
        if num_heads is None:
            raise AttributeError('cannot determine num_heads on self_attn; pass num_heads=')
        self.num_heads = int(num_heads)
        self._saved_cols: Optional[Dict[int, torch.Tensor]] = None
        self._head_dim: Optional[int] = None

    def __enter__(self) -> 'WeightPatchAblator':
        if self._saved_cols is not None:
            raise RuntimeError(f'WeightPatchAblator(L{self.layer_idx}, heads={self.heads}) entered twice without matching exit')
        W = self.o_proj.weight
        in_features = W.shape[1]
        if in_features % self.num_heads != 0:
            raise ValueError(f'o_proj.in_features={in_features} not divisible by num_heads={self.num_heads}')
        head_dim = in_features // self.num_heads
        saved: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for h in self.heads:
                if 0 <= h < self.num_heads:
                    start = h * head_dim
                    end = start + head_dim
                    saved[h] = W[:, start:end].detach().clone()
                    W[:, start:end].zero_()
        self._saved_cols = saved
        self._head_dim = head_dim
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._saved_cols is None:
            return
        W = self.o_proj.weight
        head_dim = self._head_dim
        with torch.no_grad():
            for h, col in self._saved_cols.items():
                start = h * head_dim
                end = start + head_dim
                W[:, start:end].copy_(col)
        self._saved_cols = None
        self._head_dim = None

def _ablator_mode() -> str:
    mode = os.environ.get('LOCI_ABLATION_MODE', 'weight_patch').strip().lower()
    if mode in ('weight_patch', 'pre_hook'):
        return mode
    import sys
    print(f'[loci.hooks] unknown LOCI_ABLATION_MODE={mode!r}, using weight_patch', file=sys.stderr)
    return 'weight_patch'

def make_ablator(model, layer_idx: int, head_indices: Iterable[int], num_heads: Optional[int]=None):
    cls = WeightPatchAblator if _ablator_mode() == 'weight_patch' else AttentionHeadAblator
    return cls(model, layer_idx=layer_idx, head_indices=head_indices, num_heads=num_heads)

class DirectionalSteeringHook:

    def __init__(self, model, layer_idx: int, w_raw, alpha: float=1.0):
        self.layer = _resolve_layer(model, layer_idx)
        self.layer_idx = layer_idx
        self.alpha = float(alpha)
        w = torch.as_tensor(w_raw, dtype=torch.float32).flatten()
        norm = float(w.norm().item())
        if not np.isfinite(norm) or norm < 1e-10:
            raise ValueError(f'DirectionalSteeringHook: w_raw has norm={norm}; expected a non-trivial direction (did probe_to_raw_direction return zero-vector? check probe weights / PCA basis)')
        w_hat = w / norm
        try:
            p0 = next(model.parameters())
            w_hat = w_hat.to(dtype=p0.dtype, device=p0.device)
        except StopIteration:
            pass
        self.w_hat = w_hat
        self._handle = None

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        proj = (hs * self.w_hat).sum(dim=-1, keepdim=True)
        hs_new = hs - self.alpha * proj * self.w_hat
        if isinstance(output, tuple):
            return (hs_new,) + output[1:]
        return hs_new

    def __enter__(self) -> 'DirectionalSteeringHook':
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

def probe_to_raw_direction(probe) -> np.ndarray:
    w = np.asarray(probe.w, dtype=np.float64).flatten()
    std = np.asarray(probe.std, dtype=np.float64).flatten()
    if std.shape == w.shape:
        v = w / np.where(std > 1e-10, std, 1.0)
    else:
        v = w
    basis = getattr(probe, 'pca_basis', None)
    if basis is not None:
        basis_np = np.asarray(basis, dtype=np.float64)
        if basis_np.shape[0] != v.shape[0]:
            raise ValueError(f'probe_to_raw_direction: pca_basis.shape[0]={basis_np.shape[0]} does not match w.shape[0]={v.shape[0]}')
        v = v @ basis_np
    return v

@contextmanager
def ablate_many(model, head_spec: Iterable[tuple]):
    per_layer: dict = {}
    for li, hi in head_spec:
        per_layer.setdefault(int(li), []).append(int(hi))
    ablators = [make_ablator(model, li, hs) for li, hs in per_layer.items()]
    for a in ablators:
        a.__enter__()
    try:
        yield ablators
    finally:
        for a in reversed(ablators):
            a.__exit__(None, None, None)

class ResidualPatcher:

    def __init__(self, model, layer_idx: int, donor):
        self.layer = _resolve_layer(model, layer_idx)
        self.layer_idx = layer_idx
        d = torch.as_tensor(donor, dtype=torch.float32).flatten()
        if d.numel() == 0 or not torch.isfinite(d).all():
            raise ValueError(f'ResidualPatcher: donor has numel={d.numel()} or non-finite entries')
        try:
            p0 = next(model.parameters())
            d = d.to(dtype=p0.dtype, device=p0.device)
        except StopIteration:
            pass
        self.donor = d
        self._handle = None

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs.clone()
        hs[..., -1, :] = self.donor
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def __enter__(self) -> 'ResidualPatcher':
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

class RetrievalDonorPatcher:

    def __init__(self, model, layer_idx: int, keys, keys_unit, donor_values, K: int=8, alpha: float=0.5, metric: str='cosine'):
        self.layer = _resolve_layer(model, layer_idx)
        self.layer_idx = layer_idx
        self.K = int(K)
        self.alpha = float(alpha)
        if metric not in ('cosine', 'l2'):
            raise ValueError(f'metric={metric!r} must be cosine or l2')
        self.metric = metric
        keys_t = torch.as_tensor(keys, dtype=torch.float32)
        keys_unit_t = torch.as_tensor(keys_unit, dtype=torch.float32)
        donor_values_t = torch.as_tensor(donor_values, dtype=torch.float32)
        if not keys_t.shape == keys_unit_t.shape == donor_values_t.shape:
            raise ValueError(f'shape mismatch: keys={keys_t.shape}, unit={keys_unit_t.shape}, donor={donor_values_t.shape}')
        if keys_t.numel() == 0:
            raise ValueError('RetrievalDonorPatcher: empty key bank')
        try:
            p0 = next(model.parameters())
            target_dtype = p0.dtype
            target_device = p0.device
        except StopIteration:
            target_dtype = torch.float32
            target_device = torch.device('cpu')
        self.keys = keys_t.to(dtype=target_dtype, device=target_device)
        self.keys_unit = keys_unit_t.to(dtype=target_dtype, device=target_device)
        self.donor_values = donor_values_t.to(dtype=target_dtype, device=target_device)
        self._handle = None

    def _retrieve_donor(self, h_last: torch.Tensor) -> torch.Tensor:
        if self.metric == 'cosine':
            h_norm = h_last.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            h_unit = h_last / h_norm
            sims = h_unit @ self.keys_unit.T
            top_vals, top_idx = sims.topk(min(self.K, sims.shape[-1]), dim=-1)
            weights = top_vals.clamp_min(0.0)
            wsum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            weights = weights / wsum
        else:
            d = (h_last.unsqueeze(1) - self.keys.unsqueeze(0)).pow(2).sum(-1)
            top_vals, top_idx = (-d).topk(min(self.K, d.shape[-1]), dim=-1)
            weights = torch.softmax(top_vals, dim=-1)
        gathered = self.donor_values[top_idx]
        donor = (gathered * weights.unsqueeze(-1)).sum(dim=1)
        return donor

    def _hook(self, module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.ndim != 3:
            return output
        hs = hs.clone()
        h_last = hs[:, -1, :]
        donor = self._retrieve_donor(h_last)
        hs[:, -1, :] = (1.0 - self.alpha) * h_last + self.alpha * donor
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def __enter__(self) -> 'RetrievalDonorPatcher':
        self._handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
