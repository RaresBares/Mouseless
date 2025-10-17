

import torch
import numpy as np

def normalize_points_2d(k, anchor_idx=None, eps=1e-6):
    if isinstance(k, np.ndarray):
        return normalize_points_2d_np(k, anchor_idx=anchor_idx, eps=eps)
    if k.dim() == 2:
        k0 = k.unsqueeze(0)
        sq = True
    else:
        k0 = k
        sq = False
    center = k0[:, anchor_idx:anchor_idx+1, :] if anchor_idx is not None else k0.mean(dim=1, keepdim=True)
    k0 = k0 - center
    s = torch.linalg.norm(k0, dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1)
    k0 = k0 / (s + eps)
    return k0.squeeze(0) if sq else k0

def normalize_points_2d_np(k, anchor_idx=None, eps=1e-6):
    k = np.asarray(k, dtype=np.float32)
    if k.ndim == 2:
        k0 = k[None, ...]
        sq = True
    else:
        k0 = k
        sq = False
    center = k0[:, anchor_idx:anchor_idx+1, :] if anchor_idx is not None else k0.mean(axis=1, keepdims=True)
    k0 = k0 - center
    s = np.linalg.norm(k0, axis=-1).mean(axis=1, keepdims=True)[..., None]
    k0 = k0 / (s + eps)
    return k0[0] if sq else k0