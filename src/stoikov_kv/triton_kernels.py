"""
Fused Triton GPU Kernels for StoikovKV (`stoikov_kv/triton_kernels.py`).

Provides a single-launch fused sparse dictionary reconstruction (+ optional half-split
RoPE rotation) kernel for `ElasticLexicoCodec`, avoiding intermediate `[N, m, head_dim]`
and `float32` RoPE allocations in GPU global memory.
"""

from __future__ import annotations

import os
from typing import Optional

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_stoikov")

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _lexico_reconstruct_kernel(
        idx_ptr,
        val_ptr,
        scale_ptr,
        dict_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_vecs,
        seq_len,
        m_atoms,
        half_dim,
        stride_in,
        stride_is,
        stride_vn,
        stride_vs,
        stride_sn,
        stride_da,
        stride_dd,
        stride_ct,
        stride_cd,
        stride_on,
        stride_od,
        HAS_SCALE: tl.constexpr,
        APPLY_ROPE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HALF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_vecs

        offs_h = tl.arange(0, BLOCK_HALF)
        mask_h = offs_h < half_dim
        mask_nh = mask_n[:, None] & mask_h[None, :]

        acc_lo = tl.zeros([BLOCK_N, BLOCK_HALF], dtype=tl.float32)
        acc_hi = tl.zeros([BLOCK_N, BLOCK_HALF], dtype=tl.float32)

        for k in range(m_atoms):
            atom_idx = tl.load(
                idx_ptr + offs_n * stride_in + k * stride_is,
                mask=mask_n,
                other=0,
            ).to(tl.int32)
            coeff = tl.load(
                val_ptr + offs_n * stride_vn + k * stride_vs,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            base_d = dict_ptr + atom_idx[:, None] * stride_da
            d_lo = tl.load(
                base_d + offs_h[None, :] * stride_dd,
                mask=mask_nh,
                other=0.0,
            ).to(tl.float32)
            d_hi = tl.load(
                base_d + (offs_h[None, :] + half_dim) * stride_dd,
                mask=mask_nh,
                other=0.0,
            ).to(tl.float32)
            acc_lo += d_lo * coeff[:, None]
            acc_hi += d_hi * coeff[:, None]

        if HAS_SCALE:
            sc = tl.load(scale_ptr + offs_n * stride_sn, mask=mask_n, other=0.0).to(tl.float32)
            acc_lo = acc_lo * sc[:, None]
            acc_hi = acc_hi * sc[:, None]

        if APPLY_ROPE:
            pos_idx = offs_n % seq_len
            c_val = tl.load(
                cos_ptr + pos_idx[:, None] * stride_ct + offs_h[None, :] * stride_cd,
                mask=mask_nh,
                other=1.0,
            ).to(tl.float32)
            s_val = tl.load(
                sin_ptr + pos_idx[:, None] * stride_ct + offs_h[None, :] * stride_cd,
                mask=mask_nh,
                other=0.0,
            ).to(tl.float32)
            rot_lo = acc_lo * c_val - acc_hi * s_val
            rot_hi = acc_hi * c_val + acc_lo * s_val
            acc_lo = rot_lo
            acc_hi = rot_hi

        base_o = out_ptr + offs_n[:, None] * stride_on
        tl.store(base_o + offs_h[None, :] * stride_od, acc_lo, mask=mask_nh)
        tl.store(base_o + (offs_h[None, :] + half_dim) * stride_od, acc_hi, mask=mask_nh)


def can_use_triton(tensor: torch.Tensor) -> bool:
    """Returns True when Triton is installed and `tensor` resides on a CUDA device."""
    return HAS_TRITON and tensor.is_cuda


def triton_lexico_reconstruct(
    indices: torch.Tensor,
    values: torch.Tensor,
    d_layer_t: torch.Tensor,
    active_atoms: Optional[int] = None,
    scale: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Reconstructs `[..., head_dim]` vectors from sparse dictionary atoms in one GPU kernel,
    with optional fused half-split RoPE rotation along the sequence dimension (`indices.shape[-2]`).
    """
    orig_shape = indices.shape[:-1]
    s_total = int(indices.shape[-1])
    m = s_total if active_atoms is None else int(active_atoms)
    head_dim = int(d_layer_t.shape[1])
    half_dim = head_dim // 2
    dtype = out_dtype if out_dtype is not None else d_layer_t.dtype

    n_vecs = 1
    for dim in orig_shape:
        n_vecs *= int(dim)

    if n_vecs == 0 or m == 0:
        return torch.zeros(orig_shape + (head_dim,), device=indices.device, dtype=dtype)

    idx_2d = indices.view(n_vecs, s_total) if indices.is_contiguous() else indices.contiguous().view(n_vecs, s_total)
    raw_vals = torch.Tensor._make_subclass(torch.Tensor, values) if type(values) is not torch.Tensor else values
    val_2d = raw_vals.view(n_vecs, s_total) if raw_vals.stride(-1) == 1 and raw_vals.is_contiguous() else raw_vals.contiguous().view(n_vecs, s_total)

    has_scale = scale is not None
    if has_scale:
        sc_1d = scale.view(n_vecs) if scale.is_contiguous() else scale.contiguous().view(n_vecs)
        stride_sn = sc_1d.stride(0)
    else:
        sc_1d = d_layer_t
        stride_sn = 0

    apply_rope = rope_cos is not None and rope_sin is not None and len(orig_shape) >= 1
    if apply_rope:
        seq_len = int(orig_shape[-1])
        stride_ct, stride_cd = rope_cos.stride(0), rope_cos.stride(1)
        c_ptr, s_ptr = rope_cos, rope_sin
    else:
        seq_len = 1
        stride_ct = stride_cd = 0
        c_ptr = s_ptr = d_layer_t

    out_2d = torch.empty((n_vecs, head_dim), device=indices.device, dtype=dtype)

    block_half = max(16, triton.next_power_of_2(half_dim))
    block_n = 64 if n_vecs >= 64 else max(16, triton.next_power_of_2(n_vecs))
    grid = (triton.cdiv(n_vecs, block_n),)

    _lexico_reconstruct_kernel[grid](
        idx_2d,
        val_2d,
        sc_1d,
        d_layer_t,
        c_ptr,
        s_ptr,
        out_2d,
        n_vecs,
        seq_len,
        m,
        half_dim,
        idx_2d.stride(0),
        idx_2d.stride(1),
        val_2d.stride(0),
        val_2d.stride(1),
        stride_sn,
        d_layer_t.stride(0),
        d_layer_t.stride(1),
        stride_ct,
        stride_cd,
        out_2d.stride(0),
        out_2d.stride(1),
        HAS_SCALE=has_scale,
        APPLY_ROPE=apply_rope,
        BLOCK_N=block_n,
        BLOCK_HALF=block_half,
        num_warps=4,
    )
    return out_2d.view(orig_shape + (head_dim,))
