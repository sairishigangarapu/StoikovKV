"""
General Hardware-Adaptive GPU Kernel Engine for StoikovKV (`stoikov_kv/triton_kernels.py`).
Supports NVIDIA (`sm_80`/`sm_86`/`sm_89`/`sm_90`/`sm_100`/`sm_120`), AMD ROCm (`hip`), Intel (`xpu`), and PyTorch fallback (`mps`/`cpu`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from ._security import ensure_secure_triton_cache_dir

ensure_secure_triton_cache_dir()

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

__all__ = [
    "HAS_TRITON", "GPUHardwareProfile", "can_use_triton", "detect_gpu_profile",
    "simulate_hardware_profile", "select_kernel_config", "triton_lexico_reconstruct",
    "triton_fused_kv_lexico_materialize", "triton_fused_write_probe",
]


@dataclass(frozen=True)
class GPUHardwareProfile:
    """Hardware capability descriptor for dynamic Triton tile, pipeline, and cache tuning."""

    backend: str
    vendor: str
    arch: str
    device_name: str
    tier: str
    warp_size: int
    multiprocessor_count: int
    max_shared_mem: int
    l2_cache_bytes: int
    has_tma: bool
    has_tmem_tcgen05: bool
    supports_eviction_hints: bool
    triton_supported: bool


_ACTIVE_BACKEND_PROBED = False
_ACTIVE_TRITON_BACKEND: Optional[str] = None
_PROFILE_CACHE: Dict[Tuple[str, int], GPUHardwareProfile] = {}
_CFG_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n - 1).bit_length())


def _get_active_triton_backend() -> Optional[str]:
    global _ACTIVE_BACKEND_PROBED, _ACTIVE_TRITON_BACKEND
    if not HAS_TRITON:
        return None
    if not _ACTIVE_BACKEND_PROBED:
        try:
            _ACTIVE_TRITON_BACKEND = getattr(triton.runtime.driver.active.get_current_target(), "backend", None)
        except Exception:
            _ACTIVE_TRITON_BACKEND = None
        _ACTIVE_BACKEND_PROBED = True
    return _ACTIVE_TRITON_BACKEND


def can_use_triton(tensor: torch.Tensor) -> bool:
    """Returns True when Triton is installed and its active driver backend matches `tensor.device`."""
    if not HAS_TRITON or not isinstance(tensor, torch.Tensor):
        return False
    if tensor.device.type == "cuda":
        return _get_active_triton_backend() in ("cuda", "hip")
    if tensor.device.type == "xpu":
        return _get_active_triton_backend() == "xpu"
    return False


def _query_triton_driver_props(dev_idx: int) -> Tuple[Optional[Any], Dict[str, Any]]:
    if not HAS_TRITON:
        return None, {}
    try:
        drv = triton.runtime.driver.active
        props = drv.utils.get_device_properties(dev_idx)
        return drv.get_current_target(), (props if isinstance(props, dict) else {})
    except Exception:
        return None, {}


def detect_gpu_profile(device: Optional[Union[torch.device, str, int]] = None) -> GPUHardwareProfile:
    """Introspects active GPU architecture across CUDA, ROCm/HIP, Intel XPU, Apple MPS, or CPU."""
    if device is None:
        if torch.cuda.is_available():
            dev = torch.device("cuda", torch.cuda.current_device())
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            dev = torch.device("xpu", torch.xpu.current_device())
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps", 0)
        else:
            dev = torch.device("cpu", 0)
    else:
        dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)

    dev_idx = 0 if dev.index is None else int(dev.index)
    key = (dev.type, dev_idx)
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]

    active_backend = _get_active_triton_backend()
    if dev.type == "cuda" and torch.cuda.is_available():
        is_hip = getattr(torch.version, "hip", None) is not None
        props = torch.cuda.get_device_properties(dev_idx)
        target, drv_p = _query_triton_driver_props(dev_idx)
        name = str(getattr(props, "name", "GPU"))
        sm_cnt = int(drv_p.get("multiprocessor_count", getattr(props, "multi_processor_count", 1)))
        l2 = int(getattr(props, "L2_cache_size", 0))
        if is_hip:
            arch = str(getattr(target, "arch", None) or getattr(props, "gcnArchName", "gfx900")).split(":")[0]
            warp_sz = int(drv_p.get("warpSize", getattr(props, "warp_size", 64)))
            smem = int(drv_p.get("max_shared_mem", getattr(props, "max_shared_memory_per_multiprocessor", 65536)))
            l2 = l2 if l2 > 0 else (256 * 1024**2 if arch.startswith(("gfx94", "gfx95")) else 8 * 1024**2)
            is_dc = any(k in f"{arch} {name}".lower() for k in ("gfx908", "gfx90a", "gfx94", "gfx95", "instinct", "mi2", "mi3"))
            prof = GPUHardwareProfile("hip", "amd", arch, name, "datacenter" if is_dc else "commercial", warp_sz, sm_cnt, smem, l2, False, False, True, bool(HAS_TRITON and active_backend == "hip"))
        else:
            sm = int(getattr(props, "major", 8)) * 10 + int(getattr(props, "minor", 0))
            warp_sz = int(drv_p.get("warpSize", 32))
            smem = int(drv_p.get("max_shared_mem", getattr(props, "max_shared_memory_per_multiprocessor", 101376)))
            l2 = l2 if l2 > 0 else 32 * 1024**2
            is_dc = (sm in (80, 90, 100, 103)) or any(k in name.lower() for k in ("a100", "a800", "h100", "h200", "h800", "b100", "b200", "gb200", "l40"))
            prof = GPUHardwareProfile("cuda", "nvidia", f"sm_{sm}", name, "datacenter" if is_dc else "commercial", warp_sz, sm_cnt, smem, l2, sm >= 90, sm in (100, 103), True, bool(HAS_TRITON and active_backend == "cuda"))
    elif dev.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(dev_idx)
        target, drv_p = _query_triton_driver_props(dev_idx)
        name = str(getattr(props, "name", "Intel XPU"))
        arch = str(getattr(target, "arch", None) or "xpu_xe")
        sm_cnt = int(drv_p.get("multiprocessor_count", getattr(props, "max_compute_units", 64)))
        smem = int(drv_p.get("max_shared_mem", getattr(props, "max_compute_shared_memory_size", 131072)))
        is_dc = any(k in name.lower() for k in ("max", "ponte vecchio", "gaudi", "data center"))
        prof = GPUHardwareProfile("xpu", "intel", arch, name, "datacenter" if is_dc else "commercial", int(drv_p.get("warpSize", 32)), sm_cnt, smem, (64 if is_dc else 16) * 1024**2, False, False, False, bool(HAS_TRITON and active_backend == "xpu"))
    elif dev.type == "mps":
        prof = GPUHardwareProfile("mps", "apple", "mps", "Apple Silicon MPS", "commercial", 32, 16, 32768, 16 * 1024**2, False, False, False, False)
    else:
        prof = GPUHardwareProfile("cpu", "cpu", "cpu", "CPU PyTorch Fallback", "fallback", 1, max(1, os.cpu_count() or 1), 0, 0, False, False, False, False)

    _PROFILE_CACHE[key] = prof
    return prof


_SIMULATED_PRESETS: Dict[str, GPUHardwareProfile] = {
    "rtx4060_ada": GPUHardwareProfile("cuda", "nvidia", "sm_89", "NVIDIA GeForce RTX 4060 Laptop GPU", "commercial", 32, 24, 101376, 32 * 1024**2, False, False, True, True),
    "a100_ampere": GPUHardwareProfile("cuda", "nvidia", "sm_80", "NVIDIA A100-SXM4-80GB", "datacenter", 32, 108, 167936, 40 * 1024**2, False, False, True, True),
    "h100_hopper": GPUHardwareProfile("cuda", "nvidia", "sm_90", "NVIDIA H100 80GB HBM3", "datacenter", 32, 132, 232448, 50 * 1024**2, True, False, True, True),
    "b200_blackwell": GPUHardwareProfile("cuda", "nvidia", "sm_100", "NVIDIA B200", "datacenter", 32, 148, 232448, 128 * 1024**2, True, True, True, True),
    "rtx5090_blackwell": GPUHardwareProfile("cuda", "nvidia", "sm_120", "NVIDIA GeForce RTX 5090", "commercial", 32, 170, 131072, 96 * 1024**2, True, False, True, True),
    "mi300x_rocm": GPUHardwareProfile("hip", "amd", "gfx942", "AMD Instinct MI300X", "datacenter", 64, 304, 65536, 256 * 1024**2, False, False, True, True),
    "intel_xpu": GPUHardwareProfile("xpu", "intel", "xpu_xe", "Intel Data Center GPU Max 1550", "datacenter", 32, 128, 131072, 64 * 1024**2, False, False, False, True),
    "cpu_fallback": GPUHardwareProfile("cpu", "cpu", "cpu", "CPU / Apple MPS PyTorch Fallback", "fallback", 1, 1, 0, 0, False, False, False, False),
}


def simulate_hardware_profile(preset: str) -> GPUHardwareProfile:
    """Returns a deterministic `GPUHardwareProfile` for cross-architecture tuning tests."""
    key = preset.strip().lower()
    if key not in _SIMULATED_PRESETS:
        raise ValueError(f"Unknown hardware preset '{preset}'. Valid presets: {sorted(_SIMULATED_PRESETS)}")
    return _SIMULATED_PRESETS[key]


def select_kernel_config(
    profile: GPUHardwareProfile, op: str = "lexico_reconstruct", n_vecs: int = 1024,
    head_dim: int = 128, ptr_aligned_16b: bool = True,
) -> Dict[str, Any]:
    """Selects hardware-adaptive tiles, stages, L2 hints, TMA checks, and isolated backend kwargs."""
    if not profile.triton_supported or profile.backend not in ("cuda", "hip", "xpu"):
        return {
            "use_triton": False, "BLOCK_N": 0, "num_warps": 0, "num_stages": 0,
            "evict_shared": "", "evict_stream": "", "tma_aligned": False, "tmem_eligible": False,
            "sram_budget_bytes": 0, "est_sram_bytes": 0, "launch_kwargs": {},
        }

    n_eff, d_eff = max(1, int(n_vecs)), max(16, _next_pow2(int(head_dim)))
    smem = int(profile.max_shared_mem)
    sram_budget = int(smem * 0.80)
    max_stages = 2 if smem <= 110_000 else (3 if smem <= 190_000 else 4)

    if op == "fused_write_probe" and n_eff < 16:
        block_n, num_warps, num_stages, est_sram = 1, 2, 1, 0
    else:
        if profile.has_tmem_tcgen05:
            target_bn = 128 if n_eff >= 256 else 64
        elif profile.backend == "hip":
            target_bn = 32
        elif profile.arch == "sm_90":
            target_bn = 128 if (op == "fused_write_probe" and n_eff >= 4096) else 64
        elif profile.arch in ("sm_80", "sm_120") or profile.backend == "xpu":
            target_bn = 32 if (op == "fused_kv_materialize" and n_eff <= 2048) else 64
        else:
            target_bn = 16 if (op == "fused_kv_materialize" and n_eff <= 8192) else (32 if op == "fused_kv_materialize" else 64)

        block_n = target_bn if n_eff >= target_bn else max(16, _next_pow2(n_eff))
        num_stages = max_stages
        fixed_sram = (d_eff * 16 * 4) if op == "fused_write_probe" else 4096
        while block_n > 16 and (fixed_sram + block_n * d_eff * 2) > sram_budget:
            block_n //= 2
        while num_stages > 1 and (fixed_sram + num_stages * block_n * d_eff * 2) > sram_budget:
            num_stages -= 1
        est_sram = fixed_sram + num_stages * block_n * d_eff * 2
        num_warps = 4 if profile.warp_size == 64 else (8 if (block_n >= 128 or (op == "fused_kv_materialize" and n_eff >= 512 and profile.backend == "cuda")) else 4)

    launch_kwargs: Dict[str, Any] = {"num_warps": num_warps, "num_stages": num_stages}
    cfg: Dict[str, Any] = {
        "use_triton": True, "BLOCK_N": block_n, "num_warps": num_warps, "num_stages": num_stages,
        "evict_shared": "evict_last" if profile.supports_eviction_hints else "",
        "evict_stream": "evict_first" if profile.supports_eviction_hints else "",
        "tma_aligned": bool(profile.has_tma and head_dim >= 16 and (head_dim * 2) % 16 == 0 and ptr_aligned_16b),
        "tmem_eligible": bool(profile.has_tmem_tcgen05 and block_n >= 64),
        "sram_budget_bytes": sram_budget, "est_sram_bytes": est_sram, "launch_kwargs": launch_kwargs,
    }
    if profile.backend == "hip":
        waves = 4 if block_n <= 16 else 2
        launch_kwargs["waves_per_eu"] = cfg["waves_per_eu"] = waves
    elif profile.backend == "xpu":
        grf = "large" if (block_n >= 64 or head_dim >= 128) else "auto"
        launch_kwargs["grf_mode"] = cfg["grf_mode"] = grf
    return cfg


def _cached_live_config(dev: torch.device, op: str, n_vecs: int, head_dim: int, aligned: bool) -> Dict[str, Any]:
    n_bucket = n_vecs if n_vecs < 16 else _next_pow2(min(n_vecs, 16384))
    key = (dev.type, 0 if dev.index is None else int(dev.index), op, n_bucket, head_dim, aligned)
    cfg = _CFG_CACHE.get(key)
    if cfg is None:
        cfg = select_kernel_config(detect_gpu_profile(dev), op, n_bucket, head_dim, ptr_aligned_16b=aligned)
        _CFG_CACHE[key] = cfg
    return cfg


if HAS_TRITON:

    @triton.jit
    def _lexico_reconstruct_kernel(
        idx_ptr, val_ptr, scale_ptr, dict_ptr, cos_ptr, sin_ptr, out_ptr, n_vecs, seq_len, m_atoms, n_atoms, half_dim,
        stride_in, stride_is, stride_vn, stride_vs, stride_sn, stride_da, stride_dd, stride_ct, stride_cd, stride_on, stride_od,
        HAS_SCALE: tl.constexpr, APPLY_ROPE: tl.constexpr, CONTIG_DICT: tl.constexpr,
        BLOCK_N: tl.constexpr, BLOCK_HALF: tl.constexpr, EVICT_SHARED: tl.constexpr, EVICT_STREAM: tl.constexpr,
    ):
        offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_vecs
        offs_h = tl.arange(0, BLOCK_HALF)
        if CONTIG_DICT:
            offs_h = tl.max_contiguous(tl.multiple_of(offs_h, BLOCK_HALF), BLOCK_HALF)
        mask_nh = mask_n[:, None] & (offs_h < half_dim)[None, :]
        acc_lo = tl.zeros([BLOCK_N, BLOCK_HALF], dtype=tl.float32)
        acc_hi = tl.zeros([BLOCK_N, BLOCK_HALF], dtype=tl.float32)
        d_rs, d_cs = ((2 * BLOCK_HALF) if CONTIG_DICT else stride_da), (1 if CONTIG_DICT else stride_dd)

        for k in range(m_atoms):
            atom_idx = tl.load(idx_ptr + offs_n * stride_in + k * stride_is, mask=mask_n, other=0, eviction_policy=EVICT_STREAM).to(tl.int32)
            coeff = tl.load(val_ptr + offs_n * stride_vn + k * stride_vs, mask=mask_n, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
            valid_atom = (atom_idx >= 0) & (atom_idx < n_atoms)
            safe_idx = tl.where(valid_atom, atom_idx, 0)
            coeff = tl.where(valid_atom, coeff, 0.0)
            base_d = dict_ptr + safe_idx[:, None] * d_rs
            acc_lo += tl.load(base_d + offs_h[None, :] * d_cs, mask=mask_nh, other=0.0, eviction_policy=EVICT_SHARED).to(tl.float32) * coeff[:, None]
            acc_hi += tl.load(base_d + (offs_h[None, :] + half_dim) * d_cs, mask=mask_nh, other=0.0, eviction_policy=EVICT_SHARED).to(tl.float32) * coeff[:, None]

        if HAS_SCALE:
            sc = tl.load(scale_ptr + offs_n * stride_sn, mask=mask_n, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
            acc_lo, acc_hi = acc_lo * sc[:, None], acc_hi * sc[:, None]
        if APPLY_ROPE:
            pos_idx = offs_n % seq_len
            c_val = tl.load(cos_ptr + pos_idx[:, None] * stride_ct + offs_h[None, :] * stride_cd, mask=mask_nh, other=1.0, eviction_policy=EVICT_SHARED).to(tl.float32)
            s_val = tl.load(sin_ptr + pos_idx[:, None] * stride_ct + offs_h[None, :] * stride_cd, mask=mask_nh, other=0.0, eviction_policy=EVICT_SHARED).to(tl.float32)
            acc_lo, acc_hi = acc_lo * c_val - acc_hi * s_val, acc_hi * c_val + acc_lo * s_val

        base_o = out_ptr + offs_n[:, None] * stride_on
        tl.store(base_o + offs_h[None, :] * stride_od, acc_lo, mask=mask_nh)
        tl.store(base_o + (offs_h[None, :] + half_dim) * stride_od, acc_hi, mask=mask_nh)

    @triton.jit
    def _fused_kv_materialize_kernel(
        k_idx_ptr, k_val_ptr, k_sc_ptr, v_idx_ptr, v_val_ptr, v_sc_ptr,
        dict_ptr, loc_k_ptr, loc_v_ptr, cos_ptr, sin_ptr, out_k_ptr, out_v_ptr, h_lex, t_hist, t_total, m_atoms, n_atoms, half_dim,
        stride_kib, stride_kih, stride_kit, stride_kis, stride_kvb, stride_kvh, stride_kvt, stride_kvs, stride_ksb, stride_ksh, stride_kst,
        stride_vib, stride_vih, stride_vit, stride_vis, stride_vvb, stride_vvh, stride_vvt, stride_vvs, stride_vsb, stride_vsh, stride_vst,
        stride_da, stride_dd, stride_lkb, stride_lkh, stride_lkt, stride_lkd, stride_lvb, stride_lvh, stride_lvt, stride_lvd,
        stride_ct, stride_cd, stride_okb, stride_okh, stride_okt, stride_okd, stride_ovb, stride_ovh, stride_ovt, stride_ovd,
        BLOCK_HALF: tl.constexpr, BLOCK_N: tl.constexpr, HAS_K_SCALE: tl.constexpr, HAS_V_SCALE: tl.constexpr,
        APPLY_ROPE: tl.constexpr, HAS_LOCAL: tl.constexpr, CONTIG_DICT: tl.constexpr, EVICT_SHARED: tl.constexpr, EVICT_STREAM: tl.constexpr,
    ):
        pid_t, pid_bh, is_k = tl.program_id(0), tl.program_id(1), (tl.program_id(2) == 0)
        b_idx, h_idx = pid_bh // h_lex, pid_bh % h_lex
        block_start = pid_t * BLOCK_N
        offs_t = block_start + tl.arange(0, BLOCK_N)
        offs_h = tl.arange(0, BLOCK_HALF)
        if CONTIG_DICT:
            offs_h = tl.max_contiguous(tl.multiple_of(offs_h, BLOCK_HALF), BLOCK_HALF)
        mask_h = offs_h < half_dim

        if block_start < t_hist:
            mask_t = offs_t < t_hist
            mask_th = mask_t[:, None] & mask_h[None, :]
            if is_k:
                idx_base = k_idx_ptr + b_idx * stride_kib + h_idx * stride_kih + offs_t * stride_kit
                val_base = k_val_ptr + b_idx * stride_kvb + h_idx * stride_kvh + offs_t * stride_kvt
                s_is, s_vs = stride_kis, stride_kvs
            else:
                idx_base = v_idx_ptr + b_idx * stride_vib + h_idx * stride_vih + offs_t * stride_vit
                val_base = v_val_ptr + b_idx * stride_vvb + h_idx * stride_vvh + offs_t * stride_vvt
                s_is, s_vs = stride_vis, stride_vvs

            acc_lo = tl.zeros([BLOCK_N, BLOCK_HALF], dtype=tl.float32)
            acc_hi = tl.zeros([BLOCK_N, BLOCK_HALF], dtype=tl.float32)
            d_rs, d_cs = ((2 * BLOCK_HALF) if CONTIG_DICT else stride_da), (1 if CONTIG_DICT else stride_dd)
            for k in range(m_atoms):
                atom_idx = tl.load(idx_base + k * s_is, mask=mask_t, other=0, eviction_policy=EVICT_STREAM).to(tl.int32)
                coeff = tl.load(val_base + k * s_vs, mask=mask_t, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
                valid_atom = (atom_idx >= 0) & (atom_idx < n_atoms)
                safe_idx = tl.where(valid_atom, atom_idx, 0)
                coeff = tl.where(valid_atom, coeff, 0.0)
                base_d = dict_ptr + safe_idx[:, None] * d_rs
                acc_lo += tl.load(base_d + offs_h[None, :] * d_cs, mask=mask_th, other=0.0, eviction_policy=EVICT_SHARED).to(tl.float32) * coeff[:, None]
                acc_hi += tl.load(base_d + (offs_h[None, :] + half_dim) * d_cs, mask=mask_th, other=0.0, eviction_policy=EVICT_SHARED).to(tl.float32) * coeff[:, None]

            if is_k:
                if HAS_K_SCALE:
                    sc = tl.load(k_sc_ptr + b_idx * stride_ksb + h_idx * stride_ksh + offs_t * stride_kst, mask=mask_t, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
                    acc_lo, acc_hi = acc_lo * sc[:, None], acc_hi * sc[:, None]
                if APPLY_ROPE:
                    c_val = tl.load(cos_ptr + offs_t[:, None] * stride_ct + offs_h[None, :] * stride_cd, mask=mask_th, other=1.0, eviction_policy=EVICT_SHARED).to(tl.float32)
                    s_val = tl.load(sin_ptr + offs_t[:, None] * stride_ct + offs_h[None, :] * stride_cd, mask=mask_th, other=0.0, eviction_policy=EVICT_SHARED).to(tl.float32)
                    acc_lo, acc_hi = acc_lo * c_val - acc_hi * s_val, acc_hi * c_val + acc_lo * s_val
                out_base, s_od = out_k_ptr + b_idx * stride_okb + h_idx * stride_okh + offs_t[:, None] * stride_okt, stride_okd
            else:
                if HAS_V_SCALE:
                    sc = tl.load(v_sc_ptr + b_idx * stride_vsb + h_idx * stride_vsh + offs_t * stride_vst, mask=mask_t, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
                    acc_lo, acc_hi = acc_lo * sc[:, None], acc_hi * sc[:, None]
                out_base, s_od = out_v_ptr + b_idx * stride_ovb + h_idx * stride_ovh + offs_t[:, None] * stride_ovt, stride_ovd

            tl.store(out_base + offs_h[None, :] * s_od, acc_lo, mask=mask_th)
            tl.store(out_base + (offs_h[None, :] + half_dim) * s_od, acc_hi, mask=mask_th)

        if HAS_LOCAL and (block_start + BLOCK_N > t_hist):
            loc_t = offs_t - t_hist
            mask_lh = ((offs_t >= t_hist) & (offs_t < t_total))[:, None] & mask_h[None, :]
            if is_k:
                l_base = loc_k_ptr + b_idx * stride_lkb + h_idx * stride_lkh + loc_t[:, None] * stride_lkt
                o_base, s_ld, s_od = out_k_ptr + b_idx * stride_okb + h_idx * stride_okh + offs_t[:, None] * stride_okt, stride_lkd, stride_okd
            else:
                l_base = loc_v_ptr + b_idx * stride_lvb + h_idx * stride_lvh + loc_t[:, None] * stride_lvt
                o_base, s_ld, s_od = out_v_ptr + b_idx * stride_ovb + h_idx * stride_ovh + offs_t[:, None] * stride_ovt, stride_lvd, stride_ovd
            tl.store(o_base + offs_h[None, :] * s_od, tl.load(l_base + offs_h[None, :] * s_ld, mask=mask_lh, other=0.0), mask=mask_lh)
            tl.store(o_base + (offs_h[None, :] + half_dim) * s_od, tl.load(l_base + (offs_h[None, :] + half_dim) * s_ld, mask=mask_lh, other=0.0), mask=mask_lh)

    @triton.jit
    def _fused_write_probe_kernel(
        k_ptr, v_ptr, u_ptr, out_ptr, h_kv, t_len, head_dim, r_rank, start_pos, sink_tokens, sink_bonus,
        stride_kb, stride_kh, stride_kt, stride_kd, stride_vb, stride_vh, stride_vt, stride_vd,
        stride_uh, stride_ud, stride_ur, stride_ob, stride_oh, stride_ot,
        DECODE_PATH: tl.constexpr, SOFTPLUS_MODE: tl.constexpr, SINGLE_D_TILE: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr, EVICT_SHARED: tl.constexpr, EVICT_STREAM: tl.constexpr,
    ):
        pid_t, pid_bh = tl.program_id(0), tl.program_id(1)
        b_idx, h_idx = pid_bh // h_kv, pid_bh % h_kv
        offs_r = tl.arange(0, BLOCK_R)
        mask_r = offs_r < r_rank
        u_base = u_ptr + h_idx * stride_uh
        d_limit = BLOCK_D if SINGLE_D_TILE else head_dim

        if DECODE_PATH:
            k_base = k_ptr + b_idx * stride_kb + h_idx * stride_kh + pid_t * stride_kt
            v_base = v_ptr + b_idx * stride_vb + h_idx * stride_vh + pid_t * stride_vt
            proj_acc, v_sq_acc = tl.zeros([BLOCK_R], dtype=tl.float32), tl.zeros([], dtype=tl.float32)
            for d0 in range(0, d_limit, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < head_dim
                k_vec = tl.load(k_base + offs_d * stride_kd, mask=mask_d, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
                v_vec = tl.load(v_base + offs_d * stride_vd, mask=mask_d, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
                u_mat = tl.load(u_base + offs_d[:, None] * stride_ud + offs_r[None, :] * stride_ur, mask=mask_d[:, None] & mask_r[None, :], other=0.0, eviction_policy=EVICT_SHARED).to(tl.float32)
                proj_acc += tl.sum(k_vec[:, None] * u_mat, axis=0)
                v_sq_acc += tl.sum(v_vec * v_vec, axis=0)
            proj_0 = tl.sum(tl.where(offs_r == 0, proj_acc, 0.0), axis=0)
            r_axis: tl.constexpr = 0
        else:
            offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
            mask_t = offs_t < t_len
            k_base = k_ptr + b_idx * stride_kb + h_idx * stride_kh + offs_t[:, None] * stride_kt
            v_base = v_ptr + b_idx * stride_vb + h_idx * stride_vh + offs_t[:, None] * stride_vt
            proj_acc, v_sq_acc = tl.zeros([BLOCK_T, BLOCK_R], dtype=tl.float32), tl.zeros([BLOCK_T], dtype=tl.float32)
            for d0 in range(0, d_limit, BLOCK_D):
                offs_d = d0 + tl.arange(0, BLOCK_D)
                mask_d = offs_d < head_dim
                mask_td = mask_t[:, None] & mask_d[None, :]
                k_tile = tl.load(k_base + offs_d[None, :] * stride_kd, mask=mask_td, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
                v_tile = tl.load(v_base + offs_d[None, :] * stride_vd, mask=mask_td, other=0.0, eviction_policy=EVICT_STREAM).to(tl.float32)
                u_mat = tl.load(u_base + offs_d[:, None] * stride_ud + offs_r[None, :] * stride_ur, mask=mask_d[:, None] & mask_r[None, :], other=0.0, eviction_policy=EVICT_SHARED).to(tl.float32)
                proj_acc += tl.dot(k_tile, u_mat, input_precision="ieee")
                v_sq_acc += tl.sum(v_tile * v_tile, axis=1)
            proj_0 = tl.sum(tl.where(offs_r[None, :] == 0, proj_acc, 0.0), axis=1)
            r_axis: tl.constexpr = 1

        v_norm = tl.sqrt(v_sq_acc)
        if SOFTPLUS_MODE:
            dir_gate = tl.where(proj_0 > 20.0, proj_0, tl.where(proj_0 < -20.0, tl.exp(proj_0), tl.log(1.0 + tl.exp(proj_0))))
            pos_p = tl.maximum(proj_acc, 0.0)
            proj_norm = tl.sqrt(tl.sum(pos_p * pos_p, axis=r_axis))
            score = dir_gate * (1.0 + 0.25 * proj_norm) * tl.where(v_norm > 0.0, tl.sqrt(tl.maximum(v_norm, 1e-6)), 0.0)
        else:
            score = tl.sigmoid(proj_0) * tl.sqrt(tl.sum(proj_acc * proj_acc, axis=r_axis)) * v_norm

        if DECODE_PATH:
            if (sink_tokens > 0) and (start_pos + pid_t < sink_tokens):
                score = score + sink_bonus
            tl.store(out_ptr + b_idx * stride_ob + h_idx * stride_oh + pid_t * stride_ot, score)
        else:
            score = tl.where((sink_tokens > 0) & ((start_pos + offs_t) < sink_tokens), score + sink_bonus, score)
            tl.store(out_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_t * stride_ot, score, mask=mask_t)


def _unwrap_values_and_scale(values: torch.Tensor, scale: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    raw = torch.Tensor._make_subclass(torch.Tensor, values) if type(values) is not torch.Tensor else values
    return raw, (scale if scale is not None else getattr(values, "scale", None))


def _pytorch_lexico_reconstruct(
    indices: torch.Tensor, raw_vals: torch.Tensor, d_layer_t: torch.Tensor, m: int,
    scale: Optional[torch.Tensor], rope_cos: Optional[torch.Tensor], rope_sin: Optional[torch.Tensor], dtype: torch.dtype,
) -> torch.Tensor:
    orig_shape = indices.shape[:-1]
    n_atoms = int(d_layer_t.shape[0])
    head_dim = int(d_layer_t.shape[1])
    if n_atoms == 0 or m <= 0:
        return torch.zeros(orig_shape + (head_dim,), device=indices.device, dtype=dtype)
    idx_m = indices[..., :m].long()
    val_m = raw_vals[..., :m].to(torch.float32)
    valid = (idx_m >= 0) & (idx_m < n_atoms)
    safe_idx = idx_m.clamp(0, n_atoms - 1)
    val_m = torch.where(valid, val_m, torch.zeros_like(val_m))
    if scale is not None:
        val_m = val_m * scale.reshape(orig_shape + (1,)).to(torch.float32)
    x_hat = (F.embedding(safe_idx, d_layer_t.to(torch.float32)) * val_m.unsqueeze(-1)).sum(dim=-2)
    if rope_cos is not None and rope_sin is not None and len(orig_shape) >= 1:
        t_len, half = int(orig_shape[-1]), int(x_hat.shape[-1]) // 2
        c_h = rope_cos[:t_len, :half].to(device=x_hat.device, dtype=torch.float32)
        s_h = rope_sin[:t_len, :half].to(device=x_hat.device, dtype=torch.float32)
        x_lo, x_hi = x_hat[..., :half], x_hat[..., half:]
        x_hat = torch.cat((x_lo * c_h - x_hi * s_h, x_hi * c_h + x_lo * s_h), dim=-1)
    return x_hat.to(dtype=dtype)


def triton_lexico_reconstruct(
    indices: torch.Tensor, values: torch.Tensor, d_layer_t: torch.Tensor,
    active_atoms: Optional[int] = None, scale: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Reconstructs `[..., head_dim]` vectors from sparse dictionary atoms (+ optional half-split RoPE)."""
    orig_shape, s_total = indices.shape[:-1], int(indices.shape[-1])
    raw_vals, sc_tensor = _unwrap_values_and_scale(values, scale)
    s_eff = min(s_total, int(raw_vals.shape[-1]))
    m = s_eff if active_atoms is None else max(0, min(int(active_atoms), s_eff))
    n_atoms, head_dim = int(d_layer_t.shape[0]), int(d_layer_t.shape[1])
    half_dim = head_dim // 2
    dtype = out_dtype if out_dtype is not None else d_layer_t.dtype

    n_vecs = 1
    for dim in orig_shape:
        n_vecs *= int(dim)
    if n_vecs == 0 or m == 0 or n_atoms == 0:
        return torch.zeros(orig_shape + (head_dim,), device=indices.device, dtype=dtype)

    apply_rope = rope_cos is not None and rope_sin is not None and len(orig_shape) >= 1
    seq_len = int(orig_shape[-1]) if apply_rope else 1
    rope_bounds_ok = (
        not apply_rope
        or (int(rope_cos.shape[0]) >= seq_len and int(rope_cos.shape[1]) >= half_dim
            and int(rope_sin.shape[0]) >= seq_len and int(rope_sin.shape[1]) >= half_dim)
    )
    if not can_use_triton(indices) or head_dim % 2 != 0 or not rope_bounds_ok:
        return _pytorch_lexico_reconstruct(indices, raw_vals, d_layer_t, m, sc_tensor, rope_cos, rope_sin, dtype)

    idx_2d = indices.view(n_vecs, s_total) if indices.is_contiguous() else indices.contiguous().view(n_vecs, s_total)
    val_2d = raw_vals.view(n_vecs, int(raw_vals.shape[-1])) if (raw_vals.stride(-1) == 1 and raw_vals.is_contiguous()) else raw_vals.contiguous().view(n_vecs, int(raw_vals.shape[-1]))
    has_scale = sc_tensor is not None
    sc_1d = (sc_tensor.view(n_vecs) if sc_tensor.is_contiguous() else sc_tensor.contiguous().view(n_vecs)) if has_scale else d_layer_t
    stride_sn = sc_1d.stride(0) if has_scale else 0

    c_ptr, s_ptr = (rope_cos, rope_sin) if apply_rope else (d_layer_t, d_layer_t)
    stride_ct, stride_cd = (rope_cos.stride(0), rope_cos.stride(1)) if apply_rope else (0, 0)

    out_2d = torch.empty((n_vecs, head_dim), device=indices.device, dtype=dtype)
    cfg = _cached_live_config(indices.device, "lexico_reconstruct", n_vecs, head_dim, d_layer_t.data_ptr() % 16 == 0)
    block_n, block_half = cfg["BLOCK_N"], max(16, _next_pow2(half_dim))
    contig_dict = (d_layer_t.stride(1) == 1) and (d_layer_t.stride(0) == head_dim) and (half_dim == block_half)

    _lexico_reconstruct_kernel[(triton.cdiv(n_vecs, block_n),)](
        idx_2d, val_2d, sc_1d, d_layer_t, c_ptr, s_ptr, out_2d, n_vecs, seq_len, m, n_atoms, half_dim,
        idx_2d.stride(0), idx_2d.stride(1), val_2d.stride(0), val_2d.stride(1), stride_sn,
        d_layer_t.stride(0), d_layer_t.stride(1), stride_ct, stride_cd, out_2d.stride(0), out_2d.stride(1),
        HAS_SCALE=has_scale, APPLY_ROPE=apply_rope, CONTIG_DICT=contig_dict, BLOCK_N=block_n, BLOCK_HALF=block_half,
        EVICT_SHARED=cfg["evict_shared"], EVICT_STREAM=cfg["evict_stream"], **cfg["launch_kwargs"],
    )
    return out_2d.view(orig_shape + (head_dim,))


def triton_fused_kv_lexico_materialize(
    k_idx: torch.Tensor, k_val: torch.Tensor, k_scale: Optional[torch.Tensor],
    v_idx: torch.Tensor, v_val: torch.Tensor, v_scale: Optional[torch.Tensor],
    d_layer_t: torch.Tensor, local_k: Optional[torch.Tensor] = None, local_v: Optional[torch.Tensor] = None,
    active_atoms: Optional[int] = None, rope_cos: Optional[torch.Tensor] = None, rope_sin: Optional[torch.Tensor] = None,
    out_k: Optional[torch.Tensor] = None, out_v: Optional[torch.Tensor] = None, out_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-launch multi-head K+V Lexico materializer (`[B, H_lex, T_hist + T_local, D]`)."""
    bsz, h_lex, t_hist, s_total = map(int, k_idx.shape)
    k_raw, k_sc = _unwrap_values_and_scale(k_val, k_scale)
    v_raw, v_sc = _unwrap_values_and_scale(v_val, v_scale)
    s_eff = min(s_total, int(v_idx.shape[-1]), int(k_raw.shape[-1]), int(v_raw.shape[-1]))
    m = s_eff if active_atoms is None else max(0, min(int(active_atoms), s_eff))
    n_atoms, head_dim = int(d_layer_t.shape[0]), int(d_layer_t.shape[1])
    half_dim = head_dim // 2
    has_local = local_k is not None and local_v is not None and int(local_k.shape[-2]) > 0
    t_local = int(local_k.shape[-2]) if has_local else 0
    t_total = t_hist + t_local

    dtype = out_k.dtype if out_k is not None else (out_dtype or (local_k.dtype if local_k is not None else d_layer_t.dtype))
    if out_k is None:
        out_k = torch.empty((bsz, h_lex, t_total, head_dim), device=k_idx.device, dtype=dtype)
    if out_v is None:
        out_v = torch.empty((bsz, h_lex, t_total, head_dim), device=v_idx.device, dtype=dtype)
    if bsz == 0 or h_lex == 0 or t_total == 0:
        return out_k, out_v

    apply_rope = rope_cos is not None and rope_sin is not None and t_hist > 0
    rope_bounds_ok = (
        not apply_rope
        or (int(rope_cos.shape[0]) >= t_hist and int(rope_cos.shape[1]) >= half_dim
            and int(rope_sin.shape[0]) >= t_hist and int(rope_sin.shape[1]) >= half_dim)
    )
    if m == 0 or n_atoms == 0 or not can_use_triton(k_idx) or head_dim % 2 != 0 or not rope_bounds_ok:
        if t_hist > 0:
            if m == 0 or n_atoms == 0:
                out_k[:, :, :t_hist, :].zero_()
                out_v[:, :, :t_hist, :].zero_()
            else:
                out_k[:, :, :t_hist, :] = _pytorch_lexico_reconstruct(k_idx, k_raw, d_layer_t, m, k_sc, rope_cos, rope_sin, dtype)
                out_v[:, :, :t_hist, :] = _pytorch_lexico_reconstruct(v_idx, v_raw, d_layer_t, m, v_sc, None, None, dtype)
        if has_local:
            out_k[:, :, t_hist:t_total, :] = local_k.to(dtype=dtype)
            out_v[:, :, t_hist:t_total, :] = local_v.to(dtype=dtype)
        return out_k, out_v

    has_k_sc, has_v_sc = k_sc is not None, v_sc is not None
    k_sc_3d = (k_sc.squeeze(-1) if k_sc.ndim == 4 else k_sc) if has_k_sc else d_layer_t
    v_sc_3d = (v_sc.squeeze(-1) if v_sc.ndim == 4 else v_sc) if has_v_sc else d_layer_t
    ks_strides, vs_strides = (k_sc_3d.stride() if has_k_sc else (0, 0, 0)), (v_sc_3d.stride() if has_v_sc else (0, 0, 0))
    c_ptr, s_ptr = (rope_cos, rope_sin) if apply_rope else (d_layer_t, d_layer_t)
    stride_ct, stride_cd = (rope_cos.stride(0), rope_cos.stride(1)) if apply_rope else (0, 0)
    lk_ptr, lv_ptr = (local_k, local_v) if has_local else (d_layer_t, d_layer_t)
    lk_strides, lv_strides = (local_k.stride() if has_local else (0, 0, 0, 0)), (local_v.stride() if has_local else (0, 0, 0, 0))

    cfg = _cached_live_config(k_idx.device, "fused_kv_materialize", bsz * h_lex * t_total, head_dim, d_layer_t.data_ptr() % 16 == 0)
    block_n, block_half = cfg["BLOCK_N"], max(16, _next_pow2(half_dim))
    contig_dict = (d_layer_t.stride(1) == 1) and (d_layer_t.stride(0) == head_dim) and (half_dim == block_half)

    _fused_kv_materialize_kernel[(triton.cdiv(t_total, block_n), bsz * h_lex, 2)](
        k_idx, k_raw, k_sc_3d, v_idx, v_raw, v_sc_3d, d_layer_t, lk_ptr, lv_ptr, c_ptr, s_ptr, out_k, out_v,
        h_lex, t_hist, t_total, m, n_atoms, half_dim, *k_idx.stride(), *k_raw.stride(), *ks_strides,
        *v_idx.stride(), *v_raw.stride(), *vs_strides, *d_layer_t.stride(), *lk_strides, *lv_strides,
        stride_ct, stride_cd, *out_k.stride(), *out_v.stride(), BLOCK_HALF=block_half, BLOCK_N=block_n,
        HAS_K_SCALE=has_k_sc, HAS_V_SCALE=has_v_sc, APPLY_ROPE=apply_rope, HAS_LOCAL=has_local, CONTIG_DICT=contig_dict,
        EVICT_SHARED=cfg["evict_shared"], EVICT_STREAM=cfg["evict_stream"], **cfg["launch_kwargs"],
    )
    return out_k, out_v


def _pytorch_write_probe(
    key_states: torch.Tensor, value_states: torch.Tensor, u_q_layer: torch.Tensor,
    start_pos: int, sink_tokens: int, sink_bonus: float, softplus_mode: bool,
) -> torch.Tensor:
    proj = torch.einsum("bhtd,hdr->bhtr", key_states.float(), u_q_layer.float())
    v_norm = torch.linalg.vector_norm(value_states.float(), ord=2, dim=-1)
    if softplus_mode:
        v_factor = torch.where(v_norm > 0, v_norm.clamp(min=1e-6).sqrt(), torch.zeros_like(v_norm))
        scores = F.softplus(proj[..., 0]) * (1.0 + 0.25 * torch.linalg.vector_norm(F.relu(proj), ord=2, dim=-1)) * v_factor
    else:
        scores = torch.sigmoid(proj[..., 0]) * torch.linalg.vector_norm(proj, ord=2, dim=-1) * v_norm
    if sink_tokens > 0 and start_pos < sink_tokens:
        sink_end = min(int(key_states.shape[-2]), max(0, sink_tokens - start_pos))
        if sink_end > 0:
            scores[:, :, :sink_end] = scores[:, :, :sink_end] + float(sink_bonus)
    return scores


def triton_fused_write_probe(
    key_states: torch.Tensor, value_states: torch.Tensor, u_q_layer: torch.Tensor,
    start_pos: int = 0, sink_tokens: int = 4, sink_bonus: float = 100.0, softplus_mode: bool = True,
) -> torch.Tensor:
    """Single-launch fused write-time directional subspace importance scorer (`[B, H, T]` in `float32`)."""
    bsz, h_kv, t_len, head_dim = map(int, key_states.shape)
    if tuple(value_states.shape) != (bsz, h_kv, t_len, head_dim):
        raise ValueError(f"value_states shape {tuple(value_states.shape)} must match key_states {tuple(key_states.shape)}.")
    if int(u_q_layer.shape[0]) != h_kv or int(u_q_layer.shape[1]) != head_dim:
        raise ValueError(f"u_q_layer shape {tuple(u_q_layer.shape)} incompatible with (h_kv={h_kv}, head_dim={head_dim}).")
    r_rank = int(u_q_layer.shape[-1])
    if bsz == 0 or h_kv == 0 or t_len == 0:
        return torch.zeros((bsz, h_kv, t_len), device=key_states.device, dtype=torch.float32)
    if not can_use_triton(key_states) or r_rank == 0 or r_rank > 64:
        return _pytorch_write_probe(key_states, value_states, u_q_layer, int(start_pos), int(sink_tokens), float(sink_bonus), bool(softplus_mode))

    out = torch.empty((bsz, h_kv, t_len), device=key_states.device, dtype=torch.float32)
    cfg = _cached_live_config(key_states.device, "fused_write_probe", t_len, head_dim, key_states.data_ptr() % 16 == 0)
    block_d, block_r = min(128, max(16, _next_pow2(head_dim))), max(16, _next_pow2(r_rank))
    decode_path = t_len < 16
    block_t = 1 if decode_path else max(16, cfg["BLOCK_N"])
    grid = (t_len if decode_path else triton.cdiv(t_len, block_t), bsz * h_kv)

    _fused_write_probe_kernel[grid](
        key_states, value_states, u_q_layer, out, h_kv, t_len, head_dim, r_rank, int(start_pos), int(sink_tokens), float(sink_bonus),
        *key_states.stride(), *value_states.stride(), *u_q_layer.stride(), *out.stride(),
        DECODE_PATH=decode_path, SOFTPLUS_MODE=bool(softplus_mode), SINGLE_D_TILE=(head_dim <= block_d),
        BLOCK_T=block_t, BLOCK_D=block_d, BLOCK_R=block_r,
        EVICT_SHARED=cfg["evict_shared"], EVICT_STREAM=cfg["evict_stream"], **cfg["launch_kwargs"],
    )
    return out
