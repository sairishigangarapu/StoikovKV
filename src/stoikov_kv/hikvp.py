"""
AtlasKV-inspired Hierarchical Key-Value Pruning (HiKVP) offloaded retrieval engine
for StoikovKV (`stoikov_kv/hikvp.py`).

Implements a 3-level balanced hierarchical page index for offloaded KV heads
(typically `ROLE_RETRIEVAL_FP16`):

- Level 2 (Leaves): Fixed-size pages (`page_size` tokens) stored in pinned CPU RAM
  (or memory-mapped NVMe files when `mmap_dir` is set), plus an unsealed partial tail
  buffer (`< page_size` newest tokens) on the target device.
- Level 1 (Intermediate nodes): Groups of `inter_fanout` consecutive leaf pages
  storing token-weighted mean centroids and elementwise min/max key envelopes.
- Level 0 (Root nodes): Groups of `inter_fanout` intermediate nodes whose centroids
  and min/max bounds (projected into the calibrated rank-`r` `U_Q` query subspace
  when `use_subspace_centroids=True`) live permanently on the target/GPU device for
  a sub-linear $O(N^{1/3})$ device index footprint.

Supports:
- Quest-style per-dimension upper-bound + centroid hierarchical pruning in pre-RoPE
  space for high (>99%) needle-in-a-haystack recall.
- Fixed-capacity GPU LRU page cache with exact hit/miss/eviction/PCIe telemetry.
- Asynchronous CUDA stream prefetching (`prefetch_async`, `prefetch_layer`) with
  graceful CPU-only fallback.
- Zero-PCIe speculative draft mode (`draft_zero_pcie=True`).
- Exact lossless rollback (`crop(max_pos)`) including unsealing split leaf pages.
- Full checkpoint serialization (`state_dict`/`from_state_dict`, `save`/`load`).
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

# ============================================================================
# Helpers
# ============================================================================

_DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


def _dtype_to_str(dtype: Optional[torch.dtype]) -> Optional[str]:
    if dtype is None:
        return None
    name = str(dtype)
    if name.startswith("torch."):
        name = name[len("torch.") :]
    return name


def _str_to_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    if name.startswith("torch."):
        name = name[len("torch.") :]
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported serialized dtype: {name!r}")
    return _DTYPE_MAP[name]


def _tensor_nbytes(t: Optional[torch.Tensor]) -> int:
    if t is None:
        return 0
    return int(t.numel() * t.element_size())


def _is_cuda_device(device: Optional[torch.device]) -> bool:
    if device is None:
        return False
    return device.type == "cuda" and torch.cuda.is_available()


def _safe_pin_memory(t: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled or t.device.type != "cpu" or t.numel() == 0:
        return t
    if not torch.cuda.is_available():
        return t
    try:
        return t.pin_memory()
    except RuntimeError:
        return t


def _maybe_stream_ctx(stream: Optional[Any], device: torch.device):
    if stream is not None and _is_cuda_device(device):
        return torch.cuda.stream(stream)
    return contextlib.nullcontext()


def _write_tensor_mmap(tensor: torch.Tensor, path: str) -> None:
    cpu_contig = tensor.detach().to(device="cpu").contiguous()
    raw_bytes = cpu_contig.view(torch.uint8).numpy().tobytes()
    with open(path, "wb") as f:
        f.write(raw_bytes)


def _read_tensor_mmap(
    path: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    if numel == 0:
        return torch.empty(shape, dtype=dtype, device="cpu")
    flat = torch.from_file(path, shared=False, size=numel, dtype=dtype)
    return flat.view(shape)


def _normalize_query(query_pre_rope: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Normalize query tensor to `[B, Q, D]` in float32."""
    if query_pre_rope.ndim == 1:
        q = query_pre_rope.unsqueeze(0).unsqueeze(0)
    elif query_pre_rope.ndim == 2:
        q = query_pre_rope.unsqueeze(1)
    elif query_pre_rope.ndim == 3:
        q = query_pre_rope
    else:
        raise ValueError(
            f"Expected query_pre_rope with 1, 2, or 3 dims, got shape {tuple(query_pre_rope.shape)}"
        )
    if int(q.shape[-1]) != int(head_dim):
        raise ValueError(
            f"Expected query head_dim={head_dim}, got {int(q.shape[-1])}"
        )
    return q.detach().float()


def _score_nodes(
    q: torch.Tensor,
    centroids: torch.Tensor,
    k_min: Optional[torch.Tensor],
    k_max: Optional[torch.Tensor],
    include_min_max_bounds: bool,
) -> torch.Tensor:
    """
    Score `M` candidate tree nodes against query tensor `q` (`[B_q, Q, D']`).

    Args:
        q: `[B_q, Q, D']` float32 query tensor.
        centroids: `[M, B_k, D']` float32 node centroids.
        k_min: Optional `[M, B_k, D']` float32 elementwise min bounds.
        k_max: Optional `[M, B_k, D']` float32 elementwise max bounds.
        include_min_max_bounds: If True and bounds provided, blends scale-normalized
            centroid dot-product with Quest-style per-dimension upper bound
            `sum_d max(q_d * k_min_d, q_d * k_max_d)`.

    Returns:
        `[M]` float32 relevance score per candidate node (max-aggregated across `B` and `Q`).
    """
    m_nodes = int(centroids.shape[0])
    if m_nodes == 0:
        return torch.empty((0,), device=q.device, dtype=torch.float32)

    b_q = int(q.shape[0])
    b_k = int(centroids.shape[1])

    # Convert centroids to [B_k, M, D'] for batched matmul
    cent_bm = centroids.transpose(0, 1).contiguous()
    if b_q != b_k:
        if b_k == 1:
            cent_bm = cent_bm.expand(b_q, -1, -1)
        elif b_q == 1:
            q = q.expand(b_k, -1, -1)
        else:
            min_b = min(b_q, b_k)
            q = q[:min_b]
            cent_bm = cent_bm[:min_b]

    q_t = q.transpose(1, 2).contiguous()  # [B, D', Q]
    s_cent = torch.bmm(cent_bm, q_t)  # [B, M, Q]

    if not include_min_max_bounds or k_min is None or k_max is None:
        return s_cent.amax(dim=(0, 2))

    k_min_bm = k_min.transpose(0, 1).contiguous()
    k_max_bm = k_max.transpose(0, 1).contiguous()
    eff_b = int(q.shape[0])
    if int(k_min_bm.shape[0]) != eff_b:
        if int(k_min_bm.shape[0]) == 1:
            k_min_bm = k_min_bm.expand(eff_b, -1, -1)
            k_max_bm = k_max_bm.expand(eff_b, -1, -1)
        else:
            k_min_bm = k_min_bm[:eff_b]
            k_max_bm = k_max_bm[:eff_b]

    # Exact identity: sum_d max(q_d * k_min_d, q_d * k_max_d)
    #               = q_+ @ k_max^T + q_- @ k_min^T (since k_max_d >= k_min_d)
    q_pos_t = torch.clamp(q_t, min=0.0)
    q_neg_t = torch.clamp(q_t, max=0.0)
    s_upper = torch.bmm(k_max_bm, q_pos_t) + torch.bmm(k_min_bm, q_neg_t)  # [B, M, Q]

    if m_nodes > 1:
        sc_std = s_cent.std(dim=1, unbiased=False, keepdim=True)
        su_std = s_upper.std(dim=1, unbiased=False, keepdim=True)
        sc_norm = (s_cent - s_cent.mean(dim=1, keepdim=True)) / sc_std.clamp_min(1e-6)
        su_norm = (s_upper - s_upper.mean(dim=1, keepdim=True)) / su_std.clamp_min(1e-6)
        combined = 0.35 * sc_norm + 0.65 * su_norm
    else:
        combined = 0.35 * s_cent + 0.65 * s_upper

    return combined.amax(dim=(0, 2))


# ============================================================================
# 1. HiKVPConfig
# ============================================================================


@dataclass
class HiKVPConfig:
    """Configuration for the HiKVP 3-level offloaded KV retrieval engine."""

    page_size: int = 64
    inter_fanout: int = 8
    top_k_roots: int = 4
    top_k_inter: int = 8
    top_k_pages: int = 4
    max_hot_pages_per_head: int = 8
    use_subspace_centroids: bool = True
    include_min_max_bounds: bool = True
    pin_memory: bool = True
    async_prefetch: bool = True
    mmap_dir: Optional[str] = None
    offload_roles: Tuple[int, ...] = (0,)  # 0 == ROLE_RETRIEVAL_FP16
    draft_zero_pcie: bool = True
    stoikov_spread_gate: bool = True
    gamma_risk: float = 0.5
    kappa_pcie: float = 1.5

    def __post_init__(self) -> None:
        if self.page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {self.page_size}")
        if self.inter_fanout < 1:
            raise ValueError(f"inter_fanout must be >= 1, got {self.inter_fanout}")
        if self.top_k_roots < 1:
            raise ValueError(f"top_k_roots must be >= 1, got {self.top_k_roots}")
        if self.top_k_inter < 1:
            raise ValueError(f"top_k_inter must be >= 1, got {self.top_k_inter}")
        if self.top_k_pages < 1:
            raise ValueError(f"top_k_pages must be >= 1, got {self.top_k_pages}")
        if self.max_hot_pages_per_head < 0:
            raise ValueError(
                f"max_hot_pages_per_head must be >= 0, got {self.max_hot_pages_per_head}"
            )
        self.offload_roles = tuple(int(r) for r in self.offload_roles)


# ============================================================================
# 2. HiKVPPage
# ============================================================================


@dataclass
class HiKVPPage:
    """Sealed leaf page holding up to `page_size` tokens in CPU RAM or mmap storage."""

    page_id: int
    start_pos: int
    length: int
    k_cpu: Optional[torch.Tensor]
    v_cpu: Optional[torch.Tensor]
    positions_cpu: torch.Tensor
    centroid_full: torch.Tensor
    k_min_full: torch.Tensor
    k_max_full: torch.Tensor
    mmap_k_path: Optional[str] = None
    mmap_v_path: Optional[str] = None
    kv_shape: Optional[Tuple[int, int, int]] = None
    kv_dtype: Optional[torch.dtype] = None

    def load_kv_cpu(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(k_cpu, v_cpu)` from pinned RAM or memory-mapped disk backing."""
        if self.k_cpu is not None and self.v_cpu is not None:
            return self.k_cpu, self.v_cpu
        if (
            self.mmap_k_path is not None
            and self.mmap_v_path is not None
            and self.kv_shape is not None
            and self.kv_dtype is not None
        ):
            k_mm = _read_tensor_mmap(self.mmap_k_path, self.kv_shape, self.kv_dtype)
            v_mm = _read_tensor_mmap(self.mmap_v_path, self.kv_shape, self.kv_dtype)
            return k_mm, v_mm
        raise RuntimeError(
            f"HiKVPPage(page_id={self.page_id}) has neither in-memory tensors nor valid mmap paths"
        )

    def leaf_nbytes(self) -> int:
        """Total offloaded leaf payload bytes (K + V + positions)."""
        if self.k_cpu is not None and self.v_cpu is not None:
            return (
                _tensor_nbytes(self.k_cpu)
                + _tensor_nbytes(self.v_cpu)
                + _tensor_nbytes(self.positions_cpu)
            )
        if self.kv_shape is not None and self.kv_dtype is not None:
            b, l_p, d = self.kv_shape
            elem_bytes = torch.empty((), dtype=self.kv_dtype).element_size()
            return int(2 * b * l_p * d * elem_bytes) + _tensor_nbytes(self.positions_cpu)
        return _tensor_nbytes(self.positions_cpu)

    def cleanup_mmap(self) -> None:
        """Delete backing mmap files if present on disk."""
        for p in (self.mmap_k_path, self.mmap_v_path):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass


# ============================================================================
# 3. GPULRUPageCache
# ============================================================================


class GPULRUPageCache:
    """
    Fixed-capacity LRU cache storing hot leaf pages `(k, v, pos)` on the target device.
    """

    def __init__(self, max_pages: int = 8) -> None:
        self.max_pages: int = max(0, int(max_pages))
        self._cache: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = (
            OrderedDict()
        )
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.pcie_bytes_transferred: int = 0

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, page_id: int) -> bool:
        return int(page_id) in self._cache

    def get(
        self, page_id: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pid = int(page_id)
        item = self._cache.get(pid)
        if item is not None:
            self._cache.move_to_end(pid)
            self.hits += 1
            return item
        self.misses += 1
        return None

    def put(
        self,
        page_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        transferred_bytes: int = 0,
    ) -> None:
        if transferred_bytes > 0:
            self.pcie_bytes_transferred += int(transferred_bytes)
        if self.max_pages <= 0:
            return
        pid = int(page_id)
        if pid in self._cache:
            self._cache[pid] = (k, v, pos)
            self._cache.move_to_end(pid)
            return
        while len(self._cache) >= self.max_pages:
            self._cache.popitem(last=False)
            self.evictions += 1
        self._cache[pid] = (k, v, pos)

    def invalidate_ge(self, min_page_id: int) -> None:
        threshold = int(min_page_id)
        to_delete = [pid for pid in self._cache.keys() if pid >= threshold]
        for pid in to_delete:
            del self._cache[pid]

    def clear(self) -> None:
        self._cache.clear()

    def all_hot_items(
        self,
    ) -> List[Tuple[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        return list(self._cache.items())

    @property
    def resident_bytes(self) -> int:
        total = 0
        for k, v, pos in self._cache.values():
            total += _tensor_nbytes(k) + _tensor_nbytes(v) + _tensor_nbytes(pos)
        return total

    @property
    def hit_rate(self) -> float:
        accesses = self.hits + self.misses
        if accesses == 0:
            return 0.0
        return float(self.hits) / float(accesses)


# ============================================================================
# 4. HierarchicalKVHeadIndex
# ============================================================================


class HierarchicalKVHeadIndex:
    """
    Per-head 3-level balanced hierarchical KV index with CPU/mmap leaf pages,
    device-resident Level 0 root index, and a fixed-capacity GPU LRU page cache.
    """

    def __init__(
        self,
        layer_idx: int,
        head_idx: int,
        head_dim: int,
        config: HiKVPConfig,
        u_q: Optional[torch.Tensor] = None,
    ) -> None:
        self.layer_idx: int = int(layer_idx)
        self.head_idx: int = int(head_idx)
        self.head_dim: int = int(head_dim)
        self.config: HiKVPConfig = config
        self._instance_uid: str = uuid.uuid4().hex[:8]

        self._u_q_cpu: Optional[torch.Tensor] = None
        self._u_q_dev: Optional[torch.Tensor] = None
        if u_q is not None:
            u_t = u_q.detach().to(device="cpu", dtype=torch.float32)
            if u_t.ndim != 2:
                raise ValueError(
                    f"Expected 2D u_q projection matrix, got shape {tuple(u_t.shape)}"
                )
            if int(u_t.shape[0]) == self.head_dim:
                self._u_q_cpu = u_t.contiguous()
            elif int(u_t.shape[1]) == self.head_dim:
                self._u_q_cpu = u_t.T.contiguous()
            else:
                raise ValueError(
                    f"u_q shape {tuple(u_t.shape)} incompatible with head_dim={self.head_dim}"
                )

        if self.config.mmap_dir is not None:
            os.makedirs(self.config.mmap_dir, exist_ok=True)

        self.lru_cache: GPULRUPageCache = GPULRUPageCache(
            max_pages=self.config.max_hot_pages_per_head
        )

        # Level 2: sealed leaf pages + unsealed partial tail buffer
        self._pages: List[HiKVPPage] = []
        self._leaf_k_min_sub_cpu: List[torch.Tensor] = []
        self._leaf_k_max_sub_cpu: List[torch.Tensor] = []
        self._sealed_tokens: int = 0

        self._tail_k: Optional[torch.Tensor] = None
        self._tail_v: Optional[torch.Tensor] = None
        self._tail_pos: Optional[torch.Tensor] = None

        # Level 1: intermediate nodes (CPU, full head_dim + subspace envelopes if active)
        self._inter_centroids_cpu: List[torch.Tensor] = []
        self._inter_k_min_cpu: List[torch.Tensor] = []
        self._inter_k_max_cpu: List[torch.Tensor] = []
        self._inter_k_min_sub_cpu: List[torch.Tensor] = []
        self._inter_k_max_sub_cpu: List[torch.Tensor] = []
        self._inter_token_counts: List[int] = []

        # Level 0: root nodes (CPU source + GPU/target-device cached tensors)
        self._root_centroids_cpu: List[torch.Tensor] = []
        self._root_k_min_cpu: List[torch.Tensor] = []
        self._root_k_max_cpu: List[torch.Tensor] = []
        self._root_start_pos_cpu: List[int] = []
        self._root_token_counts: List[int] = []

        self._index_device: torch.device = torch.device("cpu")
        self._root_centroids_dev: Optional[torch.Tensor] = None
        self._root_k_min_dev: Optional[torch.Tensor] = None
        self._root_k_max_dev: Optional[torch.Tensor] = None
        self._root_start_pos_dev: Optional[torch.Tensor] = None
        self._root_dev_dirty: bool = False

        self._batch_size: Optional[int] = None
        self._storage_dtype: Optional[torch.dtype] = None
        self._prefetch_stream: Optional[Any] = None
        self._pending_prefetch_event: Optional[Any] = None

    @property
    def _use_subspace(self) -> bool:
        return bool(self.config.use_subspace_centroids and self._u_q_cpu is not None)

    @property
    def num_pages(self) -> int:
        return len(self._pages)

    @property
    def num_tail_tokens(self) -> int:
        return int(self._tail_pos.shape[0]) if self._tail_pos is not None else 0

    @property
    def num_tokens(self) -> int:
        return self._sealed_tokens + self.num_tail_tokens

    def _get_u_q(self, device: torch.device) -> torch.Tensor:
        assert self._u_q_cpu is not None
        if device.type == "cpu":
            return self._u_q_cpu
        if self._u_q_dev is None or self._u_q_dev.device != device:
            self._u_q_dev = self._u_q_cpu.to(device=device)
        return self._u_q_dev

    def _sync_root_device(self, device: torch.device) -> None:
        """Materialize stacked Level 0 root index tensors on `device`."""
        if len(self._root_centroids_cpu) == 0:
            self._root_centroids_dev = None
            self._root_k_min_dev = None
            self._root_k_max_dev = None
            self._root_start_pos_dev = None
            self._root_dev_dirty = False
            return

        if (
            not self._root_dev_dirty
            and self._root_centroids_dev is not None
            and self._root_centroids_dev.device == device
        ):
            return

        c_stack = torch.stack(self._root_centroids_cpu, dim=0)
        self._root_centroids_dev = c_stack.to(device=device, dtype=torch.float32)
        if self.config.include_min_max_bounds:
            mn_stack = torch.stack(self._root_k_min_cpu, dim=0)
            mx_stack = torch.stack(self._root_k_max_cpu, dim=0)
            self._root_k_min_dev = mn_stack.to(device=device, dtype=torch.float32)
            self._root_k_max_dev = mx_stack.to(device=device, dtype=torch.float32)
        else:
            self._root_k_min_dev = None
            self._root_k_max_dev = None

        self._root_start_pos_dev = torch.tensor(
            self._root_start_pos_cpu, device=device, dtype=torch.long
        )
        self._root_dev_dirty = False

    def _recompute_inter_node(self, inter_idx: int) -> None:
        f = self.config.inter_fanout
        start_p = inter_idx * f
        end_p = min((inter_idx + 1) * f, len(self._pages))
        children = self._pages[start_p:end_p]
        assert len(children) > 0

        total_toks = sum(p.length for p in children)
        weighted_sum = torch.zeros_like(children[0].centroid_full)
        k_min = children[0].k_min_full.clone()
        k_max = children[0].k_max_full.clone()
        for p in children:
            weighted_sum.add_(p.centroid_full, alpha=float(p.length))
            torch.minimum(k_min, p.k_min_full, out=k_min)
            torch.maximum(k_max, p.k_max_full, out=k_max)
        centroid = weighted_sum / float(max(1, total_toks))

        if inter_idx == len(self._inter_centroids_cpu):
            self._inter_centroids_cpu.append(centroid)
            self._inter_k_min_cpu.append(k_min)
            self._inter_k_max_cpu.append(k_max)
            self._inter_token_counts.append(total_toks)
        else:
            self._inter_centroids_cpu[inter_idx] = centroid
            self._inter_k_min_cpu[inter_idx] = k_min
            self._inter_k_max_cpu[inter_idx] = k_max
            self._inter_token_counts[inter_idx] = total_toks

        if self._use_subspace:
            k_min_s = self._leaf_k_min_sub_cpu[start_p].clone()
            k_max_s = self._leaf_k_max_sub_cpu[start_p].clone()
            for p_idx in range(start_p + 1, end_p):
                torch.minimum(k_min_s, self._leaf_k_min_sub_cpu[p_idx], out=k_min_s)
                torch.maximum(k_max_s, self._leaf_k_max_sub_cpu[p_idx], out=k_max_s)
            if inter_idx == len(self._inter_k_min_sub_cpu):
                self._inter_k_min_sub_cpu.append(k_min_s)
                self._inter_k_max_sub_cpu.append(k_max_s)
            else:
                self._inter_k_min_sub_cpu[inter_idx] = k_min_s
                self._inter_k_max_sub_cpu[inter_idx] = k_max_s

    def _recompute_root_node(self, root_idx: int) -> None:
        f = self.config.inter_fanout
        start_i = root_idx * f
        end_i = min((root_idx + 1) * f, len(self._inter_centroids_cpu))
        assert end_i > start_i

        total_toks = sum(self._inter_token_counts[start_i:end_i])
        weighted_sum = torch.zeros_like(self._inter_centroids_cpu[start_i])
        for i_idx in range(start_i, end_i):
            weighted_sum.add_(
                self._inter_centroids_cpu[i_idx],
                alpha=float(self._inter_token_counts[i_idx]),
            )
        centroid_full = weighted_sum / float(max(1, total_toks))

        if self._use_subspace:
            assert self._u_q_cpu is not None
            root_cent = torch.matmul(centroid_full, self._u_q_cpu)  # [B, r]
            root_min = self._inter_k_min_sub_cpu[start_i].clone()
            root_max = self._inter_k_max_sub_cpu[start_i].clone()
            for i_idx in range(start_i + 1, end_i):
                torch.minimum(root_min, self._inter_k_min_sub_cpu[i_idx], out=root_min)
                torch.maximum(root_max, self._inter_k_max_sub_cpu[i_idx], out=root_max)
        else:
            root_cent = centroid_full
            root_min = self._inter_k_min_cpu[start_i].clone()
            root_max = self._inter_k_max_cpu[start_i].clone()
            for i_idx in range(start_i + 1, end_i):
                torch.minimum(root_min, self._inter_k_min_cpu[i_idx], out=root_min)
                torch.maximum(root_max, self._inter_k_max_cpu[i_idx], out=root_max)

        first_page_idx = root_idx * f * f
        start_pos = int(self._pages[first_page_idx].start_pos)

        if root_idx == len(self._root_centroids_cpu):
            self._root_centroids_cpu.append(root_cent)
            self._root_k_min_cpu.append(root_min)
            self._root_k_max_cpu.append(root_max)
            self._root_start_pos_cpu.append(start_pos)
            self._root_token_counts.append(total_toks)
        else:
            self._root_centroids_cpu[root_idx] = root_cent
            self._root_k_min_cpu[root_idx] = root_min
            self._root_k_max_cpu[root_idx] = root_max
            self._root_start_pos_cpu[root_idx] = start_pos
            self._root_token_counts[root_idx] = total_toks

        self._root_dev_dirty = True

    def _seal_page(
        self,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        pos_chunk: torch.Tensor,
        store_dtype: torch.dtype,
    ) -> int:
        page_id = len(self._pages)
        length = int(k_chunk.shape[1])
        start_pos = int(pos_chunk[0].item())

        k_f32 = k_chunk.detach().float()
        centroid_full = k_f32.mean(dim=1).to(device="cpu", dtype=torch.float32)
        k_min_full = k_f32.amin(dim=1).to(device="cpu", dtype=torch.float32)
        k_max_full = k_f32.amax(dim=1).to(device="cpu", dtype=torch.float32)

        if self._use_subspace:
            u_q_d = self._get_u_q(k_f32.device)
            k_sub = torch.matmul(k_f32, u_q_d)  # [B, L_p, r]
            k_min_sub = k_sub.amin(dim=1).to(device="cpu", dtype=torch.float32)
            k_max_sub = k_sub.amax(dim=1).to(device="cpu", dtype=torch.float32)
            self._leaf_k_min_sub_cpu.append(k_min_sub)
            self._leaf_k_max_sub_cpu.append(k_max_sub)

        k_host = k_chunk.detach().to(device="cpu", dtype=store_dtype).contiguous()
        v_host = v_chunk.detach().to(device="cpu", dtype=store_dtype).contiguous()
        pos_host = pos_chunk.detach().to(device="cpu", dtype=torch.long).contiguous()
        kv_shape = (int(k_host.shape[0]), int(k_host.shape[1]), int(k_host.shape[2]))

        mmap_k_path: Optional[str] = None
        mmap_v_path: Optional[str] = None
        k_cpu: Optional[torch.Tensor] = None
        v_cpu: Optional[torch.Tensor] = None

        if self.config.mmap_dir is not None:
            os.makedirs(self.config.mmap_dir, exist_ok=True)
            prefix = (
                f"hikvp_{self._instance_uid}_l{self.layer_idx}_h{self.head_idx}_p{page_id}"
            )
            mmap_k_path = os.path.join(self.config.mmap_dir, f"{prefix}_k.bin")
            mmap_v_path = os.path.join(self.config.mmap_dir, f"{prefix}_v.bin")
            _write_tensor_mmap(k_host, mmap_k_path)
            _write_tensor_mmap(v_host, mmap_v_path)
        else:
            k_cpu = _safe_pin_memory(k_host, self.config.pin_memory)
            v_cpu = _safe_pin_memory(v_host, self.config.pin_memory)
            pos_host = _safe_pin_memory(pos_host, self.config.pin_memory)

        page = HiKVPPage(
            page_id=page_id,
            start_pos=start_pos,
            length=length,
            k_cpu=k_cpu,
            v_cpu=v_cpu,
            positions_cpu=pos_host,
            centroid_full=centroid_full,
            k_min_full=k_min_full,
            k_max_full=k_max_full,
            mmap_k_path=mmap_k_path,
            mmap_v_path=mmap_v_path,
            kv_shape=kv_shape,
            kv_dtype=store_dtype,
        )
        self._pages.append(page)
        self._sealed_tokens += length
        return page_id

    def append_pre_rope(
        self,
        k_pre: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> int:
        """
        Append pre-RoPE keys `[B, T, D]`, values `[B, T, D]`, and positions `[T]`.
        Seals every full `page_size` chunk into a leaf page and incrementally updates
        Level 1 and Level 0 tree nodes.
        """
        if k_pre.ndim == 2:
            k_pre = k_pre.unsqueeze(0)
        if v.ndim == 2:
            v = v.unsqueeze(0)
        if k_pre.ndim != 3 or v.ndim != 3:
            raise ValueError(
                f"Expected 3D [B, T, D] tensors, got k={tuple(k_pre.shape)}, v={tuple(v.shape)}"
            )
        if int(k_pre.shape[-1]) != self.head_dim or int(v.shape[-1]) != self.head_dim:
            raise ValueError(
                f"Expected head_dim={self.head_dim}, got k={tuple(k_pre.shape)}, v={tuple(v.shape)}"
            )

        t_new = int(k_pre.shape[1])
        if t_new == 0:
            return self.num_tokens

        pos = positions.detach()
        if pos.ndim == 2:
            pos = pos[0]
        elif pos.ndim != 1:
            pos = pos.reshape(-1)
        if int(pos.shape[0]) != t_new:
            raise ValueError(
                f"positions length {int(pos.shape[0])} does not match T={t_new}"
            )

        in_device = k_pre.device
        if _is_cuda_device(in_device):
            self._index_device = in_device
        elif self._batch_size is None:
            self._index_device = in_device

        self._batch_size = int(k_pre.shape[0])
        store_dtype = torch.float16 if _is_cuda_device(in_device) else k_pre.dtype
        self._storage_dtype = store_dtype

        k_add = k_pre.detach().to(device=self._index_device, dtype=store_dtype)
        v_add = v.detach().to(device=self._index_device, dtype=store_dtype)
        pos_add = pos.to(device=self._index_device, dtype=torch.long)

        if self._tail_k is None:
            self._tail_k = k_add
            self._tail_v = v_add
            self._tail_pos = pos_add
        else:
            self._tail_k = torch.cat([self._tail_k, k_add], dim=1)
            self._tail_v = torch.cat([self._tail_v, v_add], dim=1)
            self._tail_pos = torch.cat([self._tail_pos, pos_add], dim=0)

        p_size = self.config.page_size
        f = self.config.inter_fanout
        affected_inter: List[int] = []
        affected_inter_set = set()
        affected_roots: List[int] = []
        affected_roots_set = set()

        tail_len = int(self._tail_pos.shape[0])
        if tail_len >= p_size:
            n_full = tail_len // p_size
            consumed = n_full * p_size
            k_full = self._tail_k[:, :consumed, :]
            v_full = self._tail_v[:, :consumed, :]
            pos_full = self._tail_pos[:consumed]

            if consumed == tail_len:
                self._tail_k = None
                self._tail_v = None
                self._tail_pos = None
            else:
                self._tail_k = self._tail_k[:, consumed:, :].contiguous()
                self._tail_v = self._tail_v[:, consumed:, :].contiguous()
                self._tail_pos = self._tail_pos[consumed:].contiguous()

            for chunk_i in range(n_full):
                s = chunk_i * p_size
                e = s + p_size
                pid = self._seal_page(
                    k_full[:, s:e, :],
                    v_full[:, s:e, :],
                    pos_full[s:e],
                    store_dtype=store_dtype,
                )
                i_idx = pid // f
                if i_idx not in affected_inter_set:
                    affected_inter_set.add(i_idx)
                    affected_inter.append(i_idx)
                r_idx = i_idx // f
                if r_idx not in affected_roots_set:
                    affected_roots_set.add(r_idx)
                    affected_roots.append(r_idx)

            for i_idx in affected_inter:
                self._recompute_inter_node(i_idx)
            for r_idx in affected_roots:
                self._recompute_root_node(r_idx)
            self._sync_root_device(self._index_device)

        return self.num_tokens

    def prune_page_ids(
        self,
        query_pre_rope: torch.Tensor,
        top_k_pages: Optional[int] = None,
    ) -> List[int]:
        """
        Execute 3-level hierarchical pruning (Level 0 roots -> Level 1 inter ->
        Level 2 leaves) and return top-k leaf page IDs in ascending chronological order.
        """
        n_pages = len(self._pages)
        if n_pages == 0:
            return []

        k_final = int(
            self.config.top_k_pages if top_k_pages is None else top_k_pages
        )
        if k_final <= 0:
            return []

        q = _normalize_query(query_pre_rope, self.head_dim)
        f = self.config.inter_fanout

        # Level 0: score root nodes on target/query device
        root_device = q.device if _is_cuda_device(q.device) else self._index_device
        self._sync_root_device(root_device)
        assert self._root_centroids_dev is not None
        n_roots = int(self._root_centroids_dev.shape[0])

        if n_roots <= self.config.top_k_roots:
            winning_roots = list(range(n_roots))
        else:
            q_root_dev = q.to(device=root_device, dtype=torch.float32)
            if self._use_subspace:
                q_root_in = torch.matmul(q_root_dev, self._get_u_q(root_device))
            else:
                q_root_in = q_root_dev
            root_scores = _score_nodes(
                q_root_in,
                self._root_centroids_dev,
                self._root_k_min_dev,
                self._root_k_max_dev,
                include_min_max_bounds=self.config.include_min_max_bounds,
            )
            k_r = min(self.config.top_k_roots, n_roots)
            winning_roots = torch.topk(root_scores, k=k_r, largest=True).indices.tolist()

        # Level 1: expand ONLY children of winning roots
        n_inter = len(self._inter_centroids_cpu)
        cand_inter: List[int] = []
        for r_idx in winning_roots:
            s_i = r_idx * f
            e_i = min((r_idx + 1) * f, n_inter)
            cand_inter.extend(range(s_i, e_i))

        q_cpu: Optional[torch.Tensor] = None
        if len(cand_inter) <= self.config.top_k_inter:
            winning_inter = cand_inter
        else:
            q_cpu = q.to(device="cpu", dtype=torch.float32)
            c_i = torch.stack(
                [self._inter_centroids_cpu[idx] for idx in cand_inter], dim=0
            )
            if self.config.include_min_max_bounds:
                mn_i = torch.stack(
                    [self._inter_k_min_cpu[idx] for idx in cand_inter], dim=0
                )
                mx_i = torch.stack(
                    [self._inter_k_max_cpu[idx] for idx in cand_inter], dim=0
                )
            else:
                mn_i = None
                mx_i = None
            inter_scores = _score_nodes(
                q_cpu,
                c_i,
                mn_i,
                mx_i,
                include_min_max_bounds=self.config.include_min_max_bounds,
            )
            k_i = min(self.config.top_k_inter, len(cand_inter))
            top_local_i = torch.topk(inter_scores, k=k_i, largest=True).indices.tolist()
            winning_inter = [cand_inter[j] for j in top_local_i]

        # Level 2: expand ONLY leaf pages of winning inter nodes
        cand_pages: List[int] = []
        for i_idx in winning_inter:
            s_p = i_idx * f
            e_p = min((i_idx + 1) * f, n_pages)
            cand_pages.extend(range(s_p, e_p))

        if len(cand_pages) <= k_final:
            return sorted(cand_pages)

        if q_cpu is None:
            q_cpu = q.to(device="cpu", dtype=torch.float32)
        c_p = torch.stack(
            [self._pages[p_idx].centroid_full for p_idx in cand_pages], dim=0
        )
        if self.config.include_min_max_bounds:
            mn_p = torch.stack(
                [self._pages[p_idx].k_min_full for p_idx in cand_pages], dim=0
            )
            mx_p = torch.stack(
                [self._pages[p_idx].k_max_full for p_idx in cand_pages], dim=0
            )
        else:
            mn_p = None
            mx_p = None
        page_scores = _score_nodes(
            q_cpu,
            c_p,
            mn_p,
            mx_p,
            include_min_max_bounds=self.config.include_min_max_bounds,
        )
        k_p = min(k_final, len(cand_pages))
        top_vals, top_idx_t = torch.topk(page_scores, k=k_p, largest=True)
        if top_k_pages is None and self.config.stoikov_spread_gate and k_p > 1 and len(cand_pages) > 1:
            # Avellaneda-Stoikov optimal half-spread gate: prune low-relevance tail pages
            # when top page score dominates, saving PCIe page transfers without hurting recall.
            import math
            s_std = float(page_scores.std(unbiased=False).item())
            if s_std > 1e-6:
                g = max(float(self.config.gamma_risk), 1e-4)
                kp = max(float(self.config.kappa_pcie), 1e-4)
                sigma_norm_sq = (s_std * s_std) / (s_std * s_std + 1.0)
                half_spread = 0.5 * g * sigma_norm_sq + (1.0 / g) * math.log(1.0 + g / kp)
                cutoff = float(top_vals[0].item()) - (2.0 * half_spread * s_std)
                keep_mask = top_vals >= cutoff
                keep_mask[0] = True
                top_local_p = top_idx_t[keep_mask].tolist()
            else:
                top_local_p = top_idx_t.tolist()
        else:
            top_local_p = top_idx_t.tolist()
        return sorted(cand_pages[j] for j in top_local_p)

    def _load_page_to_device(
        self,
        page_id: int,
        device: torch.device,
        dtype: torch.dtype,
        stream: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        page = self._pages[page_id]
        k_cpu, v_cpu = page.load_kv_cpu()
        pos_cpu = page.positions_cpu
        nbytes = (
            _tensor_nbytes(k_cpu) + _tensor_nbytes(v_cpu) + _tensor_nbytes(pos_cpu)
        )

        is_cuda = _is_cuda_device(device)
        non_blocking = bool(is_cuda and self.config.pin_memory and k_cpu.is_pinned())
        with _maybe_stream_ctx(stream, device):
            k_dev = k_cpu.to(device=device, dtype=dtype, non_blocking=non_blocking)
            v_dev = v_cpu.to(device=device, dtype=dtype, non_blocking=non_blocking)
            pos_dev = pos_cpu.to(
                device=device, dtype=torch.long, non_blocking=non_blocking
            )
            # Detach from mmap file view when keeping an in-RAM CPU LRU entry
            if not is_cuda and page.mmap_k_path is not None:
                k_dev = k_dev.clone()
                v_dev = v_dev.clone()
        return k_dev, v_dev, pos_dev, nbytes

    def _get_or_create_prefetch_stream(self, device: torch.device) -> Optional[Any]:
        if not self.config.async_prefetch or not _is_cuda_device(device):
            return None
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.cuda.Stream(device=device)
        return self._prefetch_stream

    def prefetch_async(
        self,
        query_pre_rope: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        top_k_pages: Optional[int] = None,
        stream: Optional[Any] = None,
    ) -> List[int]:
        """
        Prune candidate pages and asynchronously DMA-load any LRU misses onto `device`.
        """
        winning_pages = self.prune_page_ids(query_pre_rope, top_k_pages=top_k_pages)
        if not winning_pages:
            return []
        dev = torch.device(device)
        use_stream = (
            stream if stream is not None else self._get_or_create_prefetch_stream(dev)
        )
        loaded_any = False
        for pid in winning_pages:
            if pid not in self.lru_cache:
                k_p, v_p, pos_p, nbytes = self._load_page_to_device(
                    pid, device=dev, dtype=dtype, stream=use_stream
                )
                self.lru_cache.put(pid, k_p, v_p, pos_p, transferred_bytes=nbytes)
                loaded_any = True
        if loaded_any and use_stream is not None and _is_cuda_device(dev):
            evt = torch.cuda.Event()
            evt.record(use_stream)
            self._pending_prefetch_event = evt
        return winning_pages

    def retrieve(
        self,
        query_pre_rope: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        top_k_pages: Optional[int] = None,
        stream: Optional[Any] = None,
        draft_zero_pcie: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Retrieve relevant KV pages + unsealed tail buffer for `query_pre_rope`.

        When `draft_zero_pcie=True`, never performs CPU->GPU page transfers; uses only
        currently hot GPU LRU pages (or GPU-resident root centroid summaries if LRU is
        empty) plus the unsealed tail buffer, recording `pcie_bytes=0`.
        """
        dev = torch.device(device)
        bsz = (
            self._batch_size
            if self._batch_size is not None
            else (int(query_pre_rope.shape[0]) if query_pre_rope.ndim >= 2 else 1)
        )

        if self.num_tokens == 0:
            k_empty = torch.empty((bsz, 0, self.head_dim), device=dev, dtype=dtype)
            v_empty = torch.empty((bsz, 0, self.head_dim), device=dev, dtype=dtype)
            pos_empty = torch.empty((0,), device=dev, dtype=torch.long)
            stats = {
                "pages_selected": 0,
                "lru_hits": 0,
                "lru_misses": 0,
                "pcie_bytes": 0,
                "tokens_returned": 0,
                "draft_zero_pcie": int(bool(draft_zero_pcie)),
            }
            return k_empty, v_empty, pos_empty, stats

        k_limit = int(
            self.config.top_k_pages if top_k_pages is None else top_k_pages
        )

        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        pos_parts: List[torch.Tensor] = []
        step_hits = 0
        step_misses = 0
        step_pcie_bytes = 0
        pages_selected = 0

        if draft_zero_pcie:
            hot_items = self.lru_cache.all_hot_items()
            if hot_items:
                hot_ids = [pid for pid, _ in hot_items if pid < len(self._pages)]
                if len(hot_ids) > k_limit and k_limit > 0:
                    q_cpu = _normalize_query(query_pre_rope, self.head_dim).to(
                        device="cpu", dtype=torch.float32
                    )
                    c_h = torch.stack(
                        [self._pages[pid].centroid_full for pid in hot_ids], dim=0
                    )
                    if self.config.include_min_max_bounds:
                        mn_h = torch.stack(
                            [self._pages[pid].k_min_full for pid in hot_ids], dim=0
                        )
                        mx_h = torch.stack(
                            [self._pages[pid].k_max_full for pid in hot_ids], dim=0
                        )
                    else:
                        mn_h = None
                        mx_h = None
                    hot_scores = _score_nodes(
                        q_cpu,
                        c_h,
                        mn_h,
                        mx_h,
                        include_min_max_bounds=self.config.include_min_max_bounds,
                    )
                    top_local = torch.topk(
                        hot_scores, k=min(k_limit, len(hot_ids)), largest=True
                    ).indices.tolist()
                    selected_hot = sorted(hot_ids[j] for j in top_local)
                else:
                    selected_hot = sorted(hot_ids)

                pages_selected = len(selected_hot)
                for pid in selected_hot:
                    cached = self.lru_cache.get(pid)
                    if cached is None:
                        continue
                    k_p, v_p, pos_p = cached
                    step_hits += 1
                    if k_p.device != dev or k_p.dtype != dtype:
                        k_p = k_p.to(device=dev, dtype=dtype)
                        v_p = v_p.to(device=dev, dtype=dtype)
                        pos_p = pos_p.to(device=dev, dtype=torch.long)
                    k_parts.append(k_p)
                    v_parts.append(v_p)
                    pos_parts.append(pos_p)
            elif len(self._pages) > 0:
                # Zero-PCIe fallback when LRU is cold: use GPU-resident root centroids
                # as ultra-compact summary KV proxies.
                self._sync_root_device(dev)
                assert self._root_centroids_dev is not None
                assert self._root_start_pos_dev is not None
                if self._use_subspace:
                    u_q_d = self._get_u_q(dev)
                    k_proxy = torch.matmul(
                        self._root_centroids_dev, u_q_d.T
                    ).transpose(0, 1)  # [B, N_roots, D]
                else:
                    k_proxy = self._root_centroids_dev.transpose(0, 1)
                k_proxy = k_proxy.to(device=dev, dtype=dtype).contiguous()
                v_proxy = torch.zeros_like(k_proxy)
                pos_proxy = self._root_start_pos_dev.to(device=dev, dtype=torch.long)
                k_parts.append(k_proxy)
                v_parts.append(v_proxy)
                pos_parts.append(pos_proxy)
        else:
            winning_pages = self.prune_page_ids(
                query_pre_rope, top_k_pages=top_k_pages
            )
            pages_selected = len(winning_pages)
            for pid in winning_pages:
                cached = self.lru_cache.get(pid)
                if cached is not None:
                    k_p, v_p, pos_p = cached
                    step_hits += 1
                    if k_p.device != dev or k_p.dtype != dtype:
                        k_p = k_p.to(device=dev, dtype=dtype)
                        v_p = v_p.to(device=dev, dtype=dtype)
                        pos_p = pos_p.to(device=dev, dtype=torch.long)
                else:
                    step_misses += 1
                    k_p, v_p, pos_p, nbytes = self._load_page_to_device(
                        pid, device=dev, dtype=dtype, stream=stream
                    )
                    step_pcie_bytes += nbytes
                    self.lru_cache.put(pid, k_p, v_p, pos_p, transferred_bytes=nbytes)
                k_parts.append(k_p)
                v_parts.append(v_p)
                pos_parts.append(pos_p)

            if stream is not None and _is_cuda_device(dev) and step_misses > 0:
                torch.cuda.current_stream(dev).wait_stream(stream)

        if self._pending_prefetch_event is not None and _is_cuda_device(dev):
            torch.cuda.current_stream(dev).wait_event(self._pending_prefetch_event)
            self._pending_prefetch_event = None

        # Append unsealed tail buffer (already resident on index device)
        if self._tail_k is not None and self._tail_v is not None and self._tail_pos is not None:
            k_parts.append(self._tail_k.to(device=dev, dtype=dtype))
            v_parts.append(self._tail_v.to(device=dev, dtype=dtype))
            pos_parts.append(self._tail_pos.to(device=dev, dtype=torch.long))

        if k_parts:
            k_ret = torch.cat(k_parts, dim=1) if len(k_parts) > 1 else k_parts[0]
            v_ret = torch.cat(v_parts, dim=1) if len(v_parts) > 1 else v_parts[0]
            pos_ret = torch.cat(pos_parts, dim=0) if len(pos_parts) > 1 else pos_parts[0]
        else:
            k_ret = torch.empty((bsz, 0, self.head_dim), device=dev, dtype=dtype)
            v_ret = torch.empty((bsz, 0, self.head_dim), device=dev, dtype=dtype)
            pos_ret = torch.empty((0,), device=dev, dtype=torch.long)

        stats = {
            "pages_selected": pages_selected,
            "lru_hits": step_hits,
            "lru_misses": step_misses,
            "pcie_bytes": step_pcie_bytes,
            "tokens_returned": int(pos_ret.shape[0]),
            "draft_zero_pcie": int(bool(draft_zero_pcie)),
        }
        return k_ret, v_ret, pos_ret, stats

    def crop(self, max_pos: int) -> None:
        """
        Exact lossless rollback dropping all tokens with `pos >= max_pos`.
        Unseals any partially split leaf page back into the tail buffer, invalidates
        affected LRU entries, and trims/updates Level 1 and Level 0 tree nodes.
        """
        cutoff = int(max_pos)
        if cutoff <= 0:
            for page in self._pages:
                page.cleanup_mmap()
            self._pages.clear()
            self._leaf_k_min_sub_cpu.clear()
            self._leaf_k_max_sub_cpu.clear()
            self._sealed_tokens = 0
            self._tail_k = None
            self._tail_v = None
            self._tail_pos = None
            self._inter_centroids_cpu.clear()
            self._inter_k_min_cpu.clear()
            self._inter_k_max_cpu.clear()
            self._inter_k_min_sub_cpu.clear()
            self._inter_k_max_sub_cpu.clear()
            self._inter_token_counts.clear()
            self._root_centroids_cpu.clear()
            self._root_k_min_cpu.clear()
            self._root_k_max_cpu.clear()
            self._root_start_pos_cpu.clear()
            self._root_token_counts.clear()
            self._root_centroids_dev = None
            self._root_k_min_dev = None
            self._root_k_max_dev = None
            self._root_start_pos_dev = None
            self._root_dev_dirty = False
            self.lru_cache.clear()
            return

        # 1. Trim unsealed tail buffer
        if self._tail_pos is not None:
            keep_tail = self._tail_pos < cutoff
            if not bool(keep_tail.all().item()):
                if bool(keep_tail.any().item()):
                    self._tail_k = self._tail_k[:, keep_tail, :].contiguous()
                    self._tail_v = self._tail_v[:, keep_tail, :].contiguous()
                    self._tail_pos = self._tail_pos[keep_tail].contiguous()
                else:
                    self._tail_k = None
                    self._tail_v = None
                    self._tail_pos = None

        # 2. Identify earliest affected sealed page (walk backwards)
        n_pages = len(self._pages)
        first_affected = n_pages
        for pid in range(n_pages - 1, -1, -1):
            if bool((self._pages[pid].positions_cpu >= cutoff).any().item()):
                first_affected = pid
            else:
                break

        if first_affected == n_pages:
            return

        # 3. Unseal surviving tokens (< cutoff) from affected pages back into tail
        unsealed_k: List[torch.Tensor] = []
        unsealed_v: List[torch.Tensor] = []
        unsealed_pos: List[torch.Tensor] = []
        tail_dtype = (
            self._storage_dtype
            if self._storage_dtype is not None
            else (torch.float16 if _is_cuda_device(self._index_device) else torch.float32)
        )

        for pid in range(first_affected, n_pages):
            page = self._pages[pid]
            keep_mask = page.positions_cpu < cutoff
            if bool(keep_mask.any().item()):
                k_cpu, v_cpu = page.load_kv_cpu()
                unsealed_k.append(
                    k_cpu[:, keep_mask, :].to(
                        device=self._index_device, dtype=tail_dtype
                    )
                )
                unsealed_v.append(
                    v_cpu[:, keep_mask, :].to(
                        device=self._index_device, dtype=tail_dtype
                    )
                )
                unsealed_pos.append(
                    page.positions_cpu[keep_mask].to(
                        device=self._index_device, dtype=torch.long
                    )
                )
            page.cleanup_mmap()

        # Truncate leaf list
        self._pages = self._pages[:first_affected]
        if self._use_subspace:
            self._leaf_k_min_sub_cpu = self._leaf_k_min_sub_cpu[:first_affected]
            self._leaf_k_max_sub_cpu = self._leaf_k_max_sub_cpu[:first_affected]
        self._sealed_tokens = sum(p.length for p in self._pages)

        # Prepend unsealed tokens before any surviving tail tokens
        if unsealed_k:
            if self._tail_k is not None:
                unsealed_k.append(self._tail_k)
                unsealed_v.append(self._tail_v)
                unsealed_pos.append(self._tail_pos)
            self._tail_k = (
                torch.cat(unsealed_k, dim=1) if len(unsealed_k) > 1 else unsealed_k[0]
            )
            self._tail_v = (
                torch.cat(unsealed_v, dim=1) if len(unsealed_v) > 1 else unsealed_v[0]
            )
            self._tail_pos = (
                torch.cat(unsealed_pos, dim=0)
                if len(unsealed_pos) > 1
                else unsealed_pos[0]
            )

        # 4. Invalidate affected GPU LRU cache pages
        self.lru_cache.invalidate_ge(first_affected)

        # 5. Trim and incrementally update Level 1 & Level 0 nodes
        if first_affected == 0:
            self._inter_centroids_cpu.clear()
            self._inter_k_min_cpu.clear()
            self._inter_k_max_cpu.clear()
            self._inter_k_min_sub_cpu.clear()
            self._inter_k_max_sub_cpu.clear()
            self._inter_token_counts.clear()
            self._root_centroids_cpu.clear()
            self._root_k_min_cpu.clear()
            self._root_k_max_cpu.clear()
            self._root_start_pos_cpu.clear()
            self._root_token_counts.clear()
            self._root_centroids_dev = None
            self._root_k_min_dev = None
            self._root_k_max_dev = None
            self._root_start_pos_dev = None
            self._root_dev_dirty = False
        else:
            f = self.config.inter_fanout
            new_n_inter = (first_affected + f - 1) // f
            self._inter_centroids_cpu = self._inter_centroids_cpu[:new_n_inter]
            self._inter_k_min_cpu = self._inter_k_min_cpu[:new_n_inter]
            self._inter_k_max_cpu = self._inter_k_max_cpu[:new_n_inter]
            self._inter_token_counts = self._inter_token_counts[:new_n_inter]
            if self._use_subspace:
                self._inter_k_min_sub_cpu = self._inter_k_min_sub_cpu[:new_n_inter]
                self._inter_k_max_sub_cpu = self._inter_k_max_sub_cpu[:new_n_inter]
            self._recompute_inter_node(new_n_inter - 1)

            new_n_roots = (new_n_inter + f - 1) // f
            self._root_centroids_cpu = self._root_centroids_cpu[:new_n_roots]
            self._root_k_min_cpu = self._root_k_min_cpu[:new_n_roots]
            self._root_k_max_cpu = self._root_k_max_cpu[:new_n_roots]
            self._root_start_pos_cpu = self._root_start_pos_cpu[:new_n_roots]
            self._root_token_counts = self._root_token_counts[:new_n_roots]
            self._recompute_root_node(new_n_roots - 1)
            self._sync_root_device(self._index_device)

    # ------------------------------------------------------------------
    # Exact Memory Accounting
    # ------------------------------------------------------------------

    def gpu_index_bytes(self) -> int:
        """Exact bytes of the device-resident Level 0 root index tensors."""
        if len(self._root_centroids_cpu) == 0:
            return 0
        if self._root_centroids_dev is None or self._root_dev_dirty:
            self._sync_root_device(self._index_device)
        return (
            _tensor_nbytes(self._root_centroids_dev)
            + _tensor_nbytes(self._root_k_min_dev)
            + _tensor_nbytes(self._root_k_max_dev)
            + _tensor_nbytes(self._root_start_pos_dev)
        )

    def gpu_lru_bytes(self) -> int:
        """Exact bytes currently resident in this head's GPU LRU page cache."""
        return self.lru_cache.resident_bytes

    def cpu_leaf_bytes(self) -> int:
        """Exact bytes of offloaded leaf pages stored in CPU RAM or mmap backing."""
        return sum(p.leaf_nbytes() for p in self._pages)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        serialized_pages: List[Dict[str, Any]] = []
        for p in self._pages:
            k_cpu, v_cpu = p.load_kv_cpu()
            serialized_pages.append(
                {
                    "page_id": p.page_id,
                    "start_pos": p.start_pos,
                    "length": p.length,
                    "k_cpu": k_cpu.detach().cpu().clone(),
                    "v_cpu": v_cpu.detach().cpu().clone(),
                    "positions_cpu": p.positions_cpu.detach().cpu().clone(),
                    "centroid_full": p.centroid_full.detach().cpu().clone(),
                    "k_min_full": p.k_min_full.detach().cpu().clone(),
                    "k_max_full": p.k_max_full.detach().cpu().clone(),
                }
            )
        return {
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "head_dim": self.head_dim,
            "config": asdict(self.config),
            "batch_size": self._batch_size,
            "storage_dtype": _dtype_to_str(self._storage_dtype),
            "u_q": self._u_q_cpu.clone() if self._u_q_cpu is not None else None,
            "pages": serialized_pages,
            "tail_k": self._tail_k.detach().cpu().clone()
            if self._tail_k is not None
            else None,
            "tail_v": self._tail_v.detach().cpu().clone()
            if self._tail_v is not None
            else None,
            "tail_pos": self._tail_pos.detach().cpu().clone()
            if self._tail_pos is not None
            else None,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, Any],
        u_q: Optional[torch.Tensor] = None,
    ) -> "HierarchicalKVHeadIndex":
        cfg_dict = dict(state["config"])
        if "offload_roles" in cfg_dict:
            cfg_dict["offload_roles"] = tuple(cfg_dict["offload_roles"])
        config = HiKVPConfig(**cfg_dict)
        eff_u_q = u_q if u_q is not None else state.get("u_q")
        obj = cls(
            layer_idx=int(state["layer_idx"]),
            head_idx=int(state["head_idx"]),
            head_dim=int(state["head_dim"]),
            config=config,
            u_q=eff_u_q,
        )
        obj._batch_size = state.get("batch_size")
        obj._storage_dtype = _str_to_dtype(state.get("storage_dtype"))

        f = config.inter_fanout
        for p_dict in state.get("pages", []):
            k_t: torch.Tensor = p_dict["k_cpu"].to(device="cpu").contiguous()
            v_t: torch.Tensor = p_dict["v_cpu"].to(device="cpu").contiguous()
            pos_t: torch.Tensor = p_dict["positions_cpu"].to(
                device="cpu", dtype=torch.long
            ).contiguous()
            centroid_full = p_dict["centroid_full"].to(device="cpu", dtype=torch.float32)
            k_min_full = p_dict["k_min_full"].to(device="cpu", dtype=torch.float32)
            k_max_full = p_dict["k_max_full"].to(device="cpu", dtype=torch.float32)
            kv_shape = (int(k_t.shape[0]), int(k_t.shape[1]), int(k_t.shape[2]))
            kv_dtype = k_t.dtype
            if obj._batch_size is None:
                obj._batch_size = kv_shape[0]
            if obj._storage_dtype is None:
                obj._storage_dtype = kv_dtype

            if obj._use_subspace:
                assert obj._u_q_cpu is not None
                k_sub = torch.matmul(k_t.float(), obj._u_q_cpu)
                obj._leaf_k_min_sub_cpu.append(k_sub.amin(dim=1))
                obj._leaf_k_max_sub_cpu.append(k_sub.amax(dim=1))

            mmap_k_path: Optional[str] = None
            mmap_v_path: Optional[str] = None
            k_cpu: Optional[torch.Tensor] = None
            v_cpu: Optional[torch.Tensor] = None
            page_id = len(obj._pages)

            if config.mmap_dir is not None:
                os.makedirs(config.mmap_dir, exist_ok=True)
                prefix = (
                    f"hikvp_{obj._instance_uid}_l{obj.layer_idx}_h{obj.head_idx}_p{page_id}"
                )
                mmap_k_path = os.path.join(config.mmap_dir, f"{prefix}_k.bin")
                mmap_v_path = os.path.join(config.mmap_dir, f"{prefix}_v.bin")
                _write_tensor_mmap(k_t, mmap_k_path)
                _write_tensor_mmap(v_t, mmap_v_path)
            else:
                k_cpu = _safe_pin_memory(k_t, config.pin_memory)
                v_cpu = _safe_pin_memory(v_t, config.pin_memory)
                pos_t = _safe_pin_memory(pos_t, config.pin_memory)

            obj._pages.append(
                HiKVPPage(
                    page_id=page_id,
                    start_pos=int(p_dict["start_pos"]),
                    length=int(p_dict["length"]),
                    k_cpu=k_cpu,
                    v_cpu=v_cpu,
                    positions_cpu=pos_t,
                    centroid_full=centroid_full,
                    k_min_full=k_min_full,
                    k_max_full=k_max_full,
                    mmap_k_path=mmap_k_path,
                    mmap_v_path=mmap_v_path,
                    kv_shape=kv_shape,
                    kv_dtype=kv_dtype,
                )
            )
            obj._sealed_tokens += int(p_dict["length"])

        if obj._pages:
            n_inter = (len(obj._pages) + f - 1) // f
            for i_idx in range(n_inter):
                obj._recompute_inter_node(i_idx)
            n_roots = (n_inter + f - 1) // f
            for r_idx in range(n_roots):
                obj._recompute_root_node(r_idx)
            obj._sync_root_device(obj._index_device)

        if state.get("tail_k") is not None:
            obj._tail_k = state["tail_k"].to(device=obj._index_device)
            obj._tail_v = state["tail_v"].to(device=obj._index_device)
            obj._tail_pos = state["tail_pos"].to(
                device=obj._index_device, dtype=torch.long
            )
            if obj._batch_size is None:
                obj._batch_size = int(obj._tail_k.shape[0])

        return obj


# ============================================================================
# 5. HiKVPStore
# ============================================================================


class HiKVPStore:
    """
    Multi-layer, multi-head store managing per-`(layer_idx, head_idx)`
    `HierarchicalKVHeadIndex` instances for offloaded head roles.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        head_roles: Optional[torch.Tensor] = None,
        config: Optional[HiKVPConfig] = None,
        U_Q: Optional[torch.Tensor] = None,
    ) -> None:
        self.num_layers: int = int(num_layers)
        self.num_kv_heads: int = int(num_kv_heads)
        self.head_dim: int = int(head_dim)
        self.config: HiKVPConfig = config if config is not None else HiKVPConfig()

        if head_roles is None:
            roles_t = torch.zeros(
                (self.num_layers, self.num_kv_heads), dtype=torch.long
            )
        else:
            roles_t = (
                head_roles.detach().to(device="cpu", dtype=torch.long)
                if isinstance(head_roles, torch.Tensor)
                else torch.as_tensor(head_roles, dtype=torch.long, device="cpu")
            )
            if tuple(roles_t.shape) != (self.num_layers, self.num_kv_heads):
                raise ValueError(
                    f"head_roles shape {tuple(roles_t.shape)} != "
                    f"({self.num_layers}, {self.num_kv_heads})"
                )
        self.head_roles: torch.Tensor = roles_t

        self._heads: Dict[Tuple[int, int], HierarchicalKVHeadIndex] = {}
        self._layer_heads: Dict[int, List[int]] = {
            l_idx: [] for l_idx in range(self.num_layers)
        }
        self._draft_mode: bool = False
        self._prefetch_streams: Dict[int, Any] = {}

        offload_set = set(self.config.offload_roles)
        for l_idx in range(self.num_layers):
            for h_idx in range(self.num_kv_heads):
                if int(self.head_roles[l_idx, h_idx].item()) in offload_set:
                    u_q_lh: Optional[torch.Tensor] = None
                    if U_Q is not None:
                        u_q_lh = U_Q[l_idx, h_idx]
                    self._heads[(l_idx, h_idx)] = HierarchicalKVHeadIndex(
                        layer_idx=l_idx,
                        head_idx=h_idx,
                        head_dim=self.head_dim,
                        config=self.config,
                        u_q=u_q_lh,
                    )
                    self._layer_heads[l_idx].append(h_idx)

    @property
    def draft_mode(self) -> bool:
        return self._draft_mode

    def is_offloaded(self, layer_idx: int, head_idx: int) -> bool:
        return (int(layer_idx), int(head_idx)) in self._heads

    def get_head(
        self, layer_idx: int, head_idx: int
    ) -> Optional[HierarchicalKVHeadIndex]:
        return self._heads.get((int(layer_idx), int(head_idx)))

    def set_draft_mode(self, enabled: bool) -> None:
        self._draft_mode = bool(enabled)

    def append_pre_rope(
        self,
        layer_idx: int,
        head_idx: int,
        k_pre: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
    ) -> int:
        key = (int(layer_idx), int(head_idx))
        head = self._heads.get(key)
        if head is None:
            raise KeyError(
                f"Head (layer={layer_idx}, head={head_idx}) is not offloaded in HiKVPStore"
            )
        return head.append_pre_rope(k_pre, v, positions)

    def retrieve(
        self,
        layer_idx: int,
        head_idx: int,
        query_pre_rope: torch.Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        top_k_pages: Optional[int] = None,
        stream: Optional[Any] = None,
        draft_zero_pcie: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        key = (int(layer_idx), int(head_idx))
        head = self._heads.get(key)
        if head is None:
            raise KeyError(
                f"Head (layer={layer_idx}, head={head_idx}) is not offloaded in HiKVPStore"
            )
        eff_device = query_pre_rope.device if device is None else torch.device(device)
        eff_dtype = query_pre_rope.dtype if dtype is None else dtype
        eff_draft = (
            bool(self._draft_mode and self.config.draft_zero_pcie)
            if draft_zero_pcie is None
            else bool(draft_zero_pcie)
        )
        return head.retrieve(
            query_pre_rope=query_pre_rope,
            device=eff_device,
            dtype=eff_dtype,
            top_k_pages=top_k_pages,
            stream=stream,
            draft_zero_pcie=eff_draft,
        )

    def _get_layer_stream(self, device: torch.device) -> Optional[Any]:
        if not self.config.async_prefetch or not _is_cuda_device(device):
            return None
        dev_idx = 0 if device.index is None else int(device.index)
        if dev_idx not in self._prefetch_streams:
            self._prefetch_streams[dev_idx] = torch.cuda.Stream(device=device)
        return self._prefetch_streams[dev_idx]

    def prefetch_layer(
        self,
        layer_idx: int,
        query_pre_rope: Union[torch.Tensor, Dict[int, torch.Tensor]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        top_k_pages: Optional[int] = None,
        stream: Optional[Any] = None,
    ) -> Dict[int, List[int]]:
        """
        Asynchronously prefetch winning leaf pages for all offloaded heads in `layer_idx`.

        `query_pre_rope` may be:
        - `[B, H_q, Q, D]` (supports GQA grouping where `H_q` is a multiple of `num_kv_heads`),
        - `[B, Q, D]` or `[B, D]` (broadcast to all offloaded heads in the layer), or
        - `Dict[int, torch.Tensor]` mapping `head_idx -> query tensor`.
        """
        if self._draft_mode and self.config.draft_zero_pcie:
            return {}

        l_idx = int(layer_idx)
        head_indices = self._layer_heads.get(l_idx, [])
        if not head_indices:
            return {}

        results: Dict[int, List[int]] = {}
        if isinstance(query_pre_rope, dict):
            sample_q = next(iter(query_pre_rope.values()), None)
            if sample_q is None:
                return {}
            eff_device = sample_q.device if device is None else torch.device(device)
            eff_dtype = sample_q.dtype if dtype is None else dtype
            use_stream = (
                stream if stream is not None else self._get_layer_stream(eff_device)
            )
            for h_idx in head_indices:
                q_h = query_pre_rope.get(h_idx)
                if q_h is None:
                    continue
                results[h_idx] = self._heads[(l_idx, h_idx)].prefetch_async(
                    query_pre_rope=q_h,
                    device=eff_device,
                    dtype=eff_dtype,
                    top_k_pages=top_k_pages,
                    stream=use_stream,
                )
            return results

        eff_device = query_pre_rope.device if device is None else torch.device(device)
        eff_dtype = query_pre_rope.dtype if dtype is None else dtype
        use_stream = (
            stream if stream is not None else self._get_layer_stream(eff_device)
        )

        is_4d = query_pre_rope.ndim == 4
        if is_4d:
            bsz, n_qh, q_len, hdim = query_pre_rope.shape
            if n_qh % self.num_kv_heads == 0:
                q_per_kv = n_qh // self.num_kv_heads
                q_grouped = query_pre_rope.view(
                    bsz, self.num_kv_heads, q_per_kv * q_len, hdim
                )
            else:
                q_grouped = None
        else:
            q_grouped = None

        for h_idx in head_indices:
            if q_grouped is not None:
                q_h = q_grouped[:, h_idx, :, :]
            elif is_4d and h_idx < int(query_pre_rope.shape[1]):
                q_h = query_pre_rope[:, h_idx, :, :]
            else:
                q_h = query_pre_rope
            results[h_idx] = self._heads[(l_idx, h_idx)].prefetch_async(
                query_pre_rope=q_h,
                device=eff_device,
                dtype=eff_dtype,
                top_k_pages=top_k_pages,
                stream=use_stream,
            )
        return results

    def crop(self, max_pos: int) -> None:
        for head in self._heads.values():
            head.crop(max_pos)

    def clear_lru(self) -> None:
        for head in self._heads.values():
            head.lru_cache.clear()

    def telemetry(self) -> Dict[str, Any]:
        heads = list(self._heads.values())
        total_tokens = sum(h.num_tokens for h in heads)
        total_pages = sum(h.num_pages for h in heads)
        gpu_index_bytes = sum(h.gpu_index_bytes() for h in heads)
        gpu_lru_bytes = sum(h.gpu_lru_bytes() for h in heads)
        gpu_total_bytes = gpu_index_bytes + gpu_lru_bytes
        cpu_leaf_bytes = sum(h.cpu_leaf_bytes() for h in heads)
        lru_hits = sum(h.lru_cache.hits for h in heads)
        lru_misses = sum(h.lru_cache.misses for h in heads)
        accesses = lru_hits + lru_misses
        lru_hit_rate = float(lru_hits) / float(accesses) if accesses > 0 else 0.0
        pcie_bytes = sum(h.lru_cache.pcie_bytes_transferred for h in heads)
        sublinear_gpu_ratio = float(gpu_total_bytes) / float(max(1, cpu_leaf_bytes))

        return {
            "num_offloaded_heads": len(heads),
            "total_offloaded_tokens": total_tokens,
            "total_pages": total_pages,
            "gpu_index_bytes": gpu_index_bytes,
            "gpu_lru_bytes": gpu_lru_bytes,
            "gpu_total_bytes": gpu_total_bytes,
            "cpu_leaf_bytes": cpu_leaf_bytes,
            "lru_hits": lru_hits,
            "lru_misses": lru_misses,
            "lru_hit_rate": lru_hit_rate,
            "pcie_bytes_transferred": pcie_bytes,
            "draft_mode": bool(self._draft_mode),
            "sublinear_gpu_ratio": sublinear_gpu_ratio,
        }

    def save(self, path: Union[str, os.PathLike]) -> None:
        payload = {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "head_roles": self.head_roles.detach().cpu(),
            "config": asdict(self.config),
            "draft_mode": self._draft_mode,
            "heads": {
                key: head.state_dict() for key, head in self._heads.items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: Union[str, os.PathLike],
        U_Q: Optional[torch.Tensor] = None,
    ) -> "HiKVPStore":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = dict(payload["config"])
        if "offload_roles" in cfg_dict:
            cfg_dict["offload_roles"] = tuple(cfg_dict["offload_roles"])
        config = HiKVPConfig(**cfg_dict)
        store = cls(
            num_layers=int(payload["num_layers"]),
            num_kv_heads=int(payload["num_kv_heads"]),
            head_dim=int(payload["head_dim"]),
            head_roles=payload["head_roles"],
            config=config,
            U_Q=U_Q,
        )
        store._draft_mode = bool(payload.get("draft_mode", False))
        for key, h_state in payload.get("heads", {}).items():
            l_idx, h_idx = int(key[0]), int(key[1])
            u_q_lh = U_Q[l_idx, h_idx] if U_Q is not None else None
            store._heads[(l_idx, h_idx)] = HierarchicalKVHeadIndex.from_state_dict(
                h_state, u_q=u_q_lh
            )
            if h_idx not in store._layer_heads[l_idx]:
                store._layer_heads[l_idx].append(h_idx)
        for l_idx in store._layer_heads:
            store._layer_heads[l_idx].sort()
        return store


__all__ = [
    "HiKVPConfig",
    "HiKVPPage",
    "GPULRUPageCache",
    "HierarchicalKVHeadIndex",
    "HiKVPStore",
]
