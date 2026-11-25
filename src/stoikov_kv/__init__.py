"""StoikovKV: FlashAttention-Compatible Hybrid Evict-or-Encode KV Cache."""

from .calibration import (
    ROLE_RETRIEVAL_FP16,
    ROLE_PEAKED_H2O,
    ROLE_DIFFUSE_LEXICO,
    CALIBRATION_PROMPTS,
    NEEDLE_PROMPTS,
    HELDOUT_EVAL_PROMPTS,
    calibrate_model,
    score_write_time,
    score_tokens_write_time,
    reduce_attention_to_kv_heads,
    evaluate_write_time_probe,
    evaluate_compressed_cache_passkey,
    load_model_and_tokenizer,
)
from .elastic_lexico import (
    ElasticLexicoCodec,
    QuantizedInt8Values,
    compute_break_even_length,
)
from .cache import (
    DynamicCache,
    StoikovKVCache,
    OnlineRegimeChangeController,
    HMMTopicShiftController,
    stoikov_control_step,
    create_mock_artifacts,
    load_or_create_artifacts,
)
from .triton_kernels import (
    HAS_TRITON,
    can_use_triton,
    triton_lexico_reconstruct,
)
from .matryoshka_spec import (
    MatryoshkaSpecResult,
    matryoshka_speculative_generate,
)
from .server import (
    StoikovInferenceEngine,
    create_http_server,
)
from .hikvp import (
    HiKVPConfig,
    HiKVPPage,
    GPULRUPageCache,
    HierarchicalKVHeadIndex,
    HiKVPStore,
)
from .atlas_memory import (
    AtlasTriple,
    AtlasDocumentChunk,
    AtlasMemoryBank,
    format_kg_triples,
    apply_rope_to_atlas_keys,
    rectangular_attention,
)

__all__ = [
    "ROLE_RETRIEVAL_FP16",
    "ROLE_PEAKED_H2O",
    "ROLE_DIFFUSE_LEXICO",
    "CALIBRATION_PROMPTS",
    "NEEDLE_PROMPTS",
    "HELDOUT_EVAL_PROMPTS",
    "calibrate_model",
    "score_write_time",
    "score_tokens_write_time",
    "reduce_attention_to_kv_heads",
    "evaluate_write_time_probe",
    "evaluate_compressed_cache_passkey",
    "load_model_and_tokenizer",
    "ElasticLexicoCodec",
    "QuantizedInt8Values",
    "compute_break_even_length",
    "DynamicCache",
    "StoikovKVCache",
    "OnlineRegimeChangeController",
    "HMMTopicShiftController",
    "stoikov_control_step",
    "HAS_TRITON",
    "can_use_triton",
    "triton_lexico_reconstruct",
    "create_mock_artifacts",
    "load_or_create_artifacts",
    "MatryoshkaSpecResult",
    "matryoshka_speculative_generate",
    "StoikovInferenceEngine",
    "create_http_server",
    "HiKVPConfig",
    "HiKVPPage",
    "GPULRUPageCache",
    "HierarchicalKVHeadIndex",
    "HiKVPStore",
    "AtlasTriple",
    "AtlasDocumentChunk",
    "AtlasMemoryBank",
    "format_kg_triples",
    "apply_rope_to_atlas_keys",
    "rectangular_attention",
]
