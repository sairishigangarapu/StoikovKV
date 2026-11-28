import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "float16"
    device: str = "cuda"
    attn_implementation: str = "sdpa"
    max_seq_len: int = 32768
    trust_remote_code: bool = False
    fallback_to_mock: bool = True


class CacheConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    n_b: int = 64
    keep_ratio: float = 0.15
    s_budget: int = 16
    policy_mode: str = "hybrid"
    enable_hmm_decay: bool = True
    shift_z_threshold: float = 1.4
    beta_stationary: float = 0.99
    beta_shift: float = 0.65
    ema_alpha: float = 0.15
    max_rollback_window: int = 32
    sink_tokens: int = 4
    sink_bonus: float = 100.0
    lambda1: float = 0.04


class ServerConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    request_timeout: int = 300
    max_sessions: int = 32
    max_request_bytes: int = 1_048_576
    api_key: Optional[str] = os.environ.get("STOIKOV_API_KEY", "").strip() or None


def load_yaml_config(path: Path, cls):
    if not path.is_file() or path.is_symlink():
        return cls()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return cls()
    key = cls.__name__.replace("Config", "").lower()
    sub = data.get(key) or {}
    return cls(**(sub if isinstance(sub, dict) else {}))


model_config = load_yaml_config(CONFIG_DIR / "model.yaml", ModelConfig)
cache_config = load_yaml_config(CONFIG_DIR / "cache.yaml", CacheConfig)
server_config = load_yaml_config(CONFIG_DIR / "server.yaml", ServerConfig)
