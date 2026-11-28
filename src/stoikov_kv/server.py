"""
OpenAI-Compatible Local Inference Server for StoikovKV (`stoikov_kv/server.py`).

Exposes `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/telemetry`, and `/health`
backed by `StoikovKVCache` (with optional `matryoshka_speculative_generate`) so local LLM harnesses
(OpenWebUI, Continue.dev, Cursor, `lm-eval-harness`) can connect over HTTP without extra dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from ._security import (
    is_trusted_host_header,
    require_bounded_int,
    require_finite_float,
    sanitize_chat_content,
    verify_api_key,
)
from .cache import StoikovKVCache, load_or_create_artifacts
from .calibration import load_model_and_tokenizer
from .matryoshka_spec import matryoshka_speculative_generate

logger = logging.getLogger(__name__)

MAX_REQUEST_BODY_BYTES: int = 1_048_576  # 1 MiB
MAX_CHAT_MESSAGES: int = 256
MAX_PROMPT_CHARS: int = 131_072
MAX_ALLOWED_NEW_TOKENS: int = 4096
_ALLOWED_POLICIES = frozenset({"hybrid", "full_fp16", "standalone_h2o", "standalone_lexico"})
_ALLOWED_CHAT_ROLES = frozenset({"system", "user", "assistant", "tool"})


def _format_chat_messages(messages: Any, tokenizer: Any, max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Validates and formats OpenAI-style `messages` into a single prompt string."""
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("messages must be a non-empty list of role/content objects.")
    if len(messages) > MAX_CHAT_MESSAGES:
        raise ValueError(f"messages exceeds maximum turn count ({MAX_CHAT_MESSAGES}).")

    normalized: List[Dict[str, str]] = []
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("Each item in messages must be a JSON object.")
        role = str(msg.get("role", "user")).strip().lower()
        if role not in _ALLOWED_CHAT_ROLES:
            raise ValueError(f"Unsupported message role: {role!r}.")
        raw_content = msg.get("content", "")
        if not isinstance(raw_content, str):
            raise ValueError("Message content must be a string.")
        total_chars += len(raw_content)
        if total_chars > max_chars:
            raise ValueError(f"Total chat content exceeds {max_chars} characters.")
        normalized.append({"role": role, "content": raw_content})

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            clean_for_tpl = [
                {"role": m["role"], "content": m["content"].replace("\x00", "")}
                for m in normalized
            ]
            return str(
                tokenizer.apply_chat_template(
                    clean_for_tpl, tokenize=False, add_generation_prompt=True
                )
            )
        except Exception:
            pass

    parts: List[str] = []
    for m in normalized:
        safe_body = sanitize_chat_content(m["content"], max_chars=max_chars).strip()
        parts.append(f"{m['role']}: {safe_body}")
    parts.append("assistant:")
    return "\n".join(parts)


class StoikovInferenceEngine:
    """
    Stateful local inference engine wrapping a CausalLM, tokenizer, and calibrated
    `StoikovKVCache` artifacts for OpenAI-compatible serving and CLI chat.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        artifacts: Dict[str, Any],
        model_id: str = "stoikov-qwen2.5",
        default_policy_mode: str = "hybrid",
        n_b: int = 64,
        keep_ratio: float = 0.25,
        s_budget: int = 16,
        draft_atoms: int = 4,
        gamma_spec: int = 4,
        default_speculative: bool = True,
        api_key: Optional[str] = None,
        max_new_tokens_limit: int = MAX_ALLOWED_NEW_TOKENS,
        max_prompt_chars: int = MAX_PROMPT_CHARS,
    ) -> None:
        if str(default_policy_mode) not in _ALLOWED_POLICIES:
            raise ValueError(f"Unsupported default_policy_mode: {default_policy_mode!r}")
        self.model = model
        self.tokenizer = tokenizer
        self.artifacts = artifacts
        self.model_id = str(model_id)
        self.default_policy_mode = str(default_policy_mode)
        self.n_b = require_bounded_int(int(n_b), "n_b", 1, 65536)
        self.keep_ratio = require_finite_float(keep_ratio, "keep_ratio", 0.01, 1.0)
        self.s_budget = require_bounded_int(int(s_budget), "s_budget", 1, 32767)
        self.draft_atoms = require_bounded_int(int(draft_atoms), "draft_atoms", 1, self.s_budget)
        self.gamma_spec = require_bounded_int(int(gamma_spec), "gamma_spec", 1, 64)
        self.default_speculative = bool(default_speculative)
        env_key = os.environ.get("STOIKOV_API_KEY", "").strip() or None
        self.api_key: Optional[str] = api_key if api_key is not None else env_key
        self.max_new_tokens_limit = require_bounded_int(
            int(max_new_tokens_limit), "max_new_tokens_limit", 1, 32768
        )
        self.max_prompt_chars = require_bounded_int(
            int(max_prompt_chars), "max_prompt_chars", 64, 1_048_576
        )
        self._gpu_lock = threading.Lock()
        self.last_telemetry: Dict[str, Any] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        artifacts_path: str = "artifacts/calibrated_artifacts.pt",
        device: Optional[str] = None,
        use_real_model: bool = False,
        load_in_4bit: bool = False,
        policy_mode: str = "hybrid",
        n_b: int = 64,
        keep_ratio: float = 0.25,
        s_budget: int = 16,
        draft_atoms: int = 4,
        gamma_spec: int = 4,
        speculative: bool = True,
        trust_remote_code: bool = False,
        api_key: Optional[str] = None,
    ) -> "StoikovInferenceEngine":
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            device=device,
            use_real_model=use_real_model,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
        artifacts = load_or_create_artifacts(
            model=model,
            artifacts_path=artifacts_path,
            tokenizer=tokenizer,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            artifacts=artifacts,
            model_id=model_name,
            default_policy_mode=policy_mode,
            n_b=n_b,
            keep_ratio=keep_ratio,
            s_budget=s_budget,
            draft_atoms=draft_atoms,
            gamma_spec=gamma_spec,
            default_speculative=speculative,
            api_key=api_key,
        )

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.6,
        top_k: int = 32,
        speculative: Optional[bool] = None,
        policy_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string.")
        if len(prompt) > self.max_prompt_chars:
            raise ValueError(f"prompt exceeds maximum length ({self.max_prompt_chars} chars).")
        clean_prompt = prompt.replace("\x00", "")

        max_new_tokens = require_bounded_int(
            max_new_tokens, "max_new_tokens", 1, self.max_new_tokens_limit
        )
        temperature = require_finite_float(temperature, "temperature", 0.0, 5.0)
        top_k = require_bounded_int(top_k, "top_k", 0, 250_000)

        if speculative is not None and not isinstance(speculative, bool):
            raise ValueError("speculative must be a boolean when provided.")
        use_spec = self.default_speculative if speculative is None else speculative

        mode = self.default_policy_mode if policy_mode is None else str(policy_mode)
        if mode not in _ALLOWED_POLICIES:
            raise ValueError(f"Invalid policy_mode {mode!r}; allowed: {sorted(_ALLOWED_POLICIES)}.")

        with self._gpu_lock:
            cache = StoikovKVCache(
                model=self.model,
                artifacts=self.artifacts,
                n_b=self.n_b,
                keep_ratio=self.keep_ratio,
                s_budget=self.s_budget,
                policy_mode=mode,
            )

            device = next(self.model.parameters()).device
            enc = self.tokenizer(clean_prompt, return_tensors="pt")
            input_ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(device)
            if input_ids.shape[-1] < 2:
                pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
                pad_fill = 2 - int(input_ids.shape[-1])
                input_ids = torch.cat(
                    [torch.full((1, pad_fill), pad_id, device=device, dtype=input_ids.dtype), input_ids],
                    dim=-1,
                )
            prompt_tokens = int(input_ids.shape[-1])

            t0 = time.perf_counter()
            if use_spec:
                spec_res = matryoshka_speculative_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=input_ids,
                    cache=cache,
                    gamma_spec=self.gamma_spec,
                    draft_atoms=min(self.draft_atoms, self.s_budget),
                    verify_atoms=self.s_budget,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.05),
                    top_k=top_k,
                )
                text = str(spec_res["generated_text"])
                completion_tokens = int(spec_res["num_generated_tokens"])
                acceptance_rate = float(spec_res["acceptance_rate"])
                tokens_per_sec = float(spec_res["tokens_per_sec"])
            else:
                out_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=max(temperature, 1e-5),
                    top_k=top_k,
                    past_key_values=cache,
                    use_cache=True,
                )
                elapsed = max(time.perf_counter() - t0, 1e-6)
                new_ids = out_ids[0, prompt_tokens:]
                completion_tokens = int(new_ids.shape[0])
                text = (
                    self.tokenizer.decode(new_ids, skip_special_tokens=True)
                    if hasattr(self.tokenizer, "decode")
                    else " ".join(str(int(t)) for t in new_ids)
                )
                acceptance_rate = 1.0
                tokens_per_sec = float(completion_tokens) / elapsed

            telemetry = cache.get_telemetry_summary()
            telemetry["speculative"] = use_spec
            telemetry["acceptance_rate"] = acceptance_rate
            telemetry["tokens_per_sec"] = tokens_per_sec
            self.last_telemetry = dict(telemetry)

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "telemetry": telemetry,
        }

    def _check_auth(self, headers: Optional[Mapping[str, str]]) -> bool:
        if not self.api_key:
            return True
        if headers is None:
            return False
        auth_val = headers.get("Authorization") or headers.get("authorization") or headers.get("X-API-Key") or headers.get("x-api-key")
        return verify_api_key(auth_val, self.api_key)

    def handle_request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Handles an HTTP request in-memory and returns `(status_code, json_payload)`.
        """
        m = method.upper().strip()
        p = path.split("?")[0].rstrip("/") or "/"

        if m == "GET" and p == "/health":
            return 200, {
                "status": "ok",
                "model": self.model_id,
                "default_policy_mode": self.default_policy_mode,
                "default_speculative": self.default_speculative,
            }

        if not self._check_auth(headers):
            return 401, {
                "error": {"message": "Unauthorized: valid Bearer API key required.", "type": "authentication_error"}
            }

        if body is not None and not isinstance(body, dict):
            return 400, {
                "error": {"message": "Request JSON body must be an object.", "type": "invalid_request_error"}
            }

        if m == "GET" and p == "/v1/models":
            return 200, {
                "object": "list",
                "data": [
                    {
                        "id": self.model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "stoikov-kv",
                    }
                ],
            }

        if m == "GET" and p == "/v1/telemetry":
            return 200, {"object": "stoikov.telemetry", "data": dict(self.last_telemetry)}

        try:
            if m == "POST" and p == "/v1/completions":
                req = body or {}
                raw_prompt = req.get("prompt", "Hello")
                if not isinstance(raw_prompt, str):
                    raise ValueError("prompt must be a string.")
                max_tokens = require_bounded_int(
                    req.get("max_tokens", 24), "max_tokens", 1, self.max_new_tokens_limit
                )
                temperature = require_finite_float(
                    req.get("temperature", 0.5), "temperature", 0.0, 5.0
                )
                top_k = require_bounded_int(req.get("top_k", 32), "top_k", 0, 250_000)
                speculative = req.get("speculative", None)
                policy_mode = req.get("policy_mode", None)

                gen = self.generate_text(
                    prompt=raw_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    speculative=speculative,
                    policy_mode=policy_mode,
                )
                return 200, {
                    "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_id,
                    "choices": [
                        {
                            "text": gen["text"],
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": gen["prompt_tokens"],
                        "completion_tokens": gen["completion_tokens"],
                        "total_tokens": gen["total_tokens"],
                    },
                    "stoikov_telemetry": gen["telemetry"],
                }

            if m == "POST" and p == "/v1/chat/completions":
                req = body or {}
                messages = req.get("messages", [{"role": "user", "content": "Hello"}])
                prompt = _format_chat_messages(
                    messages, self.tokenizer, max_chars=self.max_prompt_chars
                )
                max_tokens = require_bounded_int(
                    req.get("max_tokens", 24), "max_tokens", 1, self.max_new_tokens_limit
                )
                temperature = require_finite_float(
                    req.get("temperature", 0.5), "temperature", 0.0, 5.0
                )
                top_k = require_bounded_int(req.get("top_k", 32), "top_k", 0, 250_000)
                speculative = req.get("speculative", None)
                policy_mode = req.get("policy_mode", None)

                gen = self.generate_text(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    speculative=speculative,
                    policy_mode=policy_mode,
                )
                return 200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": gen["text"]},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": gen["prompt_tokens"],
                        "completion_tokens": gen["completion_tokens"],
                        "total_tokens": gen["total_tokens"],
                    },
                    "stoikov_telemetry": gen["telemetry"],
                }
        except (ValueError, TypeError) as exc:
            return 400, {"error": {"message": str(exc), "type": "invalid_request_error"}}
        except Exception:
            logger.exception("Unhandled inference failure in StoikovInferenceEngine")
            return 500, {"error": {"message": "Internal inference error.", "type": "server_error"}}

        return 404, {"error": {"message": "Unknown route.", "type": "invalid_request_error"}}


def create_http_server(
    engine: StoikovInferenceEngine,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_body_bytes: int = MAX_REQUEST_BODY_BYTES,
    max_concurrent_requests: int = 8,
    request_timeout_sec: float = 30.0,
) -> ThreadingHTTPServer:
    """Creates a hardened `ThreadingHTTPServer` bound to `(host, port)` serving `engine`."""
    port_num = require_bounded_int(int(port), "port", 0, 65535)
    body_cap = require_bounded_int(int(max_body_bytes), "max_body_bytes", 1024, 64 * 1024 * 1024)
    concurrency_sem = threading.BoundedSemaphore(max(1, int(max_concurrent_requests)))

    class _Handler(BaseHTTPRequestHandler):
        timeout = float(request_timeout_sec)

        def _check_host_and_origin(self, is_post: bool = False) -> bool:
            if not is_trusted_host_header(self.headers.get("Host"), host):
                self._send_json(400, {"error": {"message": "Untrusted Host header.", "type": "invalid_request_error"}})
                return False
            origin = self.headers.get("Origin")
            if is_post and origin:
                orig_l = origin.strip().lower()
                if not (orig_l.startswith("http://localhost:") or orig_l.startswith("http://127.0.0.1:") or orig_l in ("http://localhost", "http://127.0.0.1")):
                    self._send_json(403, {"error": {"message": "Cross-origin POST rejected.", "type": "permission_error"}})
                    return False
            return True

        def _write_security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self._write_security_headers()
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if len(self.path) > 4096:
                self._send_json(414, {"error": {"message": "URI too long.", "type": "invalid_request_error"}})
                return
            if not self._check_host_and_origin(is_post=False):
                return
            if not concurrency_sem.acquire(blocking=False):
                self._send_json(429, {"error": {"message": "Server busy; retry shortly.", "type": "rate_limit_error"}})
                return
            try:
                status, payload = engine.handle_request("GET", self.path, headers=self.headers)
                self._send_json(status, payload)
            finally:
                concurrency_sem.release()

        def do_POST(self) -> None:
            if len(self.path) > 4096:
                self._send_json(414, {"error": {"message": "URI too long.", "type": "invalid_request_error"}})
                return
            if not self._check_host_and_origin(is_post=True):
                return
            if not concurrency_sem.acquire(blocking=False):
                self._send_json(429, {"error": {"message": "Server busy; retry shortly.", "type": "rate_limit_error"}})
                return
            try:
                if self.headers.get("Transfer-Encoding") is not None:
                    self._send_json(400, {"error": {"message": "Chunked Transfer-Encoding is not supported.", "type": "invalid_request_error"}})
                    return
                raw_cl = self.headers.get("Content-Length", "0").strip()
                try:
                    length = int(raw_cl)
                except ValueError:
                    self._send_json(400, {"error": {"message": "Malformed Content-Length header.", "type": "invalid_request_error"}})
                    return
                if length < 0:
                    self._send_json(400, {"error": {"message": "Negative Content-Length header.", "type": "invalid_request_error"}})
                    return
                if length > body_cap:
                    self._send_json(413, {"error": {"message": f"Payload exceeds {body_cap} bytes.", "type": "invalid_request_error"}})
                    return

                try:
                    raw_body = self.rfile.read(length).decode("utf-8", errors="strict") if length > 0 else "{}"
                except UnicodeDecodeError:
                    self._send_json(400, {"error": {"message": "Request body must be valid UTF-8.", "type": "invalid_request_error"}})
                    return

                try:
                    body = json.loads(raw_body) if raw_body.strip() else {}
                except Exception:
                    self._send_json(400, {"error": {"message": "Invalid JSON body.", "type": "invalid_request_error"}})
                    return

                if not isinstance(body, dict):
                    self._send_json(400, {"error": {"message": "JSON body must be an object.", "type": "invalid_request_error"}})
                    return

                clean_route = self.path.split("?")[0].rstrip("/")
                if body.get("stream", False) and clean_route == "/v1/chat/completions":
                    status, payload = engine.handle_request("POST", self.path, body, headers=self.headers)
                    if status != 200:
                        self._send_json(status, payload)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self._write_security_headers()
                    self.end_headers()
                    chunk = {
                        "id": payload["id"],
                        "object": "chat.completion.chunk",
                        "created": payload["created"],
                        "model": payload["model"],
                        "choices": [
                            {
                                "index": 0,
                                "delta": payload["choices"][0]["message"],
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                    return

                status, payload = engine.handle_request("POST", self.path, body, headers=self.headers)
                self._send_json(status, payload)
            finally:
                concurrency_sem.release()

        def log_message(self, format: str, *args: Any) -> None:
            return

    srv = ThreadingHTTPServer((host, port_num), _Handler)
    srv.daemon_threads = True
    return srv
