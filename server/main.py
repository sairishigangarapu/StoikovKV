from contextlib import asynccontextmanager
import hmac
import json
from typing import Dict, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from server.config import server_config
from server.routes import generate as generate_router
from server.services.cache_manager import cache_manager
from server.services.model_loader import create_cache_factory, load_model_and_tokenizer
from stoikov_kv._security import is_trusted_host_header


class _PayloadTooLargeError(Exception):
    pass


class SecurityGuardASGIMiddleware:
    """
    Pure ASGI boundary security middleware enforcing:
    1. Loopback DNS-rebinding `Host` header validation (`400`)
    2. Cross-origin state-changing `Origin` CSRF validation (`403`)
    3. Both `Content-Length` AND chunked-stream cumulative byte ceiling (`413`)
    4. Optional constant-time `STOIKOV_API_KEY` authentication (`401`)
    5. Hardened HTTP response headers (`nosniff`, `DENY`, `no-referrer`, `no-store`)
    """

    _SEC_HEADERS = (
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
        (b"cache-control", b"no-store"),
    )

    def __init__(self, app):
        self.app = app

    @staticmethod
    def _verify_api_key(headers: Dict[bytes, bytes]) -> bool:
        expected = server_config.api_key
        if not expected:
            return True
        raw_bytes = headers.get(b"authorization") or headers.get(b"x-api-key") or b""
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return False
        token = raw[7:].strip() if raw.lower().startswith("bearer ") else raw.strip()
        if not token:
            return False
        return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))

    async def _send_json_error(self, send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json; charset=utf-8"),
            (b"content-length", str(len(body)).encode("ascii")),
            *self._SEC_HEADERS,
        ]
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        hdrs: Dict[bytes, bytes] = {k.lower(): v for k, v in scope.get("headers", ())}
        method: str = scope.get("method", "GET").upper()
        path: str = (scope.get("path") or "/").rstrip("/") or "/"

        # 1. DNS rebinding check
        host_hdr = hdrs.get(b"host", b"").decode("latin-1")
        if not is_trusted_host_header(host_hdr, server_config.host):
            await self._send_json_error(send, 400, "Untrusted Host header.")
            return

        # 2. Cross-origin CSRF check on state-changing requests
        origin_hdr = hdrs.get(b"origin", b"").decode("latin-1").strip()
        if origin_hdr and method in ("POST", "DELETE"):
            allowed = set(server_config.cors_origins)
            if "*" not in allowed and origin_hdr.rstrip("/") not in {o.rstrip("/") for o in allowed}:
                await self._send_json_error(send, 403, "Cross-origin request rejected.")
                return

        # 3. Content-Length header check
        max_bytes = int(server_config.max_request_bytes)
        if b"content-length" in hdrs:
            try:
                cl_val = int(hdrs[b"content-length"].decode("ascii").strip())
            except ValueError:
                await self._send_json_error(send, 400, "Malformed Content-Length header.")
                return
            if cl_val < 0:
                await self._send_json_error(send, 400, "Negative Content-Length header.")
                return
            if cl_val > max_bytes:
                await self._send_json_error(send, 413, f"Request payload exceeds {max_bytes} bytes.")
                return

        # 4. Optional constant-time API key check
        if method != "OPTIONS" and path != "/api/v1/health":
            if not self._verify_api_key(hdrs):
                await self._send_json_error(send, 401, "Unauthorized.")
                return

        # 5. True streaming/chunked receive byte-limit enforcement + security header injection
        seen_bytes = 0
        exceeded_limit = False
        response_started = False

        async def _bounded_receive():
            nonlocal seen_bytes, exceeded_limit
            msg = await receive()
            if msg.get("type") == "http.request":
                seen_bytes += len(msg.get("body", b""))
                if seen_bytes > max_bytes:
                    exceeded_limit = True
                    raise _PayloadTooLargeError()
            return msg

        async def _hardened_send(msg):
            nonlocal response_started
            if exceeded_limit:
                return
            if msg.get("type") == "http.response.start":
                response_started = True
                existing = list(msg.get("headers", ()))
                present = {k.lower() for k, _ in existing}
                for sec_k, sec_v in self._SEC_HEADERS:
                    if sec_k not in present:
                        existing.append((sec_k, sec_v))
                msg = dict(msg, headers=existing)
            await send(msg)

        try:
            await self.app(scope, _bounded_receive, _hardened_send)
        except _PayloadTooLargeError:
            pass
        if exceeded_limit and not response_started:
            await self._send_json_error(send, 413, f"Streamed request body exceeds {max_bytes} bytes.")


@asynccontextmanager
async def _lifespan(application: FastAPI):
    print("[Server] Loading model and tokenizer...")
    model, tokenizer, artifacts = load_model_and_tokenizer()

    print("[Server] Creating cache factory...")
    cache_factory, _, _, _ = create_cache_factory()

    cache_manager.cache_factory = cache_factory
    cache_manager.max_sessions = max(1, int(server_config.max_sessions))

    application.state.model = model
    application.state.tokenizer = tokenizer
    application.state.artifacts = artifacts

    print(f"[Server] Ready on {server_config.host}:{server_config.port}")
    try:
        yield
    finally:
        print("[Server] Shutting down...")


app = FastAPI(
    title="StoikovKV Server",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=server_config.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)
app.add_middleware(SecurityGuardASGIMiddleware)

app.include_router(generate_router.router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=server_config.host,
        port=server_config.port,
        workers=server_config.workers,
        timeout_keep_alive=server_config.request_timeout,
    )
