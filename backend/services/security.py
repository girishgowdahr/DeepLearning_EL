"""
ForestGuard AI — remote-access protection for the phone-camera tunnel.

Threat model
------------
A Cloudflare Quick Tunnel (or a LAN IP over self-signed HTTPS) makes the whole
app reachable by anyone who learns the URL. Only the *laptop owner* should see
the dashboard, history, snapshots, settings and API docs. The phone needs just:

  * the static /camera page (+ its JS/CSS assets), and
  * the /ws/stream WebSocket, authenticated with a random access token.

A request is "local/privileged" only when BOTH hold:
  1. the TCP peer is loopback, and
  2. it carries no proxy-forwarding headers.
cloudflared connects to the origin from 127.0.0.1, so rule 1 alone is NOT enough;
rule 2 (Cf-Connecting-Ip / X-Forwarded-For / ...) is what identifies tunnel traffic.
Anything else is "remote" and is restricted to the allow-list below.
"""
import os
import secrets
import logging
from pathlib import Path
from backend.config import settings

logger = logging.getLogger("forestguard.security")

# TCP peers that count as the laptop itself. "testclient" is Starlette's TestClient
# placeholder (an in-process pseudo peer that can never come from the network).
LOCAL_PEERS = {"127.0.0.1", "::1", "localhost", "testclient"}

# Headers that only appear when a reverse proxy / tunnel sits in front of the app.
FORWARDING_HEADERS = (
    b"cf-connecting-ip",
    b"cf-ray",
    b"cf-visitor",
    b"x-forwarded-for",
    b"x-forwarded-host",
    b"x-real-ip",
    b"forwarded",
)

TOKEN_FILE: Path = settings.BASE_DIR / "run" / "phone_token.txt"
_token_cache: str | None = None


def get_phone_token() -> str:
    """
    Resolve the phone access token (never logged):
      1. FG_PHONE_TOKEN environment variable (set by scripts/run_demo.ps1)
      2. run/phone_token.txt (git-ignored)
      3. freshly generated and written to run/phone_token.txt
    """
    global _token_cache
    if _token_cache:
        return _token_cache

    env_tok = os.getenv("FG_PHONE_TOKEN", "").strip()
    if len(env_tok) >= 16:
        _token_cache = env_tok
        return _token_cache

    try:
        if TOKEN_FILE.exists():
            file_tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
            if len(file_tok) >= 16:
                _token_cache = file_tok
                return _token_cache
    except Exception:
        pass

    _token_cache = secrets.token_urlsafe(24)  # 192 bits
    try:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(_token_cache, encoding="utf-8")
    except Exception as e:  # token still works in memory
        logger.warning(f"Could not persist phone token file: {e}")
    return _token_cache


def reset_token_cache_for_tests() -> None:
    global _token_cache
    _token_cache = None


def verify_phone_token(candidate: str | None) -> bool:
    if not candidate or not isinstance(candidate, str):
        return False
    return secrets.compare_digest(candidate.encode("utf-8"), get_phone_token().encode("utf-8"))


def _header_names(scope) -> set[bytes]:
    return {k.lower() for k, _ in scope.get("headers", [])}


def has_forwarding_headers(scope) -> bool:
    names = _header_names(scope)
    return any(h in names for h in FORWARDING_HEADERS)


def is_remote(scope) -> bool:
    """True for tunnel / LAN / internet requests; False only for the laptop itself."""
    client = scope.get("client")
    peer = client[0] if client else None
    if peer not in LOCAL_PEERS:
        return True
    return has_forwarding_headers(scope)


# --- allow-list for remote (phone) visitors -----------------------------------------
REMOTE_HTTP_EXACT = {"/camera", "/camera/", "/favicon.ico", "/api/phone/config"}
REMOTE_HTTP_PREFIX = ("/assets/",)
REMOTE_WS_EXACT = {"/ws/stream"}


def remote_http_allowed(method: str, path: str) -> bool:
    if method not in ("GET", "HEAD", "OPTIONS"):
        return False
    if ".." in path:
        return False
    return path in REMOTE_HTTP_EXACT or path.startswith(REMOTE_HTTP_PREFIX)


def remote_ws_allowed(path: str) -> bool:
    return path in REMOTE_WS_EXACT


class RemoteAccessGuard:
    """
    Pure ASGI middleware (covers both HTTP and WebSocket scopes).
    Local requests pass through untouched, so every existing dashboard feature
    keeps working on http://localhost:8000.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        stype = scope["type"]
        if stype not in ("http", "websocket") or not is_remote(scope):
            return await self.app(scope, receive, send)

        path = scope.get("path", "")
        if stype == "http":
            if remote_http_allowed(scope.get("method", "GET"), path):
                return await self.app(scope, receive, send)
            body = b'{"detail":"Forbidden: this resource is only available on the laptop."}'
            await send({
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                            (b"cache-control", b"no-store")],
            })
            await send({"type": "http.response.body", "body": body})
            return

        # websocket
        if remote_ws_allowed(path):
            return await self.app(scope, receive, send)
        await receive()  # consume websocket.connect
        await send({"type": "websocket.close", "code": 4403})
