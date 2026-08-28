"""Request-edge protection for the FortyCool service.

Four endpoints spend real money on someone else's account: a run, a background
run, site discovery, and the copilot. None of them had authentication, a rate
limit, or a body-size ceiling, so a single anonymous host could bill the team's
OpenAI and FortyGuard keys without an account.

The controls here are deliberately layered so the demo keeps working:

* the **body-size ceiling** and the **rate limits** are always on, because they
  protect the process itself and cost a legitimate caller nothing;
* the **API key** is enforced only when ``FORTYCOOL_API_KEY`` is set, so a
  keyless local checkout still runs the full flow while a deployed instance can
  close the doors with one environment variable.
"""

from __future__ import annotations

import hmac
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

API_KEY_HEADER = "X-API-Key"
API_KEY_ENV = "FORTYCOOL_API_KEY"

# Ceiling on any request body. The telemetry route used to read the whole body
# into memory and only then compare it with the 10 MB limit, so a 300 MB body
# allocated roughly twice its size before being rejected.
MAX_REQUEST_BYTES = int(os.getenv("FORTYCOOL_MAX_REQUEST_BYTES", 12 * 1024 * 1024))


def api_key_required() -> bool:
    return bool(os.getenv(API_KEY_ENV))


def verify_api_key(request: Request) -> None:
    """Reject a request when a key is configured and the caller has none.

    Compared with ``hmac.compare_digest`` so a wrong key cannot be found one
    character at a time.
    """

    expected = os.getenv(API_KEY_ENV)
    if not expected:
        return
    supplied = request.headers.get(API_KEY_HEADER, "")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"a valid {API_KEY_HEADER} header is required",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


@dataclass
class RateLimitRule:
    """A sliding-window allowance, named so the error can explain itself."""

    name: str
    limit: int
    window_seconds: float

    def describe(self) -> str:
        return f"{self.limit} requests per {int(self.window_seconds)}s"


@dataclass
class SlidingWindowRateLimiter:
    """Per-caller sliding window, tracked in memory.

    Adequate for a single-process service, which is what this deployment is;
    the store is capped so the limiter itself cannot be used to exhaust memory.
    """

    max_tracked_callers: int = 4096
    _hits: dict[tuple[str, str], deque[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def check(self, caller: str, rule: RateLimitRule) -> None:
        now = time.monotonic()
        key = (rule.name, caller)
        with self._lock:
            if key not in self._hits and len(self._hits) >= self.max_tracked_callers:
                self._evict_stale(now)
            window = self._hits.setdefault(key, deque())
            cutoff = now - rule.window_seconds
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= rule.limit:
                retry_after = max(1, int(window[0] + rule.window_seconds - now) + 1)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"rate limit exceeded for {rule.name} ({rule.describe()})",
                    headers={"Retry-After": str(retry_after)},
                )
            window.append(now)

    def _evict_stale(self, now: float) -> None:
        for key, window in list(self._hits.items()):
            if not window or now - window[-1] > 3600:
                self._hits.pop(key, None)
        if len(self._hits) >= self.max_tracked_callers:
            # Still full: drop the least recently used half rather than refuse
            # service to everyone.
            ordered = sorted(self._hits.items(), key=lambda item: item[1][-1])
            for key, _ in ordered[: len(ordered) // 2]:
                self._hits.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def caller_identity(request: Request) -> str:
    """Identify the caller by API key when present, otherwise by client host."""

    supplied = request.headers.get(API_KEY_HEADER)
    if supplied:
        return f"key:{supplied[:16]}"
    client = request.client
    return f"ip:{client.host}" if client else "ip:unknown"


rate_limiter = SlidingWindowRateLimiter()

# Expensive routes call out to paid APIs; cheap routes only touch local state.
EXPENSIVE_RULE = RateLimitRule(
    name="analysis",
    limit=int(os.getenv("FORTYCOOL_RATE_LIMIT_ANALYSIS", 20)),
    window_seconds=float(os.getenv("FORTYCOOL_RATE_WINDOW_SECONDS", 60)),
)
COPILOT_RULE = RateLimitRule(
    name="copilot",
    limit=int(os.getenv("FORTYCOOL_RATE_LIMIT_COPILOT", 10)),
    window_seconds=float(os.getenv("FORTYCOOL_RATE_WINDOW_SECONDS", 60)),
)
UPLOAD_RULE = RateLimitRule(
    name="upload",
    limit=int(os.getenv("FORTYCOOL_RATE_LIMIT_UPLOAD", 10)),
    window_seconds=float(os.getenv("FORTYCOOL_RATE_WINDOW_SECONDS", 60)),
)
READ_RULE = RateLimitRule(
    name="read",
    limit=int(os.getenv("FORTYCOOL_RATE_LIMIT_READ", 240)),
    window_seconds=float(os.getenv("FORTYCOOL_RATE_WINDOW_SECONDS", 60)),
)


def _guard(rule: RateLimitRule):
    def dependency(request: Request) -> None:
        verify_api_key(request)
        rate_limiter.check(caller_identity(request), rule)

    return dependency


guard_analysis = _guard(EXPENSIVE_RULE)
guard_copilot = _guard(COPILOT_RULE)
guard_upload = _guard(UPLOAD_RULE)
guard_read = _guard(READ_RULE)


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Refuse an oversized body before anything buffers it.

    Checking the size inside the handler is too late: by then the whole body is
    already resident, which is the amplification an attacker wants.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_REQUEST_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                if int(declared) > self.max_bytes:
                    return self._too_large()
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"detail": "invalid Content-Length header"},
                )
        elif request.headers.get("transfer-encoding", "").lower() == "chunked":
            # No declared length: count as it streams and abort past the cap.
            total = 0
            chunks: list[bytes] = []
            async for chunk in request.stream():
                total += len(chunk)
                if total > self.max_bytes:
                    return self._too_large()
                chunks.append(chunk)
            body = b"".join(chunks)

            async def receive() -> dict:
                return {"type": "http.request", "body": body, "more_body": False}

            request._receive = receive  # type: ignore[attr-defined]
        return await call_next(request)

    def _too_large(self) -> JSONResponse:
        return JSONResponse(
            status_code=413,
            content={
                "detail": (
                    f"request body exceeds the {self.max_bytes // (1024 * 1024)} MB limit"
                )
            },
        )
