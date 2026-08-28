from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


class FortyGuardError(RuntimeError):
    pass


# States FortyGuard may report. Anything outside this set is treated as a
# failure rather than polled until the attempt budget runs out, which used to
# burn the full three minutes per activity on an unexpected spelling.
_TERMINAL_SUCCESS_STATES = {"completed", "complete", "success", "succeeded", "done"}
_TERMINAL_FAILURE_STATES = {"failed", "failure", "error", "cancelled", "canceled"}
_PENDING_STATES = {"pending", "queued", "running", "processing", "in_progress", "started"}

# Coordinates are snapped before they reach the cache key. Without this a shift
# of 1e-9 degrees - about a tenth of a millimetre - produced a different key and
# a fresh paid upstream call on every request.
_CACHE_COORDINATE_DECIMALS = 5


class FortyGuardClient:
    """Low-level, async FortyGuard client with bounded polling and response caching.

    High-level normalization remains separate because heatmap result properties
    should be validated against the team's real hackathon API responses first.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.fortyguard.com/v1",
        cache_dir: str | Path = ".fortycool-cache/fortyguard",
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 2.0,
        max_poll_attempts: int = 90,
        max_wait_seconds: float | None = None,
        max_attempts: int = 3,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("FORTYGUARD_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_attempts = max_poll_attempts
        # A wall-clock ceiling on one activity. Attempt counting alone allowed
        # 90 polls each carrying a 30 s timeout, so a single slow activity could
        # hold a request open for the better part of an hour.
        self.max_wait_seconds = (
            max_wait_seconds
            if max_wait_seconds is not None
            else float(os.getenv("FORTYGUARD_MAX_WAIT_SECONDS", 120.0))
        )
        self.max_attempts = max_attempts
        self.cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else float(os.getenv("FORTYGUARD_CACHE_TTL_SECONDS", 7 * 24 * 3600))
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @classmethod
    def _snap(cls, value: Any) -> Any:
        """Round coordinates so imperceptible jitter cannot bust the cache."""

        if isinstance(value, bool):
            return value
        if isinstance(value, float):
            return round(value, _CACHE_COORDINATE_DECIMALS)
        if isinstance(value, dict):
            return {key: cls._snap(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._snap(item) for item in value]
        return value

    @classmethod
    def _cache_key(cls, endpoint: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"endpoint": endpoint, "payload": cls._snap(payload)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        """Read a cached response, tolerating a truncated or stale file.

        The write was not atomic and the read had no error handling, so a
        process killed mid-write left a partial file that raised on every later
        request for that payload - permanently, until someone deleted it by hand.
        """

        path = self.cache_dir / f"{key}.json"
        try:
            raw = path.read_text()
        except OSError:
            return None
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            return None
        if not isinstance(entry, dict) or "cached_at" not in entry:
            # A file written before the envelope existed. Treat it as stale
            # rather than trusting an unbounded-age response.
            path.unlink(missing_ok=True)
            return None
        if time.time() - float(entry["cached_at"]) > self.cache_ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        response = entry.get("response")
        return response if isinstance(response, dict) else None

    def _store_cache(self, key: str, response: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        temporary = path.with_suffix(f".json.{os.getpid()}.tmp")
        envelope = {"cached_at": time.time(), "response": response}
        temporary.write_text(json.dumps(envelope, indent=2, sort_keys=True))
        # Atomic within a filesystem: a concurrent reader sees either the old
        # file or the complete new one, never a half-written one.
        os.replace(temporary, path)

    @staticmethod
    def _error_detail(body: Any) -> str:
        """Summarise an upstream error without echoing the whole body.

        `body.get('message', body)` embedded the entire upstream JSON in an
        exception that reaches the client.
        """

        if isinstance(body, dict):
            message = body.get("message") or body.get("error")
            if isinstance(message, str):
                return message[:200]
        return "upstream request failed"

    async def _request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        """One HTTP round trip with retries, mapped onto FortyGuardError.

        Raw httpx exceptions used to escape this client and surface as a 500;
        transient failures had no retry at all.
        """

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                    response = await client.request(method, url, **kwargs)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt + 1 >= self.max_attempts:
                    break
                await asyncio.sleep(min(2**attempt, 8))
                continue
            try:
                body = response.json()
            except ValueError as exc:
                raise FortyGuardError(
                    f"FortyGuard returned non-JSON status {response.status_code}"
                ) from exc
            if response.status_code >= 500 and attempt + 1 < self.max_attempts:
                await asyncio.sleep(min(2**attempt, 8))
                continue
            if response.is_error or (isinstance(body, dict) and body.get("error")):
                raise FortyGuardError(
                    f"FortyGuard request failed ({response.status_code}): "
                    f"{self._error_detail(body)}"
                )
            if not isinstance(body, dict):
                raise FortyGuardError("FortyGuard returned a non-object response")
            return body
        raise FortyGuardError(
            f"FortyGuard was unreachable after {self.max_attempts} attempts"
        ) from last_error

    async def submit(self, endpoint: str, payload: dict[str, Any]) -> str:
        if not self.api_key:
            raise FortyGuardError("FORTYGUARD_API_KEY is not configured")
        body = await self._request(
            "POST",
            f"{self.base_url}/{endpoint.lstrip('/')}",
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )
        data = body.get("data")
        activity_id = data.get("activity_id") if isinstance(data, dict) else None
        if not activity_id:
            raise FortyGuardError(
                "FortyGuard response did not contain data.activity_id"
            )
        return str(activity_id)

    async def status(self, activity_id: str) -> dict[str, Any]:
        if not self.api_key:
            raise FortyGuardError("FORTYGUARD_API_KEY is not configured")
        return await self._request(
            "GET",
            f"{self.base_url}/status/{activity_id}",
            headers={"api-key": self.api_key},
        )

    async def wait(self, activity_id: str) -> dict[str, Any]:
        started = time.monotonic()
        for _ in range(self.max_poll_attempts):
            body = await self.status(activity_id)
            data = body.get("data")
            state = str((data or {}).get("status", "")).strip().lower()
            if state in _TERMINAL_SUCCESS_STATES:
                return body
            if state in _TERMINAL_FAILURE_STATES:
                raise FortyGuardError(
                    f"FortyGuard activity {activity_id} reported {state}"
                )
            if state and state not in _PENDING_STATES:
                # An unrecognised terminal state is a failure. Polling it until
                # the budget expires wastes the whole window and still errors.
                raise FortyGuardError(
                    f"FortyGuard activity {activity_id} returned unknown state "
                    f"'{state}'"
                )
            if time.monotonic() - started > self.max_wait_seconds:
                break
            await asyncio.sleep(self.poll_interval_seconds)
        raise FortyGuardError(
            f"FortyGuard activity {activity_id} exceeded the "
            f"{self.max_wait_seconds:.0f}s wait budget"
        )

    async def run(
        self, endpoint: str, payload: dict[str, Any], *, use_cache: bool = True
    ) -> dict[str, Any]:
        key = self._cache_key(endpoint, payload)
        if use_cache and (cached := self._load_cache(key)) is not None:
            return cached
        activity_id = await self.submit(endpoint, payload)
        result = await self.wait(activity_id)
        self._store_cache(key, result)
        return result

    async def create_heatmap(
        self, payload: dict[str, Any], *, use_cache: bool = True
    ) -> dict[str, Any]:
        return await self.run("heatmap", payload, use_cache=use_cache)

    async def environmental_parameters(
        self, payload: dict[str, Any], *, use_cache: bool = True
    ) -> dict[str, Any]:
        return await self.run("env_params", payload, use_cache=use_cache)

    async def satellite_segmentation(
        self, payload: dict[str, Any], *, use_cache: bool = True
    ) -> dict[str, Any]:
        """Submit and retrieve a FortyGuard Satellite Segmentation activity."""

        return await self.run("satellite", payload, use_cache=use_cache)
