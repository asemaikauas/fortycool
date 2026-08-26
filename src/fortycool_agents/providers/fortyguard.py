from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx


class FortyGuardError(RuntimeError):
    pass


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
    ) -> None:
        self.api_key = api_key or os.getenv("FORTYGUARD_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.max_poll_attempts = max_poll_attempts

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _cache_key(endpoint: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            {"endpoint": endpoint, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _load_cache(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _store_cache(self, key: str, response: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(response, indent=2, sort_keys=True))

    async def submit(self, endpoint: str, payload: dict[str, Any]) -> str:
        if not self.api_key:
            raise FortyGuardError("FORTYGUARD_API_KEY is not configured")
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/{endpoint.lstrip('/')}",
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise FortyGuardError(
                f"FortyGuard returned non-JSON status {response.status_code}"
            ) from exc
        if response.is_error or body.get("error"):
            raise FortyGuardError(
                f"FortyGuard submission failed ({response.status_code}): {body.get('message', body)}"
            )
        activity_id = body.get("data", {}).get("activity_id")
        if not activity_id:
            raise FortyGuardError(
                "FortyGuard response did not contain data.activity_id"
            )
        return str(activity_id)

    async def status(self, activity_id: str) -> dict[str, Any]:
        if not self.api_key:
            raise FortyGuardError("FORTYGUARD_API_KEY is not configured")
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                f"{self.base_url}/status/{activity_id}",
                headers={"api-key": self.api_key},
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise FortyGuardError(
                f"FortyGuard returned non-JSON status {response.status_code}"
            ) from exc
        if response.is_error or body.get("error"):
            raise FortyGuardError(
                f"FortyGuard status failed ({response.status_code}): {body.get('message', body)}"
            )
        return body

    async def wait(self, activity_id: str) -> dict[str, Any]:
        for _ in range(self.max_poll_attempts):
            body = await self.status(activity_id)
            state = str(body.get("data", {}).get("status", "")).lower()
            if state == "completed":
                return body
            if state == "failed":
                raise FortyGuardError(f"FortyGuard activity {activity_id} failed")
            await asyncio.sleep(self.poll_interval_seconds)
        raise FortyGuardError(
            f"FortyGuard activity {activity_id} exceeded polling limit"
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
