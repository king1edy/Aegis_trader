"""
Rate Limiting Middleware
========================
Redis-backed sliding window rate limiter for FastAPI.

Enforces per-user limits based on subscription tier.  Falls back to
no-op if Redis is unavailable (graceful degradation).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from core.config import get_settings

logger = logging.getLogger("core.rate_limiter")

# Paths that skip rate limiting entirely
_EXEMPT_PATHS = frozenset({"/health", "/", "/docs", "/openapi.json", "/redoc"})
# Paths that get an IP-based global limit (no auth required)
_PUBLIC_PATHS = frozenset({"/api/auth/login", "/api/auth/register"})

# Default limits for unauthenticated / IP-based requests
_IP_REQUESTS_PER_MINUTE = 30


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-request rate limiter.

    For authenticated users: enforces tier-based limits from the
    ``rate_limits`` table (cached in memory).

    For unauthenticated public endpoints: IP-based 30/min.

    Requires Redis.  If Redis is down, requests pass through
    without rate limiting.
    """

    def __init__(self, app) -> None:
        super().__init__(app)
        self._redis = None
        self._redis_checked = False
        self._limits_cache: dict[str, dict] = {}
        self._cache_loaded_at: float = 0
        self._cache_ttl: float = 300  # 5 minutes

    async def _get_redis(self):
        """Lazy-connect to Redis.  Returns None if unavailable."""
        if self._redis_checked and self._redis is None:
            return None
        if self._redis is not None:
            return self._redis

        self._redis_checked = True
        try:
            import redis.asyncio as aioredis

            settings = get_settings()
            self._redis = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            await self._redis.ping()
            logger.info("Rate limiter connected to Redis")
            return self._redis
        except Exception as exc:
            logger.warning(
                "Redis unavailable — rate limiting disabled",
                extra={"error": str(exc)},
            )
            self._redis = None
            return None

    async def _load_limits_cache(self) -> None:
        """Refresh the in-memory tier→limits cache from the DB."""
        now = time.monotonic()
        if (now - self._cache_loaded_at) < self._cache_ttl and self._limits_cache:
            return

        try:
            from sqlalchemy import select as sa_select

            from auth.subscription_models import RateLimits
            from database.repository import get_session

            async with get_session() as session:
                result = await session.execute(sa_select(RateLimits))
                rows = result.scalars().all()
                self._limits_cache = {
                    row.tier: {
                        "api_per_min": row.api_requests_per_minute,
                        "api_per_day": row.api_requests_per_day,
                        "webhook_per_min": row.webhook_events_per_minute,
                    }
                    for row in rows
                }
            self._cache_loaded_at = now
        except Exception as exc:
            logger.warning("Failed to load rate limits from DB", extra={"error": str(exc)})

    async def _get_user_tier(self, request: Request) -> tuple[str | None, str | None]:
        """
        Extract user_id and tier from the request.

        Returns (user_id, tier) or (None, None) if unauthenticated.
        """
        # Try JWT bearer token
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from auth.security import decode_access_token

                payload = decode_access_token(auth_header[7:])
                user_id = payload.get("sub")
                if user_id:
                    tier = await self._lookup_tier(user_id)
                    return user_id, tier
            except Exception:
                pass

        # Try API key
        api_key = request.headers.get("x-api-key", "")
        if api_key:
            try:
                from auth.security import hash_api_key

                from sqlalchemy import select as sa_select

                from auth.models import ApiKey
                from auth.subscription_models import Subscription
                from database.repository import get_session

                key_hash = hash_api_key(api_key)
                async with get_session() as session:
                    result = await session.execute(
                        sa_select(ApiKey.user_id).where(
                            ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True)
                        )
                    )
                    row = result.one_or_none()
                    if row:
                        user_id = str(row[0])
                        tier = await self._lookup_tier(user_id)
                        return user_id, tier
            except Exception:
                pass

        return None, None

    async def _lookup_tier(self, user_id: str) -> str:
        """Look up user's subscription tier.  Defaults to 'journal'."""
        try:
            from sqlalchemy import select as sa_select

            from auth.subscription_models import Subscription
            from database.repository import get_session

            async with get_session() as session:
                result = await session.execute(
                    sa_select(Subscription.tier).where(
                        Subscription.user_id == user_id
                    )
                )
                row = result.one_or_none()
                return row[0] if row else "journal"
        except Exception:
            return "journal"

    async def _check_limit(
        self, r, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int]:
        """
        Sliding window counter in Redis.

        Returns (allowed: bool, remaining: int).
        """
        now = time.time()
        window_start = now - window_seconds

        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds + 1)
        results = await pipe.execute()

        count = results[2]
        remaining = max(0, limit - count)
        allowed = count <= limit
        return allowed, remaining

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip exempt paths
        if path in _EXEMPT_PATHS:
            return await call_next(request)

        r = await self._get_redis()
        if r is None:
            # No Redis — pass through
            return await call_next(request)

        await self._load_limits_cache()

        # Identify the requester
        user_id, tier = await self._get_user_tier(request)

        if user_id is None:
            # Unauthenticated — IP-based limit for public paths
            if path in _PUBLIC_PATHS:
                client_ip = request.client.host if request.client else "unknown"
                key = f"rl:ip:{client_ip}:min"
                allowed, remaining = await self._check_limit(
                    r, key, _IP_REQUESTS_PER_MINUTE, 60
                )
                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many requests"},
                        headers={"Retry-After": "60"},
                    )
            # Non-public unauthenticated paths will hit 401 at the route level
            return await call_next(request)

        # Authenticated — tier-based limits
        limits = self._limits_cache.get(tier or "journal", {})
        per_min = limits.get("api_per_min", 30)
        per_day = limits.get("api_per_day", 5000)

        # Per-minute check
        min_key = f"rl:{user_id}:min"
        allowed, remaining = await self._check_limit(r, min_key, per_min, 60)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded (per minute)"},
                headers={"Retry-After": "60"},
            )

        # Per-day check
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        day_key = f"rl:{user_id}:day:{today}"
        allowed, remaining = await self._check_limit(r, day_key, per_day, 86400)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded (daily)"},
                headers={"Retry-After": "3600"},
            )

        # Webhook-specific limit for POST /trade
        if path == "/trade" and request.method == "POST":
            webhook_per_min = limits.get("webhook_per_min", 60)
            wh_key = f"rl:{user_id}:webhook:min"
            allowed, remaining = await self._check_limit(r, wh_key, webhook_per_min, 60)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Webhook rate limit exceeded"},
                    headers={"Retry-After": "60"},
                )

        response = await call_next(request)
        return response
