"""
Auth Dependencies
=================
FastAPI dependency injection for route protection.

- ``get_current_user``        — JWT bearer token (dashboard)
- ``verify_api_key``          — X-API-Key header (EA webhook)
- ``get_tenant_id``           — tenant UUID from JWT user
- ``get_tenant_id_from_api_key`` — tenant UUID from API key owner
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from auth.security import decode_access_token, hash_api_key
from database.repository import get_session

# Lazy import to avoid circular dependency at module level
_User = None
_ApiKey = None


def _get_models():
    global _User, _ApiKey
    if _User is None:
        from auth.models import ApiKey, User
        _User = User
        _ApiKey = ApiKey
    return _User, _ApiKey


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login", auto_error=False
)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    """Validate JWT and return the User ORM object. Raises 401 on failure."""
    User, _ = _get_models()

    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )

    async with get_session() as session:
        result = await session.execute(
            select(User)
            .options(selectinload(User.subscription))
            .where(User.id == user_id, User.is_active.is_(True))
        )
        user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
):
    """Return the current user or None (no 401)."""
    if token is None:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Look up an API key by its SHA-256 hash. Return the owning User."""
    User, ApiKey = _get_models()

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    hashed = hash_api_key(api_key)

    async with get_session() as session:
        result = await session.execute(
            select(ApiKey)
            .where(ApiKey.key_hash == hashed, ApiKey.is_active.is_(True))
        )
        key_obj = result.scalar_one_or_none()

        if key_obj is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key",
            )

        # Check expiry
        if key_obj.expires_at and key_obj.expires_at < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key expired",
            )

        # Update last_used_at
        key_obj.last_used_at = datetime.now(timezone.utc)
        await session.commit()

        # Load the owning user
        user_result = await session.execute(
            select(User).where(User.id == key_obj.user_id, User.is_active.is_(True))
        )
        user = user_result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key owner not found or inactive",
        )
    return user


async def get_tenant_id(user=Depends(get_current_user)) -> UUID:
    """Extract tenant_id from JWT-authenticated user. tenant_id == user.id."""
    return user.id


async def get_tenant_id_from_api_key(user=Depends(verify_api_key)) -> UUID:
    """Extract tenant_id from API-key-authenticated user."""
    return user.id
