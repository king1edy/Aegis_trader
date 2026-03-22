"""
Auth Router
===========
Registration, login (OAuth2 password flow), profile, and API key management.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select

from auth.models import ApiKey, User
from auth.security import (
    create_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)
from auth.dependencies import get_current_user
from database.repository import get_session

auth_router = APIRouter(prefix="/api/auth", tags=["Auth"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=128)


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    username: str
    access_token: str
    token_type: str = "bearer"


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserProfile(BaseModel):
    id: str
    email: str
    username: str
    is_admin: bool
    created_at: str


class CreateApiKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class ApiKeyCreated(BaseModel):
    id: str
    name: str
    key_prefix: str
    full_key: str  # shown once
    created_at: str


class ApiKeyInfo(BaseModel):
    id: str
    name: str
    key_prefix: str
    is_active: bool
    last_used_at: Optional[str]
    created_at: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@auth_router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(body: RegisterRequest):
    """Create a new user account and return a JWT."""
    async with get_session() as session:
        # Check uniqueness
        existing = await session.execute(
            select(User).where(
                (User.email == body.email) | (User.username == body.username)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email or username already registered",
            )

        user = User(
            email=body.email,
            username=body.username,
            hashed_password=hash_password(body.password),
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        token = create_access_token({"sub": str(user.id)})

        return RegisterResponse(
            user_id=str(user.id),
            email=user.email,
            username=user.username,
            access_token=token,
        )


@auth_router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password grant. Username field accepts email or username."""
    async with get_session() as session:
        result = await session.execute(
            select(User).where(
                (User.email == form_data.username)
                | (User.username == form_data.username)
            )
        )
        user = result.scalar_one_or_none()

        if user is None or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated",
            )

        token = create_access_token({"sub": str(user.id)})
        return TokenResponse(access_token=token)


@auth_router.get("/me", response_model=UserProfile)
async def get_me(user: User = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return UserProfile(
        id=str(user.id),
        email=user.email,
        username=user.username,
        is_admin=user.is_admin,
        created_at=user.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# API Key management
# ---------------------------------------------------------------------------

@auth_router.post("/api-keys", response_model=ApiKeyCreated, status_code=201)
async def create_api_key_endpoint(
    body: CreateApiKeyRequest,
    user: User = Depends(get_current_user),
):
    """Generate a new API key. The full key is returned once — save it."""
    full_key, prefix, key_hash = generate_api_key()

    async with get_session() as session:
        api_key = ApiKey(
            user_id=user.id,
            key_hash=key_hash,
            key_prefix=prefix,
            name=body.name,
        )
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

        return ApiKeyCreated(
            id=str(api_key.id),
            name=api_key.name,
            key_prefix=prefix,
            full_key=full_key,
            created_at=api_key.created_at.isoformat(),
        )


@auth_router.get("/api-keys", response_model=list[ApiKeyInfo])
async def list_api_keys(user: User = Depends(get_current_user)):
    """List the user's API keys (prefix + metadata, not the full key)."""
    async with get_session() as session:
        result = await session.execute(
            select(ApiKey)
            .where(ApiKey.user_id == user.id)
            .order_by(ApiKey.created_at.desc())
        )
        keys = result.scalars().all()

        return [
            ApiKeyInfo(
                id=str(k.id),
                name=k.name,
                key_prefix=k.key_prefix,
                is_active=k.is_active,
                last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
                created_at=k.created_at.isoformat(),
            )
            for k in keys
        ]


@auth_router.delete("/api-keys/{key_id}", status_code=204)
async def revoke_api_key(key_id: UUID, user: User = Depends(get_current_user)):
    """Deactivate an API key."""
    async with get_session() as session:
        result = await session.execute(
            select(ApiKey).where(
                ApiKey.id == key_id, ApiKey.user_id == user.id
            )
        )
        api_key = result.scalar_one_or_none()

        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

        api_key.is_active = False
        await session.commit()
