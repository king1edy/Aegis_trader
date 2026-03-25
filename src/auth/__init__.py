"""
Auth Module
===========
User authentication and API key management for multi-tenant isolation.

- JWT-based auth for the dashboard (browser)
- API key auth for the EA webhook (MT5)
"""

from auth.models import User, ApiKey
from auth.dependencies import (
    get_current_user,
    get_current_user_optional,
    get_tenant_id,
    get_tenant_id_from_api_key,
    verify_api_key,
)
from auth.security import create_access_token
from auth.router import auth_router

__all__ = [
    "User",
    "ApiKey",
    "get_current_user",
    "get_current_user_optional",
    "get_tenant_id",
    "get_tenant_id_from_api_key",
    "verify_api_key",
    "create_access_token",
    "auth_router",
]
