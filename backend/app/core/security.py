"""
JWT authentication and security utilities.

Provides:
- JWT token creation / verification
- Password hashing (for future admin users)
- FastAPI dependency for protected endpoints
- HMAC webhook signature verification
"""

import hashlib
import hmac
import logging
import uuid as _uuid
from datetime import UTC, datetime, timedelta
from uuid import UUID

import bcrypt
from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.core.cache import cache_get, cache_set
from app.core.config import settings
from app.core.exceptions import AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


def _is_explicit_dev_mode() -> bool:
    """Allow bypasses only when explicitly in development with DEBUG enabled."""
    env = settings.ENVIRONMENT.lower()
    if env in ("production", "staging"):
        return False
    return env == "development" and settings.DEBUG


def hash_password(password: str) -> str:
    """Hash a plain-text password using bcrypt."""
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against a bcrypt hash."""
    pwd_bytes = plain_password.encode("utf-8")
    hash_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(pwd_bytes, hash_bytes)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create a signed JWT refresh token with unique jti for blacklisting."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh", "jti": str(_uuid.uuid4())})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_token(token: str) -> dict:
    """Decode and verify a JWT token."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc


async def blacklist_token(token: str, ttl: int | None = None) -> None:
    """Add a token to the Redis blacklist."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    if ttl is None or ttl <= 0:
        ttl = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400
    await cache_set(f"token_blacklist:{token_hash}", "1", ttl=ttl)


async def is_token_blacklisted(token: str) -> bool:
    """Check if a token has been revoked."""
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return await cache_get(f"token_blacklist:{token_hash}") is not None


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency that extracts and verifies the current user."""
    if token is None:
        if _is_explicit_dev_mode():
            logger.warning("DEV MODE: Unauthenticated request allowed as dev-admin")
            return {"sub": "dev-admin", "role": "admin"}
        raise AuthenticationError("Missing authentication token.")
    payload = verify_token(token)
    return payload


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency that requires admin role."""
    if current_user.get("role") != "admin":
        raise AuthorizationError("Admin access required.")
    return current_user


def is_admin_user(current_user: dict) -> bool:
    """Return True when the authenticated user is a platform admin."""
    return current_user.get("role") == "admin"


TENANT_ROLES = ("owner", "manager", "agent", "viewer")
ALL_ROLES = ("admin",) + TENANT_ROLES

_ROLE_RANK = {"viewer": 1, "agent": 2, "manager": 3, "owner": 4, "admin": 5}


def has_role_at_least(current_user: dict, minimum: str) -> bool:
    rank = _ROLE_RANK.get(current_user.get("role") or "", 0)
    return rank >= _ROLE_RANK.get(minimum, 99)


def require_role(*allowed: str):
    allowed_set = set(allowed) | {"admin"}

    async def _dep(current_user: dict = Depends(get_current_user)) -> dict:
        role = current_user.get("role")
        if role not in allowed_set:
            raise AuthorizationError(f"This action requires one of: {', '.join(sorted(allowed_set))}.")
        return current_user

    return _dep


def require_min_role(minimum: str):
    async def _dep(current_user: dict = Depends(get_current_user)) -> dict:
        if not has_role_at_least(current_user, minimum):
            raise AuthorizationError(f"This action requires {minimum} role or higher.")
        return current_user

    return _dep


def get_user_tenant_id(current_user: dict) -> UUID | None:
    """Return the authenticated user's tenant UUID, if present."""
    tenant_id = current_user.get("tenant_id")
    if not tenant_id:
        return None
    try:
        return UUID(str(tenant_id))
    except (TypeError, ValueError) as exc:
        raise AuthorizationError("Invalid tenant claim in authentication token.") from exc


def resolve_tenant_scope(current_user: dict, requested_tenant_id: UUID | None = None, *, require_explicit_admin_tenant: bool = False) -> UUID:
    """Resolve the tenant scope for the current request."""
    if is_admin_user(current_user):
        if requested_tenant_id is None and require_explicit_admin_tenant:
            raise AuthorizationError("Admin requests must specify a tenant_id.")
        if requested_tenant_id is None:
            raise AuthorizationError("Unable to resolve tenant scope for this request.")
        return requested_tenant_id

    token_tenant_id = get_user_tenant_id(current_user)
    if token_tenant_id is None:
        raise AuthorizationError("Tenant-scoped endpoint requires a tenant user.")

    if requested_tenant_id is not None and requested_tenant_id != token_tenant_id:
        raise AuthorizationError("You do not have access to the requested tenant.")

    return token_tenant_id


def ensure_tenant_access(current_user: dict, tenant_id: UUID) -> UUID:
    """Ensure the authenticated user may access *tenant_id*."""
    return resolve_tenant_scope(current_user, tenant_id, require_explicit_admin_tenant=True)


async def require_verified_email(current_user: dict = Depends(get_current_user)) -> dict:
    """Dependency that requires the user's email to be verified."""
    if current_user.get("role") == "admin":
        return current_user
    if not current_user.get("email_verified", False):
        raise AuthorizationError("Email verification required.")
    return current_user


async def verify_webhook_signature(request: Request) -> bytes:
    """FastAPI dependency that verifies the X-Hub-Signature-256 header."""
    body = await request.body()
    secrets_to_try: list[tuple[str, str]] = []
    if settings.META_APP_SECRET:
        secrets_to_try.append(("META_APP_SECRET", settings.META_APP_SECRET))
    if settings.INSTAGRAM_APP_SECRET:
        secrets_to_try.append(("INSTAGRAM_APP_SECRET", settings.INSTAGRAM_APP_SECRET))

    if not secrets_to_try:
        if _is_explicit_dev_mode():
            logger.warning("No app secrets set — skipping HMAC verification (dev mode)")
            return body
        raise AuthorizationError("META_APP_SECRET not configured.")

    signature_header = request.headers.get("X-Hub-Signature-256", "")
    if not signature_header.startswith("sha256="):
        raise AuthorizationError("Missing or malformed X-Hub-Signature-256 header.")

    received_sig = signature_header[len("sha256="):]

    for secret_name, secret_value in secrets_to_try:
        expected_sig = hmac.HMAC(secret_value.encode("utf-8"), body, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected_sig, received_sig):
            return body

    raise AuthorizationError("Invalid webhook signature.")
