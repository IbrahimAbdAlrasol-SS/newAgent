"""
Custom exception hierarchy for the application.

All business exceptions inherit from AppError.
Each exception maps to an appropriate HTTP status code.
"""

from typing import Any


class AppError(Exception):
    """Base application error."""

    status_code: int = 500
    detail: str = "An internal error occurred."

    def __init__(self, detail: str | None = None, **kwargs: Any):
        self.detail = detail or self.__class__.detail
        self.extra = kwargs
        super().__init__(self.detail)


class NotFoundError(AppError):
    """Resource not found (404)."""

    status_code = 404
    detail = "Resource not found."


class ConflictError(AppError):
    """Duplicate / conflict (409)."""

    status_code = 409
    detail = "Resource already exists."


class ValidationError(AppError):
    """Business validation failed (422)."""

    status_code = 422
    detail = "Validation failed."


class AuthenticationError(AppError):
    """Authentication required or failed (401)."""

    status_code = 401
    detail = "Authentication required."


class AuthorizationError(AppError):
    """Insufficient permissions (403)."""

    status_code = 403
    detail = "Insufficient permissions."


class RateLimitError(AppError):
    """Rate limit exceeded (429)."""

    status_code = 429
    detail = "Rate limit exceeded. Please try again later."


class ExternalServiceError(AppError):
    """External service (Meta API, AI engine) failure (502)."""

    status_code = 502
    detail = "External service unavailable."
