"""
Correlation ID middleware.

Ensures every request/response carries an ``X-Request-ID`` header.
"""

import uuid

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_HEADER = "X-Request-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every HTTP request and response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id

        with logger.contextualize(request_id=request_id):
            response = await call_next(request)

        response.headers[_HEADER] = request_id
        return response
