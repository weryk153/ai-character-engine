from __future__ import annotations


class CharacterServiceError(Exception):
    """Base error exposed by the service layer."""

    status_code = 500
    code = "service_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


class ServiceNotFoundError(CharacterServiceError):
    status_code = 404
    code = "not_found"


class ServiceSessionUnavailableError(CharacterServiceError):
    status_code = 410
    code = "session_unavailable"


class ServiceValidationError(CharacterServiceError):
    status_code = 400
    code = "validation_error"


class ServiceTimeoutError(CharacterServiceError):
    status_code = 504
    code = "timeout"


class ServiceUpstreamError(CharacterServiceError):
    status_code = 502
    code = "upstream_error"


class ServiceToolError(CharacterServiceError):
    status_code = 502
    code = "tool_error"


class ServiceUnauthorizedError(CharacterServiceError):
    status_code = 401
    code = "unauthorized"


class ServiceForbiddenError(CharacterServiceError):
    status_code = 403
    code = "forbidden"


class ServiceRateLimitError(CharacterServiceError):
    status_code = 429
    code = "rate_limited"
