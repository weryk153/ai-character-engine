class ToolError(Exception):
    """Base error for tool registration, validation, and execution."""


class ToolNotFoundError(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


class ToolTimeoutError(ToolExecutionError):
    pass


class ToolPermissionError(ToolError):
    pass
