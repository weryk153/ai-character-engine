class TaskRuntimeError(RuntimeError):
    """Base exception for the multi-task runtime."""


class TaskRuntimeClosedError(TaskRuntimeError):
    """The task runtime is closed and cannot accept new work."""


class TaskQueueFullError(TaskRuntimeError):
    """The bounded background queue is full."""


class UnknownTaskTypeError(TaskRuntimeError):
    """No background worker is registered for a requested task type."""


class UnknownTaskError(TaskRuntimeError):
    """The requested task id is unknown to this runtime."""
