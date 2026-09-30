class DistributedRuntimeError(RuntimeError):
    """Base error for transport-neutral distributed task execution."""


class DistributedQueueFullError(DistributedRuntimeError):
    pass


class DistributedTaskNotFoundError(DistributedRuntimeError):
    pass


class DuplicateDistributedTaskError(DistributedRuntimeError):
    pass
