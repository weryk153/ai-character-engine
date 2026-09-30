from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


class InjectedFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FailureRule:
    calls: frozenset[int]
    exception_factory: Callable[[int], BaseException]

    def __init__(self, calls: Iterable[int], exception_factory: Callable[[int], BaseException]) -> None:
        cleaned = frozenset(int(item) for item in calls)
        if any(item < 1 for item in cleaned):
            raise ValueError("failure call numbers must be >= 1")
        object.__setattr__(self, "calls", cleaned)
        object.__setattr__(self, "exception_factory", exception_factory)


class FailureInjector:
    """Deterministic, test-only fault injection counter.

    It contains no provider/runtime authority and never patches global functions.
    The caller explicitly invokes `checkpoint()` at a failure boundary.
    """

    def __init__(self, *rules: FailureRule) -> None:
        self._rules = tuple(rules)
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def reset(self) -> None:
        self._calls = 0

    def checkpoint(self) -> int:
        self._calls += 1
        call = self._calls
        for rule in self._rules:
            if call in rule.calls:
                raise rule.exception_factory(call)
        return call


def injected_failure(message: str = "injected failure") -> Callable[[int], BaseException]:
    return lambda call: InjectedFailure(f"{message} at call {call}")
