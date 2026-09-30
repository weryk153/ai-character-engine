class CognitiveRoutingError(RuntimeError):
    """Base error for cognitive model routing."""


class MissingCognitiveRolePolicyError(CognitiveRoutingError):
    """No routing policy exists for a requested cognitive role."""


class NoEligibleCognitiveModelError(CognitiveRoutingError):
    """No configured model endpoint satisfies a cognitive routing request."""
