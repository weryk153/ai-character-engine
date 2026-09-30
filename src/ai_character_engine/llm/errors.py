class CharacterEngineError(Exception):
    """Base error for the package."""


class LLMError(CharacterEngineError):
    """Raised when an LLM provider call fails."""
