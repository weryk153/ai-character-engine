from __future__ import annotations


class ExtensionError(RuntimeError):
    """Base error for the Plugin / Extension SDK."""


class PluginCompatibilityError(ExtensionError):
    """Raised when a plugin does not target this extension API."""


class PluginPolicyError(ExtensionError):
    """Raised when a host policy denies a plugin capability or extension point."""


class PluginRegistrationError(ExtensionError):
    """Raised when plugin registration violates the SDK contract."""


class ExtensionNotFoundError(ExtensionError, LookupError):
    """Raised when an extension id is unknown."""


class ExtensionValidationError(ExtensionError, TypeError):
    """Raised when a factory returns an object that does not satisfy its contract."""
