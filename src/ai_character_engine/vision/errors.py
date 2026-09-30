class VisionError(Exception):
    pass


class VisionInputError(VisionError):
    pass


class VisionRateLimitError(VisionError):
    pass


class VisionProviderError(VisionError):
    pass
