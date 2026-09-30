class MultiCharacterError(RuntimeError):
    pass


class UnknownCharacterError(MultiCharacterError):
    pass


class CharacterIsolationError(MultiCharacterError):
    pass


class SharedCognitionAccessError(MultiCharacterError):
    pass


class CharacterSchedulerOverloadedError(MultiCharacterError):
    pass
