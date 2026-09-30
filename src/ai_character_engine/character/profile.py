from dataclasses import dataclass, field


@dataclass(slots=True)
class CharacterProfile:
    id: str
    name: str
    description: str
    personality: list[str] = field(default_factory=list)
    speaking_style: list[str] = field(default_factory=list)
    background: str | None = None
    rules: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("character id must not be empty")
        if not self.name.strip():
            raise ValueError("character name must not be empty")
