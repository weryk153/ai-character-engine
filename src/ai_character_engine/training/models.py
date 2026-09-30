from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]
QuantizationMode = Literal["none", "4bit", "8bit"]


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported role: {self.role}")
        _nonempty(self.content, "content")
        if self.role == "tool" and self.tool_call_id is not None:
            _nonempty(self.tool_call_id, "tool_call_id")
        if self.name is not None:
            _nonempty(self.name, "name")

    def to_dict(self) -> dict[str, Any]:
        data = {"role": self.role, "content": self.content}
        if self.name is not None:
            data["name"] = self.name
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass(frozen=True, slots=True)
class SFTExample:
    example_id: str
    messages: tuple[ChatMessage, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.example_id, "example_id")
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if not all(isinstance(m, ChatMessage) for m in self.messages):
            raise ValueError("messages must contain ChatMessage values")
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("SFT example requires at least one user message")
        if not any(m.role == "assistant" for m in self.messages):
            raise ValueError("SFT example requires at least one assistant message")
        if self.messages[-1].role != "assistant":
            raise ValueError("SFT example must end with an assistant message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.example_id,
            "messages": [m.to_dict() for m in self.messages],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SFTExample":
        return cls(
            example_id=data.get("id") or data.get("example_id"),
            messages=tuple(ChatMessage.from_dict(m) for m in data["messages"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    severity: Literal["warning", "error"]
    code: str
    message: str
    example_id: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetStats:
    examples: int
    messages: int
    user_messages: int
    assistant_messages: int
    assistant_chars: int
    average_messages_per_example: float
    average_assistant_chars: float


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    stats: DatasetStats
    issues: tuple[DatasetIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "stats": asdict(self.stats),
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class LoRAConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: tuple[str, ...] = ()
    modules_to_save: tuple[str, ...] = ()
    use_rslora: bool = False

    def __post_init__(self) -> None:
        if self.r < 1:
            raise ValueError("LoRA r must be >= 1")
        if self.alpha < 1:
            raise ValueError("LoRA alpha must be >= 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("unsupported LoRA bias mode")
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        object.__setattr__(self, "modules_to_save", tuple(self.modules_to_save))


@dataclass(frozen=True, slots=True)
class QuantizationConfigSpec:
    mode: QuantizationMode = "none"
    compute_dtype: str = "bfloat16"
    quant_type: Literal["nf4", "fp4"] = "nf4"
    double_quant: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"none", "4bit", "8bit"}:
            raise ValueError("quantization mode must be none/4bit/8bit")
        _nonempty(self.compute_dtype, "compute_dtype")


@dataclass(frozen=True, slots=True)
class SFTTrainingConfig:
    base_model: str
    output_dir: str
    learning_rate: float = 2e-4
    epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    packing: bool = False
    assistant_only_loss: bool = True
    seed: int = 42
    eval_ratio: float = 0.1
    lora: LoRAConfigSpec = field(default_factory=LoRAConfigSpec)
    quantization: QuantizationConfigSpec = field(default_factory=QuantizationConfigSpec)

    def __post_init__(self) -> None:
        _nonempty(self.base_model, "base_model")
        _nonempty(self.output_dir, "output_dir")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.max_length < 128:
            raise ValueError("max_length must be >= 128")
        if not 0 <= self.eval_ratio < 1:
            raise ValueError("eval_ratio must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    config: SFTTrainingConfig
    train_examples: int
    eval_examples: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "0.21",
            "config": asdict(self.config),
            "train_examples": self.train_examples,
            "eval_examples": self.eval_examples,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class AdapterManifest:
    adapter_path: str
    base_model: str
    method: Literal["lora", "qlora"]
    dataset_fingerprint: str
    examples: int
    created_by: str = "ai-character-engine"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.adapter_path, "adapter_path")
        _nonempty(self.base_model, "base_model")
        _nonempty(self.dataset_fingerprint, "dataset_fingerprint")
        if self.examples < 1:
            raise ValueError("examples must be >= 1")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        import json
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
