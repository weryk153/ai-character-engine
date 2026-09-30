from __future__ import annotations

import re
from typing import Literal

from ai_character_engine.events.models import CharacterEvent

MemoryEvidenceType = Literal[
    "asserted_fact",
    "memory_operation",
    "user_question",
    "quoted_reference",
    "user_instruction",
    "event_observation",
    "unknown",
]

_VALID_EVIDENCE_TYPES: set[str] = {
    "asserted_fact",
    "memory_operation",
    "user_question",
    "quoted_reference",
    "user_instruction",
    "event_observation",
    "unknown",
}

# Direct first-person assertions outrank punctuation-based question detection.
# This preserves messages such as "我喜歡七武士，你呢？" ("I like Seven Samurai,
# and you?") as user evidence.
_ASSERTION_PATTERNS = (
    re.compile(r"(?:^|[，,。；;：:\s])我叫\S+"),
    re.compile(r"(?:^|[，,。；;：:\s])我是\S+"),
    re.compile(r"(?:^|[，,。；;：:\s])我(?:正在|目前|現在|现在)\S*"),
    re.compile(r"(?:^|[，,。；;：:\s])我(?:最)?(?:喜歡|喜欢|偏好|想要|需要)\S*"),
    re.compile(r"(?:^|[，,。；;：:\s])我的\S{0,18}(?:是|叫|為|为|改成|變成|变成)\S+"),
    re.compile(r"(?:^|\s)(?:my name is|i am|i'm|i like|i prefer|my .{0,30} is)\b", re.I),
)

_FORGET_PATTERNS = (
    "忘記",
    "別記",
    "不要記得",
    "不要再記",
    "forget",
    "don't remember",
    "do not remember",
)

_QUOTE_MARKERS = (
    "引文",
    "引用",
    "待分析",
    "以下文字",
    "下面文字",
    "quoted text",
    "quote:",
    "quotation",
)

_INSTRUCTION_PREFIXES = (
    "請",
    "请",
    "幫",
    "帮",
    "用",
    "列出",
    "安排",
    "解釋",
    "解释",
    "告訴",
    "告诉",
    "分析",
    "比較",
    "比较",
    "summarize",
    "explain",
    "list",
    "plan",
    "tell me",
)

_QUESTION_MARKERS = (
    "嗎",
    "吗",
    "什麼",
    "什么",
    "哪",
    "幾",
    "几",
    "多少",
    "怎麼",
    "怎么",
    "是否",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
)


def classify_user_text(text: str) -> MemoryEvidenceType:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return "unknown"

    if any(signal in lowered for signal in _FORGET_PATTERNS):
        return "memory_operation"

    # Property-recall questions can look syntactically like assertions
    # ("我的名字是什麼？", "what is my name?"). Catch them before first-person
    # assertion rules.
    if re.search(
        r"我的.{0,24}(?:是|叫|為|为)(?:什麼|什么|誰|谁|哪|幾|几|多少)", stripped
    ):
        return "user_question"

    # Direct assertions win even when the same sentence also contains a question.
    if any(pattern.search(stripped) for pattern in _ASSERTION_PATTERNS):
        return "asserted_fact"

    if any(marker in lowered for marker in _QUOTE_MARKERS):
        return "quoted_reference"
    if ("「" in stripped and "」" in stripped) or ("“" in stripped and "”" in stripped):
        return "quoted_reference"

    if "?" in stripped or "？" in stripped or any(marker in lowered for marker in _QUESTION_MARKERS):
        return "user_question"

    if any(lowered.startswith(prefix.lower()) for prefix in _INSTRUCTION_PREFIXES):
        return "user_instruction"

    # A plain declarative user message is the safest default evidence class.
    return "asserted_fact"


def classify_memory_evidence(event: CharacterEvent) -> MemoryEvidenceType:
    explicit = str(event.payload.get("memory_evidence_type", "")).strip().lower()
    if explicit in _VALID_EVIDENCE_TYPES:
        return explicit  # type: ignore[return-value]
    if event.type != "user_message":
        return "event_observation"
    return classify_user_text(event.content)


def evidence_weight(evidence_type: str) -> float:
    """Conservative retrieval weight for old and new stores.

    Audit data remains in the ledger, but facts directly asserted by the user
    should dominate questions, quoted material and one-off instructions.
    """

    return {
        "asserted_fact": 1.25,
        "event_observation": 1.00,
        "unknown": 1.00,
        "memory_operation": 0.25,
        "user_question": 0.25,
        "user_instruction": 0.20,
        "quoted_reference": 0.10,
    }.get(evidence_type, 1.0)
