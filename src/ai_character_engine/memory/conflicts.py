"""A new fact about the user against the facts she already holds.

"I changed jobs" next to "works at X" is not two facts to keep side by side.
After a memory of the user is written, the earlier ones on the same topic are
picked out here without a model (``conflict_candidates``); only when there are
any is a model asked how the new fact stands to each of them, and its answer is
applied here (``apply_conflicts``):

- ``supersedes``: time moved on (a new job, a move, a break-up). The earlier
  fact is marked superseded and leaves her mind; nothing is asked.
- ``contradicts``: both cannot be true and nothing tells which came later. Both
  stay; the new one is marked, and she asks the user about it once.
- ``refines``: a detail added. Both stay as they are; the new one notes it.

Everything is kept on the records themselves, so it survives a restart.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from .models import MemoryRecord

CONFLICT_RELATIONS: tuple[str, ...] = ("supersedes", "contradicts", "refines")
# How many earlier facts one new fact is held against, at most.
CONFLICT_CANDIDATES_KEPT = 5
# Metadata of the newer record: the facts it contradicts and that are not
# settled, those of them she has been told to ask about, the facts it adds a
# detail to, and every relation applied, with the model's reason.
CONFLICT_WITH = "conflict_with"
CONFLICT_ASKED = "conflict_asked"
REFINES = "refines"
CONFLICTS_LOG = "memory_conflicts"

# Two embeddings this alike speak of the same thing.
_EMBEDDING_ALIKE = 0.85
# Words for the user, the subject of nearly every fact about the user: shared
# by all of them, they say nothing about the topic.
_THE_USER = re.compile(
    r"使用者|用戶|用户|ユーザー|ユーザ|對方|对方|\b(?:the\s+)?users?(?:'s)?\b|\bhe\b|\bshe\b|\bthey\b",
    re.IGNORECASE,
)
_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]+")
_LATIN_WORD = re.compile(r"[a-z][a-z']{3,}")
_STOP_WORDS = frozenset(
    "that this with have from they them their there what when were been into "
    "about also just like likes very some more most much will would could "
    "should does doing done than then said says user users".split()
)
# Topics a fact about someone's life is often about, and that a later fact
# changes without sharing a word: "works at X" and "changed jobs". A word of
# the same topic in both makes them candidates.
_TOPICS: Mapping[str, tuple[str, ...]] = {
    "work": (
        "工作", "上班", "公司", "職", "职", "辭", "辞", "離職", "跳槽", "老闆", "老板", "同事",
        "打工", "兼職", "仕事", "会社", "勤め", "転職", "退職", "バイト",
        "job", "work", "company", "employ", "office", "career", "boss", "colleague", "hired",
        "fired", "quit",
    ),
    "home": (
        "住", "搬家", "搬到", "搬去", "老家", "租屋", "房子", "公寓", "引っ越", "実家",
        "live", "lives", "living", "moved", "moving", "apartment", "flat", "hometown",
    ),
    "partner": (
        "男友", "女友", "男朋友", "女朋友", "交往", "分手", "結婚", "结婚", "離婚", "离婚",
        "老婆", "老公", "太太", "丈夫", "妻子", "未婚", "單身", "单身", "彼氏", "彼女",
        "付き合", "別れ", "boyfriend", "girlfriend", "partner", "married", "marry",
        "divorce", "dating", "broke up", "single", "wife", "husband", "fiancé", "fiance",
    ),
    "school": (
        "學校", "学校", "大學", "大学", "高中", "國中", "研究所", "畢業", "毕业", "念書",
        "讀書", "读书", "學生", "学生", "主修", "科系", "留學", "留学", "卒業", "専攻",
        "school", "university", "college", "graduat", "student", "major", "study", "studies",
        "studying", "degree",
    ),
    "pet": ("貓", "猫", "狗", "犬", "寵物", "宠物", "ペット", "cat", "cats", "dog", "dogs", "pet", "pets"),
    "age": ("歲", "岁", "歳", "年紀", "年纪", "生日", "age", "years old", "birthday"),
    "name": ("名字", "叫", "名前", "name", "called", "nickname"),
}
# Characters in nearly every sentence, which say nothing about its topic.
_FUNCTION_CHARACTERS = frozenset(
    "的了是在有和與与也都很就不沒没我你他她它們们這这那個个一會会要說说到過过著着得把被讓让給给"
    "對对從从為为而及或嗎吗呢吧啊喔哦嗯人很還还又再才已經经最多少些"
    "のはがをにでともへやかなだですますたしてい"
)
# Characters of Chinese or Japanese two facts share, besides the above, to
# be about the same thing: 「準備考試」 and 「考完試了」 share no two in a row.
_SHARED_CHARACTERS = 2


# What the user's own words must say for a fact to replace an earlier one:
# that something changed, ended or was done. A 9B model, asked whether 「我25歲」
# says a change, said yes; these words are not left to it. Narrow on purpose:
# a change missed is asked about, a fact replaced wrongly is gone.
_CHANGE_CUES = (
    "換", "换", "改成", "改到", "搬", "分手", "離職", "离职", "辭職", "辞职", "辭掉", "辞掉",
    "不做了", "畢業", "毕业", "結婚了", "结婚了", "離婚", "离婚", "不再", "已經不", "已经不",
    "不住", "沒在", "没在", "轉職", "转职", "轉學", "转学", "退休", "退學", "退学", "回來了",
    "回来了", "結束了", "结束了", "過世", "过世", "去世", "考完", "跳槽", "被裁", "失業", "失业",
    "入職", "入职", "錄取", "录取", "交往了", "在一起了", "賣掉", "卖掉", "戒了", "戒掉",
    "変え", "変わっ", "引っ越", "辞め", "別れ", "卒業", "結婚した", "離婚", "転職", "退職",
    "亡くな", "戻っ", "終わっ",
)
_CHANGE_WORDS = re.compile(
    r"\b(?:changed|switched|moved|quit|left|broke up|graduated|finished|got married|married|"
    r"divorced|retired|passed away|died|came back|no longer|any ?more|resigned|laid off|"
    r"fired|sold|new job)\b"
)
# A change still to come: the earlier fact is still true.
_PLAN_CUES = (
    "要", "打算", "準備", "准备", "計畫", "计划", "計劃", "下個", "下个", "下週", "下周", "明天",
    "明年", "之後", "之后", "以後", "以后", "將", "将", "即將", "即将", "預計", "预计",
    "予定", "つもり", "来月", "来年", "来週", "明日", "これから", "しよう", "したい",
)
_PLAN_WORDS = re.compile(
    r"\b(?:will|going to|gonna|plans?|planning|next|tomorrow|soon|want to|thinking of|"
    r"thinking about|about to|intend)\b|'ll\b"
)
# One day, set against how things usually are, is neither a change nor a
# contradiction: 「這週末沒去爬山」 against 「週末常去爬山」.
_ONE_TIME_CUES = (
    "今天", "昨天", "前天", "今晚", "昨晚", "今早", "這週", "这周", "這禮拜", "这礼拜", "剛剛",
    "刚刚", "剛才", "刚才", "這次", "这次", "那天", "上次", "今日", "昨日", "今朝", "今夜", "今週",
    "今回", "さっき",
)
_ONE_TIME_WORDS = re.compile(
    r"\b(?:today|yesterday|tonight|last night|this morning|this weekend|this week|just now|"
    r"this time)\b"
)
_HABIT_CUES = (
    "常", "每天", "每週", "每周", "每個", "每个", "每次", "總是", "总是", "習慣", "习惯", "通常",
    "平常", "平時", "平时", "固定", "いつも", "毎日", "毎週", "毎朝", "よく", "普段",
)
_HABIT_WORDS = re.compile(r"\b(?:usually|every|often|always|habit|regularly|normally)\b")


def _has(text: str, cues: Sequence[str], words: re.Pattern[str]) -> bool:
    folded = text.casefold()
    return any(cue in folded for cue in cues) or words.search(folded) is not None


def says_a_plan(text: str) -> bool:
    """Whether the words speak of something still to come."""
    return _has(text, _PLAN_CUES, _PLAN_WORDS)


def says_a_change(text: str) -> bool:
    """Whether the words say, in so many words, that something changed,
    ended or was done; a plan is no change yet."""
    return _has(text, _CHANGE_CUES, _CHANGE_WORDS) and not says_a_plan(text)


def one_day_against_a_habit(newer: str, earlier: str) -> bool:
    """Whether the new fact is about one day and the earlier one about how
    things usually are: both stay true."""
    return (
        _has(newer, _ONE_TIME_CUES, _ONE_TIME_WORDS)
        and _has(earlier, _HABIT_CUES, _HABIT_WORDS)
        and not _has(newer, _CHANGE_CUES, _CHANGE_WORDS)
    )


def _plain(summary: str) -> str:
    return _THE_USER.sub(" ", summary.casefold())


def _pieces(summary: str) -> set[str]:
    """What two facts must share to be about the same thing: two characters
    in a row of Chinese, Japanese or Korean, or the first four letters of a
    longer word."""
    text = _plain(summary)
    pieces = {
        run[index : index + 2]
        for run in _CJK_RUN.findall(text)
        for index in range(len(run) - 1)
    }
    pieces.update(word[:4] for word in _LATIN_WORD.findall(text) if word not in _STOP_WORDS)
    return pieces


def _characters(summary: str) -> set[str]:
    return {
        char
        for run in _CJK_RUN.findall(_plain(summary))
        for char in run
        if char not in _FUNCTION_CHARACTERS
    }


def _topics(summary: str) -> set[str]:
    text = _plain(summary)
    words = set(re.findall(r"[a-z]+", text))
    found = set()
    for topic, cues in _TOPICS.items():
        for cue in cues:
            if cue.isascii() and " " not in cue:
                if cue in words or (len(cue) >= 6 and any(w.startswith(cue) for w in words)):
                    found.add(topic)
                    break
            elif cue in text:
                found.add(topic)
                break
    return found


def _cosine(one: Sequence[float] | None, other: Sequence[float] | None) -> float:
    if not one or not other or len(one) != len(other):
        return 0.0
    dot = sum(a * b for a, b in zip(one, other))
    norm = math.sqrt(sum(a * a for a in one)) * math.sqrt(sum(b * b for b in other))
    return dot / norm if norm else 0.0


def about_the_user(record: MemoryRecord) -> bool:
    """A fact about the user, not a summary of the conversation."""
    return (
        record.kind != "conversation_summary"
        and record.metadata.get("evidence_type") != "conversation_summary"
    )


def conflict_candidates(
    new: MemoryRecord,
    records: Iterable[MemoryRecord],
    *,
    limit: int = CONFLICT_CANDIDATES_KEPT,
) -> list[MemoryRecord]:
    """The earlier facts about the user, still held, that ``new`` may change:
    on the same topic, sharing a name or words, or with alike embeddings. The
    most alike first, then the newest. No model is asked."""
    earlier = [
        record
        for record in records
        if record.id != new.id
        and record.is_active
        and about_the_user(record)
        and record.created_at <= new.created_at
    ]
    if not earlier or limit <= 0:
        return []
    pieces_of = {
        record.id: _pieces(record.summary) | {f"#{c}" for c in _characters(record.summary)}
        for record in earlier
    }
    # A piece or character in more than half of what she holds is how facts
    # are worded here ("喜歡"), not what they are about.
    common: set[str] = set()
    if len(earlier) >= 4:
        counts: dict[str, int] = {}
        for pieces in pieces_of.values():
            for piece in pieces:
                counts[piece] = counts.get(piece, 0) + 1
        common = {piece for piece, count in counts.items() if count * 2 > len(earlier)}
    new_pieces = (
        _pieces(new.summary) | {f"#{c}" for c in _characters(new.summary)}
    ) - common
    new_topics = _topics(new.summary)
    scored: list[tuple[float, MemoryRecord]] = []
    for record in earlier:
        both = new_pieces & pieces_of[record.id]
        characters = sum(piece.startswith("#") for piece in both)
        shared = {piece for piece in both if not piece.startswith("#")}
        topics = new_topics & _topics(record.summary)
        alike = _cosine(new.embedding, record.embedding)
        if (
            not shared
            and not topics
            and characters < _SHARED_CHARACTERS
            and alike < _EMBEDDING_ALIKE
        ):
            continue
        score = len(shared) + 2 * len(topics) + 0.5 * characters + 4 * max(0.0, alike)
        scored.append((score, record))
    scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
    return [record for _, record in scored[:limit]]


def _ids(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def apply_conflicts(
    records: Sequence[MemoryRecord],
    new_id: str,
    conflicts: Iterable[Any],
) -> tuple[list[MemoryRecord], tuple[dict[str, str], ...]]:
    """``records`` with each relation of the new fact applied, and what was
    applied. A relation off the list, a fact that is not held, or the new fact
    itself is passed over: nothing is changed for it."""
    by_id = {record.id: record for record in records}
    new = by_id.get(new_id)
    if new is None or not new.is_active:
        return list(records), ()
    applied: list[dict[str, str]] = []
    superseded: dict[str, MemoryRecord] = {}
    metadata = dict(new.metadata)
    supersedes = list(new.supersedes)
    for item in conflicts:
        if not isinstance(item, Mapping):
            continue
        old_id = str(item.get("old_id") or "")
        relation = str(item.get("relation") or "").strip().casefold()
        old = by_id.get(old_id)
        if (
            relation not in CONFLICT_RELATIONS
            or old is None
            or old_id == new_id
            or not old.is_active
            or old_id in superseded
            or any(done["old_id"] == old_id for done in applied)
        ):
            continue
        entry = {
            "old_id": old_id,
            "relation": relation,
            "reason": str(item.get("reason") or "").strip(),
        }
        if relation == "supersedes":
            superseded[old_id] = replace(old, status="superseded", superseded_by=new_id)
            supersedes.append(old_id)
        elif relation == "contradicts":
            metadata[CONFLICT_WITH] = [*_ids(metadata.get(CONFLICT_WITH)), old_id]
        else:
            metadata[REFINES] = [*_ids(metadata.get(REFINES)), old_id]
        applied.append(entry)
    if not applied:
        return list(records), ()
    metadata[CONFLICTS_LOG] = [*list(metadata.get(CONFLICTS_LOG) or ()), *applied]
    revised = replace(new, metadata=metadata, supersedes=tuple(dict.fromkeys(supersedes)))
    result = [
        revised if record.id == new_id else superseded.get(record.id, record)
        for record in records
    ]
    return result, tuple(applied)


def _reason(record: MemoryRecord, old_id: str) -> str:
    for entry in record.metadata.get(CONFLICTS_LOG) or ():
        if isinstance(entry, Mapping) and entry.get("old_id") == old_id:
            return str(entry.get("reason") or "")
    return ""


def unresolved_conflicts(
    records: Sequence[MemoryRecord],
) -> list[tuple[MemoryRecord, MemoryRecord, str]]:
    """(the newer fact, the earlier one, the reason) for every contradiction
    of two facts both still held, oldest first."""
    by_id = {record.id: record for record in records}
    found = []
    for record in sorted(records, key=lambda item: item.created_at):
        if not record.is_active:
            continue
        for old_id in _ids(record.metadata.get(CONFLICT_WITH)):
            old = by_id.get(old_id)
            if old is not None and old.is_active:
                found.append((record, old, _reason(record, old_id)))
    return found


def take_unasked_conflicts(
    records: Sequence[MemoryRecord],
) -> tuple[list[MemoryRecord], list[tuple[MemoryRecord, MemoryRecord, str]]]:
    """The contradictions she has not been told to ask about, and ``records``
    with them marked as told: each is asked about once."""
    unasked = [
        conflict
        for conflict in unresolved_conflicts(records)
        if conflict[1].id not in _ids(conflict[0].metadata.get(CONFLICT_ASKED))
    ]
    if not unasked:
        return list(records), []
    told: dict[str, list[str]] = {}
    for newer, older, _ in unasked:
        told.setdefault(newer.id, _ids(newer.metadata.get(CONFLICT_ASKED))).append(older.id)
    result = [
        replace(record, metadata={**record.metadata, CONFLICT_ASKED: told[record.id]})
        if record.id in told
        else record
        for record in records
    ]
    return result, unasked


def resolve_conflict(
    records: Sequence[MemoryRecord], keep_id: str
) -> tuple[list[MemoryRecord], bool]:
    """Settle every contradiction ``keep_id`` is in: it stays, the fact
    against it is marked superseded by it. Unchanged, and False, when it is in
    none."""
    pairs = [
        (newer, older)
        for newer, older, _ in unresolved_conflicts(records)
        if keep_id in (newer.id, older.id)
    ]
    if not pairs:
        return list(records), False
    dropped = {older.id if newer.id == keep_id else newer.id for newer, older in pairs}
    settled = {newer.id: [] for newer, _ in pairs}
    for newer, older in pairs:
        settled[newer.id].append(older.id)
    result = []
    for record in records:
        if record.id in dropped:
            record = replace(record, status="superseded", superseded_by=keep_id)
        if record.id in settled:
            remaining = [
                old_id
                for old_id in _ids(record.metadata.get(CONFLICT_WITH))
                if old_id not in settled[record.id]
            ]
            record = replace(record, metadata={**record.metadata, CONFLICT_WITH: remaining})
        result.append(record)
    return result, True
