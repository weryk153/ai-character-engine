import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


def test_readme_chat_example_carries_context_without_network():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
    assert len(blocks) == 1, "README must contain one complete integration example"
    example = blocks[0]
    # Run the actual README code through CharacterRuntime; replace only inference.
    # Check the second call includes the first exchange, without a live provider.
    harness = r'''import asyncio
from unittest.mock import patch
from ai_character_engine.llm.local import OpenAICompatibleChatClient
from ai_character_engine.llm.models import LLMResponse
calls = []
async def offline_generate(self, messages, *, tools=None):
    calls.append([(m.role, m.content) for m in messages])
    return LLMResponse(text="Hello Alex.", model="offline")
with patch.object(OpenAICompatibleChatClient, "generate", offline_generate):
    exec(EXAMPLE, {"__name__": "__main__"})
assert len(calls) == 2
assert ("user", "Call me Alex.") in calls[1]
assert ("assistant", "Hello Alex.") in calls[1]
assert ("user", "What name did I ask you to use?") in calls[1]
'''.replace("EXAMPLE", repr(example))
    result = subprocess.run([sys.executable, "-c", harness], cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("Hello Alex.") == 2


def test_companion_example_runs_without_network(tmp_path):
    text = (ROOT / "docs/companion.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
    assert len(blocks) == 1, "the companion page must contain one complete example"
    harness = r'''import asyncio
from unittest.mock import patch
from ai_character_engine.llm.local import OpenAICompatibleChatClient
from ai_character_engine.llm.models import LLMResponse, LLMStreamChunk
calls = []
async def offline_generate(self, messages, *, tools=None):
    return LLMResponse(text="{}", model="offline")
async def offline_stream(self, messages, *, tools=None):
    calls.append([(m.role, m.content) for m in messages])
    # She does not say the same thing twice, so the second answer differs.
    text = "Hello Alex." if len(calls) == 1 else "You asked me to call you Alex."
    yield LLMStreamChunk(text=text)
    yield LLMStreamChunk(final=True, response=LLMResponse(text=text, model="offline"))
with patch.object(OpenAICompatibleChatClient, "generate", offline_generate):
    with patch.object(OpenAICompatibleChatClient, "stream_generate", offline_stream):
        exec(EXAMPLE, {"__name__": "__main__"})
assert len(calls) == 2
assert ("user", "Call me Alex.") in calls[1]
assert ("assistant", "Hello Alex.") in calls[1]
'''.replace("EXAMPLE", repr(blocks[0]))
    # Run where the example may write: it stores the character under the
    # working directory.
    result = subprocess.run(
        [sys.executable, "-c", harness], cwd=tmp_path, capture_output=True, text=True, timeout=20,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr
    assert "Hello Alex." in result.stdout
    assert "You asked me to call you Alex." in result.stdout
    assert (tmp_path / "companion-data/guide/state.json").is_file()


def test_product_documentation_local_links_resolve():
    pages = [*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md"), ROOT / "examples/README.md",
             *(ROOT / "packages").glob("*/README.md")]
    errors = []
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not (page.parent / target).exists():
                errors.append(f"{page.relative_to(ROOT)} -> {target}")
    assert not errors, "\n".join(errors)


def test_api_reference_covers_the_entire_stable_root_contract():
    manifest = json.loads((ROOT / "docs/public_api_v1_stable.json").read_text(encoding="utf-8"))
    headings = re.findall(r"^## (\w+)$", (ROOT / "docs/api-reference.md").read_text(encoding="utf-8"), re.M)
    assert sorted(headings) == sorted(symbol["name"] for symbol in manifest["symbols"])
