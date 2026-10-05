# Installation and first integration

## Requirements

Use Python 3.11, 3.12 or 3.13. Core CI covers Linux, macOS and Windows.
Optional audio, training and renderer integrations may need their own native
libraries, services or hardware; core CI does not certify every device stack.

Clone the repository:

```sh
git clone https://github.com/weryk153/ai-character-engine.git
cd ai-character-engine
python -m venv .venv
```

Activate `.venv/bin/activate` on macOS/Linux or `.venv\Scripts\Activate.ps1`
in Windows PowerShell, then install:

```sh
python -m pip install .
python -c "import ai_character_engine; print(ai_character_engine.__version__)"
```

The version should be `1.1.1`. The root README shows a two-turn character connected
to a running local model. For an already built distribution, install the local
wheel instead of the source directory.

## Model-backed conversation

Copy `.env.example` to `.env`, supply `OPENAI_API_KEY` and an available
`AI_CHARACTER_MODEL`, then run:

```sh
python examples/basic_chat.py
```

`basic_chat.py` uses the Responses adapter. It reads the selected model from the
host environment; the engine does not choose a provider or model automatically.
Use `/exit` to stop the example.

For a local or private compatible endpoint, construct
`ai_character_engine.llm.local.OpenAICompatibleChatClient` with explicit `model`
and `base_url` arguments, and pass it as `llm` to `CharacterRuntime`. The host is
responsible for running and authenticating the inference server. Provider sample
entry points are listed in [examples](../examples/README.md).

A custom client implements asynchronous `generate(messages, *, tools=None)` and
returns `LLMResponse`. True streaming additionally implements `stream_generate`;
buffered HTTP output alone is not proof of provider token streaming.

## Persist a session

Use `CharacterRuntimeFactory` and `SessionManager` to create or restore a runtime.
Supply `JsonFileSessionStore` and `JsonFileRelationshipStore` with host-owned
paths. `python examples/session_runtime.py` exercises save and restore using a
temporary directory and an offline client. Replace those temporary paths and the
client in a real host; the example deliberately leaves no persistent user data.

## Optional packages

```sh
python -m pip install '.[service]'
python -m pip install ./packages/renderer-vrm
```

Install only the integrations your host uses. See [configuration](configuration.md)
for the available extras and [operations](operations.md) for deployment boundaries.
