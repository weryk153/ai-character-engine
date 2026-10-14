# Configuration

The host supplies dependencies, clients, stores, limits and lifecycle policy.
Core construction is explicit; there is no global cloud control plane or renderer
configuration that changes runtime authority.

## Environment used by sample hosts

| Variable | Consumer | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | Responses sample / provider SDK | Authentication for the chosen endpoint |
| `AI_CHARACTER_MODEL` | Interactive chat samples | Explicit model identifier |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Ollama sample | Host-provided local inference endpoint and model |
| `VLLM_BASE_URL`, `VLLM_MODEL` | vLLM samples | Private inference endpoint and model |
| `VLLM_API_KEY` | Remote vLLM sample | Optional endpoint authentication |

`.env.example` is a template. The sample hosts load `.env`; embedding applications
can construct clients directly. Keep credentials and local endpoint settings out
of version control. The core does not read these variables to make autonomous
routing decisions.

## Optional dependency groups

| Extra | Purpose |
|---|---|
| `service` | FastAPI, HTTP streaming and WebSocket host |
| `local` | HTTP client support for local inference integrations |
| `qdrant` | Qdrant-backed retrieval integration |
| `voice-audio` | Audio I/O and HTTP voice integrations |
| `voice-sherpa` | Sherpa ONNX speech recognition integration |
| `voice-mac` | macOS speech host dependencies |
| `training` / `training-qlora` | Explicit offline model-training workflow |
| `dev` | Tests and development tools |

Training dependencies and audio providers are opt-in. They are not required to
run a text character or import the engine. A package extra installs dependencies;
it does not download models, start servers or certify hardware compatibility.

## Runtime choices

- Define identity and response constraints in `CharacterProfile`.
- Pass an `LLMClient` implementation to `CharacterRuntime`.
- Register permitted tools and schemas in `ToolRegistry`; handlers stay host-owned.
- Configure context, retrieval and state policies explicitly for the character.
- Assign stable session and memory scopes. Choose persistent store paths and
  backup/retention rules in the host.
- Set worker timeouts, concurrency and idempotency policy before dispatching work.

`OpenAICompatibleChatClient` exposes endpoint, timeout, retry and concurrency
settings. Retry policy must account for tool side effects; a successful remote
request and a successful local commit are separate events.

### Interaction experience (development candidate)

With `CompanionSettings(character_development=True)`, `interaction_every=1`
reads explicit feedback on a preceding character reply. Set it to `0` to stop
learning. `interaction_topics_shown=4` bounds the recent topic interpretations
appended to context. For a background client mapping, supply the `interaction`
key; a single background client is shared with this worker automatically.

Pass a stable, host-owned `user_id` to `reply` or `speak_up`. The default `None`
disables both retrieval and learning for that turn. A conversation ID or display
name is not a user identity. Hosts must use a separate conversation for a different
person and omit identity for unrecognized/group/stream input. `speak_up` can read
existing experiences but cannot learn a response from an idle event.

The background worker proposes one interpretation using exact quotations of the
preceding character reply and current feedback. The coordinator validates quotes,
source event, topic ownership and confidence before appending it to the existing
memory store under a character-and-user scope. Replays of the same event do not
add support. Later feedback can revise a topic while retaining original episodes.
These are fallible interpretations, not facts, personality rewrites, relationship
stages or permissions. Quotation validation does not prove semantic correctness.

Use `companion.interaction_experiences(user_id)` to inspect active records and
`companion.forget_interaction_experiences(user_id, record_ids)` to withdraw records.
Persistence/export use the existing memory JSONL surface. Raw quotations and user
identity are retained; the scope hash is not anonymization. Different users' scopes
are isolated; initial persona and global self-memory remain separate.

Local paired conversations with Qwen3.5-9B demonstrated retrieval after restart
and partner isolation, but also speaker confusion and unsupported interpretations
even when the quotations were valid. This candidate remains off by default.
Do not treat an accepted record or passing mechanism tests as evidence that a
model has learned the right lesson; inspect complete conversations and records
before enabling it in a host.

Interaction candidates now require a separate evidence-review call before commit.
The reviewer receives the exchange and proposed interpretation, without the
character's persona. It checks speaker attribution, explicit feedback, support
for the lesson and outcome, scope, and the proposed topic revision. Unsupported
lessons stay out of context. When the outcome and all other checks pass, the
verified outcome can be retained without the lesson; the rejected lesson is
kept only in audit metadata. Missing, malformed, or uncertain required checks
produce no interaction memory. This is
a fresh call to the configured interaction model, not an independent model vendor
or a guarantee of semantic correctness.

The coordinator requires a positive review bound to the exact candidate and
feedback, and persists that review with the quotations. A later change cannot
reuse the earlier review. Existing records without a valid review are preserved
for inspection/withdrawal but excluded from context and from the prior topics
used for new learning. `interaction_experiences` still exposes those historical
records; the presence of a record alone does not mean it is used in prompts.
Each valid candidate adds one background model call, under the existing job
limit and timeout. Review failure leaves the conversation reply intact.
