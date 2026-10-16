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
The evidence review, the condition comparison and votes use the
`interaction_review` key of that mapping, or `CharacterCompanion(review_llm=...)`,
when given: a larger model than the one that extracts (a local 9B extracts
candidates well enough, but let reworded conditions through when it reviewed
them; a 27B did not). Without one they use the interaction client.

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
character's persona. It paraphrases the feedback, lists every condition,
exception or limit the user attached (`user_conditions`), checks explicit
feedback, and then names the problems it finds as short sentences, separately
for the topic update, the outcome and the lesson (`topic_problems`,
`outcome_problems`, `lesson_problems`; an empty list is a pass). Problems
include unsupported feelings, motives or facts, widened limits and wrong
speakers. A boolean verdict beside a free-text reason came back contradicting
it on three models, so there is no verdict apart from the problems named. A
problem found in the lesson no longer discards a supported outcome; a problem
in the outcome discards everything.
The supporting quote must be an exact excerpt of the feedback; several exact
excerpts may be joined in their original order with an ellipsis, and nothing
else counts as a quotation. When the review lists conditions, a second call
compares each condition with the outcome and with the lesson, seeing only
those texts (`kept`, `absent`, or `changed`); a component that changes a
condition (weakened, widened or replaced) is not stored. A `changed` verdict
must cite the exact words of that component which state the different
trigger, scope or consequence; a verdict with nothing to cite makes the
comparison unusable. Leaving a condition unstated is `absent`, not `changed`:
the conditions that are the user's own words are stored with the review and
shown beside the interpretation as `conditions`, so a shorter summary does not
lose them. Missing, malformed, or uncertain required checks produce no
interaction memory. These
are fresh calls to the configured interaction model, not an independent model
vendor or a guarantee of semantic correctness.

An interpretation starts with a support of 2. Later feedback on a topic she
already holds an interpretation of (the extractor names its `topic_id`
without `revises`) is not extracted and reviewed again: one call reads the
exchange against that interpretation and answers `UPVOTE`, `DOWNVOTE`,
`EDIT` or `NOOP`; an up- or downvote counts only with the user's words quoted
exactly, and `EDIT` (the words add or change a limit the interpretation does
not state) sends the candidate through the full review instead. Support 3 and above is repeated support; at 1 the lesson leaves the
context and only the observed outcome is shown; at 0 the record is marked
forgotten, kept on disk with its objections, and the topic shows nothing
rather than falling back to the view that record had replaced. One wrong
vote in either direction is a step, not a verdict. A revision (`revises=true`) still goes
through the full extraction review and supersedes the earlier view.

`interaction_review_checklist=True` replaces the review's problem lists with
six yes/no questions, one call each: for the outcome and for the lesson,
whether it states something the user did not say, puts words on the wrong
speaker, or widens a permission or limit. A yes counts only when it copies the
exact words of that text (and, for widening, of the user's narrower words);
a yes with nothing to cite is treated as not found, and a missing or
malformed answer stores nothing. The first review still has to establish
explicit feedback, the supporting quote, the conditions and a clean topic
(its `topic_problems`), and the condition comparison runs as before. This is for a reviewer that invents problems when
asked to list them, or never leaves the list empty (a 12B and a 9B did, in
turn); it costs six more short calls per candidate and stores proof version 4.

The coordinator requires a positive review bound to the exact candidate and
feedback, and persists that review with the quotations. A later change cannot
reuse the earlier review. Existing records without a valid review are preserved
for inspection/withdrawal but excluded from context and from the prior topics
used for new learning. `interaction_experiences` still exposes those historical
records; the presence of a record alone does not mean it is used in prompts.
Records reviewed under the earlier single-verdict review (proof version 2)
still verify; new reviews are stored as version 3 with the condition
comparison. Each valid candidate adds one background model call, two when the
user stated conditions, under the existing job limit and timeout. Review
failure leaves the conversation reply intact.

`interaction_review_mode` chooses how the extraction review judges the outcome
and the lesson: `""` (the default) uses the problem lists, or the six-question
checklist when `interaction_review_checklist` is set; `"checklist"` forces the
checklist; `"claims"` asks, once per component, for the claims it makes, each as
`{claim, subject, said_by, quote}` with `subject` one of `user`, `character`,
`other` and `said_by` one of `user`, `character`, `nobody`, `plan`. Code then
judges the rest. A claim is unstated when nobody said it, when its plan
breaks the plan rule (a plan may stand only in the lesson, with no quote and no
reason word), or when its quote is found on neither side; it is on the wrong
speaker when its quote is found only on the other speaker's side, or when its
subject is the user and the character is named as having said it. A quote needs
at least 4 non-space characters, unless it is the whole of a source shorter than
that. Whether the lesson widens a
limit stays a yes/no question. This takes 3 calls instead of 6 and stores proof
version 5. It is not the default; compare it with your model before enabling it.

### Relationship views (development candidate)

With `character_development=True` and a `user_id`, a background worker turns
reviewed experiences of one person into a few first-person views of being with
that person ("when she is tired she wants me to stay quiet and close"). They are
stored in the same per-user experience scope as `kind "relationship_view"`;
they are not facts about the user, and nothing is written for a turn without a
`user_id`.

- `relationship_every=1` runs the worker every N turns (0: off). Supply the
  `relationship` key in a background client mapping; a single shared client is
  used automatically.
- It reflects only when enough has happened: the importance of experiences and
  counted votes since the last reflection reaches
  `relationship_reflection_threshold` (default 1.0), or a view lost a source.
  The last reflection keeps a watermark of what it read, so the same experiences
  are not counted twice. Importance comes from the extractor (0..1, default 0.5;
  experiences stored before 1.6.0 count as 0.7).
- Every view cites the experience ids it rests on. It holds only while each of
  them is still its topic's current interpretation; if one is forgotten, voted
  out or revised, the view is withdrawn. A new batch supersedes the previous one.
- `relationship_views_shown=3` caps how many views one reflection keeps, and so
  how many reach her. In the
  transcript placement they are `- view: ` lines of the turn's note, and a
  withdrawn one is `- view withdrawn: `.
- `companion.relationship_views(user_id)` returns the views that still hold.

Extraction and votes see the older topics that share the user's words (the
extractor's `keywords`, else a topic's wording), so feedback on a topic she
already holds is found even when it is not among the newest four.
