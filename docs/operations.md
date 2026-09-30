# Operations

## Persistence and lifecycle

Use stable session identifiers and explicitly chosen store paths. JSON session,
relationship, event, memory and cognitive stores are local persistence components;
a host must coordinate writers, backups, retention and access to their files.
In-memory stores and temporary sample paths are suitable for isolated execution,
not durable multi-process storage.

Loading an older persistence schema migrates in memory. Rewriting stored bytes is
an explicit action. Back up data before an intentional save or migration, and do
not downgrade records from an unknown future schema. See [compatibility](compatibility.md).

During shutdown, stop admitting new work, cancel or drain host tasks according
to policy, and close/flush owned sessions and resources. Preserve uncertain tool
outcomes for review rather than assuming a timeout means no side effect occurred.

## Background and distributed workers

**Remote execution is not remote authority.** A `DistributedTaskEnvelope` contains
serializable work and an immutable snapshot. Workers return completion data;
coordinators retain authoritative acceptance and commit decisions.

**At-least-once execution** requires explicit idempotency and completion handling.
Lease expiry, fencing and attempt identity prevent stale workers from replacing
newer accepted results. Unsafe or unknown side effects require review before
retry. The in-memory broker implements the contract for one process; a deployment
must supply a durable broker adapter for cross-process persistence.

## Reliability and observability

Configure timeouts, bounded retries, backoff, circuit breakers and admission limits
for the environment. Keep model-call failure, tool failure and commit uncertainty
distinct. Correlate requests and traces at the host boundary; retain operational
metadata according to the application's data policy.

The service provides `/health` and `/ready`. These report host/service state;
they do not prove model response quality or the availability of every optional
provider. Configure authentication before exposing the service beyond local use.

## Performance and validation

Run the acceptance collectors described in [release validation](release.md).
Compare performance only within the same environment profile. Results are not a
universal hardware SLA. Soak/failure injection is bounded deterministic evidence,
not a substitute for measuring the production workload.

Each release retains source identity, command receipts and artifact hashes.
Missing evidence blocks release; failures remain failures until fixed and rerun.
