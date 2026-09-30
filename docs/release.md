# Release and validation

## Candidate identity

Release evaluation is read-only. It consumes executable evidence; it does not
repair runtime state, infer missing platforms or publish artifacts.

A candidate fingerprint covers sorted relative source paths and exact bytes,
including documentation, tests, workflows and LICENSE. Generated caches, build
outputs and Git metadata are excluded. Keep evidence and distributions outside
the source tree. Any source change invalidates earlier candidate evidence.
Windows checkout must preserve LF bytes (`core.autocrlf=false`).

## Supported matrix

All **9** Linux/macOS/Windows × Python 3.11/3.12/3.13 rows must execute:

1. full pytest;
2. compileall;
3. release matrix self-check.

The collector records the actual interpreter/platform, engine version, source
fingerprint, command exit codes and output digests. Missing rows are BLOCKED;
failed, mismatched or tampered records are FAIL.

```sh
python -m ai_character_engine.release_candidate evidence --project-root . --output /outside/matrix/local.json
python tools/final_acceptance.py --output /outside/final
```

Use real outside-source paths appropriate to the host. Install build tooling and
populate the locked development environment before running final acceptance;
see [contributing](../CONTRIBUTING.md).

## Final acceptance

The collector runs soak/failure injection, same-profile performance, architecture
checks, offline lock validation, four package builds and clean wheel/sdist installs.
Packaging checks verify Apache-2.0 metadata and exact LICENSE bytes. Fresh-install
imports run outside the source tree in isolated environments and verify package
versions and import locations.

Aggregate the nine matrix records, final receipts and all four artifacts:

```sh
python -m ai_character_engine.release_candidate check --project-root . --evidence-dir /outside/matrix --acceptance-evidence-dir /outside/final/acceptance --artifact /outside/final/distributions/core/ai_character_engine-1.0.0-py3-none-any.whl --artifact /outside/final/distributions/core/ai_character_engine-1.0.0.tar.gz --artifact /outside/final/distributions/vrm/ai_character_engine_vrm-1.0.0-py3-none-any.whl --artifact /outside/final/distributions/vrm/ai_character_engine_vrm-1.0.0.tar.gz --output /outside/report.json --fail-on-blocker
```

Exit zero and `ready_for_v1=true` require every gate to pass. API checks compare
374 stable symbols with sealed contracts and replay eleven persistence fixtures.
The report binds packaging and fresh-install evidence to exact distribution hashes.

## CI and artifacts

CI uploads records from the actual matrix and acceptance jobs. The release
workflow aggregates artifacts from the same workflow run without rebuilding
distributions after acceptance. A failed producer job blocks aggregation.

Deliver a source ZIP, four-distribution bundle, SHA-256 checksums and machine-readable
report. Keep build logs and receipts with the artifact delivery, rather than
committing stale PASS reports into the source candidate. Publication and registry
credentials are separate owner-controlled actions; these workflows do not publish.
