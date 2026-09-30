# Validation

The current candidate is verified by the [CI workflow](https://github.com/weryk153/ai-character-engine/actions/workflows/ci.yml)
and the strict release report delivered with its artifacts. Test counts and PASS
receipts are not copied into this source file: modifying the source creates a
new candidate identity and makes old evidence inapplicable.

## Required checks

- Linux, macOS and Windows × Python 3.11, 3.12 and 3.13: full pytest, compileall and matrix self-check.
- Stable root API exact match and sealed-version compatibility.
- Eleven sealed persistence fixtures replayed deterministically.
- Architecture neutrality and authority boundaries.
- Soak/failure injection and performance comparisons within the same environment.
- Four wheel/sdist artifacts, synchronized versions and Apache-2.0 license inspection.
- Clean wheel and sdist installation, with isolated imports outside the source tree.
- `uv lock --check --offline`.
- Product documentation links, runnable installation example and complete root API reference.

Every receipt must match the source fingerprint and engine version. Artifact
acceptance must match the precise distribution hashes. Missing evidence blocks
release; any failed check fails it. See [release instructions](docs/release.md).

Hardware-specific audio, GPU, renderer assets, live model quality and production
capacity require acceptance in the application's own environment. The core matrix
does not make those claims.
