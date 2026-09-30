# Compatibility policy

## Stable Python API

The 374 names exported through `ai_character_engine.__all__` are recorded in
[public_api_v1_stable.json](public_api_v1_stable.json). The stable contract covers
those root exports and their recorded call shapes. Importable implementation
modules are not automatically part of that freeze.

The root contract was frozen in 0.48.0 and is marked `v1-stable` for 1.0.0. Sealed
manifests are retained as regression fixtures. A release must preserve their
compatible surface or explicitly introduce a new major contract and migration
policy; editing a baseline is not an acceptable compatibility repair.

## Independent contracts

| Contract | Version |
|---|---|
| Public Python API | 1 |
| Extension API | 1 |
| Distributed protocol | 1 |
| Persistence schema | 1 |

Package SemVer does not automatically change these contracts. Core and the two
optional packages ship synchronized package versions.

## Deprecation

Deprecation identifies its introduction version, the replacement when available
and a declared removal window. Stable symbols require a future major-version
policy for incompatible removal. Compatibility tooling detects signature and
symbol changes before a distribution is accepted.

## Persistence

JSON/JSONL stores use explicit `_schema_version` metadata and deterministic
forward-only migration. Legacy unversioned records are schema 0. Unknown future
schemas and downgrades fail closed. Loading old bytes migrates data in memory
without rewriting the source file; saving is an explicit operation.

The release gate replays eleven sealed persistence fixtures. Application hosts
must additionally validate their own data and operational backup/restore path.
