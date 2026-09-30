# Contributing

## Development environment

Use Python 3.11–3.13 in a virtual environment:

```sh
python -m pip install -e '.[dev,service]' build
python -m pytest -q
```

The `service` extra is needed by the full test suite. With uv, use
`uv sync --locked --extra dev --extra service`; retain `uv.lock` consistency.

Tests use offline providers unless an integration test explicitly requires an
external service. Component scenarios live under `tests/scenarios`; maintained
host entry points are indexed under `examples/README.md`.

## Change boundaries

Preserve the stable root API, independent contract versions and synchronized
package versions. Hosts, providers, plugins and renderers must not acquire core
state authority. Do not rewrite sealed manifests or migration fixtures to hide
incompatibility. See [architecture](docs/architecture.md) and [compatibility](docs/compatibility.md).

## Documentation

Product documentation describes current behavior, configuration and operations.
Keep historical development narrative in Git history. Regenerate the root API
reference with `python tools/generate_api_reference.py` when its source descriptions
change. Documentation examples and local links are checked by the test suite.

## Release checks

Follow [release validation](docs/release.md). Run full pytest, compilation,
architecture, replay, soak/performance, packaging, fresh installation and all nine
matrix rows. A documentation or packaging change still changes the candidate
fingerprint and requires fresh evidence. Never infer PASS from a missing row.
