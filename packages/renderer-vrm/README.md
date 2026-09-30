# AI Character Engine VRM Adapter

Optional VRM command mapping for AI Character Engine 1.0.0. The host owns the
renderer, loaded assets and frame lifecycle; this package translates neutral
engine output without becoming a character-state authority.

Install from a checkout with `python -m pip install ./packages/renderer-vrm`, or
install its matching wheel alongside the 1.0.0 core wheel. Import the package as
`ai_character_engine_vrm`.

VRM/VRMA assets are supplied separately. The package does not grant asset rights
or guarantee a particular model's visual calibration. Adapter fixtures contain
technical metadata used by tests; hardware/visual acceptance remains host-owned.

License: [Apache-2.0](LICENSE).
