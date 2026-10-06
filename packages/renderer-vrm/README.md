# AI Character Engine VRM Adapter

Optional VRM command mapping for AI Character Engine 1.2.0. The host owns the
renderer, loaded assets and frame lifecycle; this package translates neutral
engine output without becoming a character-state authority.

Install from a checkout with `python -m pip install ./packages/renderer-vrm`, or
install its matching wheel alongside the 1.2.0 core wheel. Import the package as
`ai_character_engine_vrm`.

VRM/VRMA assets are supplied separately. The package does not grant asset rights
or guarantee a particular model's visual calibration; tests use synthetic models
only, and visual acceptance remains host-owned. `examples/vrm_calibration.py`
inspects your own model; `examples/vrm_renderer_bridge.py` compiles cues for it.

License: [Apache-2.0](LICENSE).
