# VRM visual acceptance host

This diagnostic host checks the optional VRM adapter with a locally supplied model. All renderer code remains in the external adapter package. The model is supplied locally and is not bundled.

1. Install the engine and renderer-vrm Python packages.
2. Run `python build_packets.py /absolute/path/model.vrm` to compile renderer-neutral engine cues through the actual VRM bridge.
3. Place or link that model as `model.vrm` in this directory.
4. Run `npm install`, `npm run build`, and `npm test` here.
5. Serve this directory on localhost and open `index.html`.

Use each expression, each signed head axis, all gaze directions, reset, and spring-bone motion. Positive/negative directions and left/right eyes still require a human observer. The diagnostic intentionally holds each cue until the next button press; it does not claim cue scheduling, smooth blending, or audio lip-sync acceptance. `Spring bone` drives head motion relative to the model's spring centers; translating only the whole root may produce no local spring rotation.

The packet adapter owns gaze angles after receiving an angle/reset packet (`autoUpdate=false`). A host that subsequently wants Object3D target tracking must explicitly restore `vrm.lookAt.autoUpdate=true`. Expression resets are isolated to mouth, face, and blink channels.

`adapter-regression.ts` checks legacy expression aliases, independent expression reset channels, and gaze state/reset. Real-model browser observations are separate from these unit checks. No UI button marks an item passed on behalf of a human.
