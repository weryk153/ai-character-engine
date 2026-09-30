# Extensions and host integrations

## Provider clients

The runtime consumes the asynchronous `LLMClient` protocol. Use a supplied adapter
or implement `generate(messages, *, tools=None)` returning `LLMResponse`.
Streaming clients may additionally provide `stream_generate`. Keep authentication,
model selection and endpoint behavior at the adapter/host boundary.

## Plugin contracts

`PluginManifest`, `PluginActivationPolicy`, `ExtensionRegistry` and `PluginManager`
provide explicit discovery, loading and activation. Supported extension points
include model clients, embedding/vision providers, observability sinks,
world-perception policies, cognitive specialists and tools.

Plugin **activation is not authority**. Factories receive the bounded extension
context, not unrestricted access to character managers or commit coordinators.
Tools require a separate host installation step after activation. There is no
automatic plugin import or activation at engine startup.

A Python plugin is trusted process code: this API is **not a security sandbox**.
Choose and review plugins at the host boundary. Capabilities describe allowed
engine extension surfaces; they do not isolate arbitrary Python code.

## HTTP host

Install the `service` extra and construct `CharacterService` and `create_app` with
your runtime factory. The [service example](../examples/character_service.py) uses
an offline client to exercise the transport without an inference service:

```sh
python -m pip install '.[service]'
python -m uvicorn examples.character_service:app --host 127.0.0.1 --port 8000
```

Endpoints include health/readiness, session creation and retrieval, turn messages,
SSE and WebSocket transport. Open `/docs` on the running service for its generated
request/response schema. `create_app` defaults to `NoopAuthHook`; an exposed
service must supply its own authentication and tenant/session admission policy.
CORS configuration is not authentication.

## Renderer package

The core renderer contract remains independent of any renderer SDK. Install
`packages/renderer-vrm` when the host needs VRM command mapping. Its README
specifies the adapter boundary and asset requirements. It shares the engine
version and ships its own Apache-2.0 license.

## Audio and training hosts

Audio input/output, STT/TTS endpoints, model files and training processes are
external resources selected by the host. See the [examples index](../examples/README.md)
for optional entry points and requirements. Core portability evidence does not
imply microphone, GPU, VRM asset or voice-quality acceptance.
