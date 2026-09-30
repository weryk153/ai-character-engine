from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai_character_engine.avatar import (
        AvatarBehaviorRuntime,
        AvatarRuntime,
        ExpressionRequest,
        GazeRequest,
        GestureRequest,
    )

from ai_character_engine.autonomy import (
    AdmissionResult,
    AutonomyController,
    AutonomyScheduler,
    DispatchStatus,
    ProactiveCandidate,
)
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.voice.base import AudioOutputSink, SpeechToTextProvider, TextToSpeechProvider
from ai_character_engine.voice.duplex import (
    DuplexVoiceConfig,
    DuplexVoiceError,
    LiveVoicePhase,
    supports_interruptible_playback,
)
from ai_character_engine.voice.models import AudioChunk, AudioFormat
from ai_character_engine.voice.text import SentenceSegmenter
from ai_character_engine.voice.vad import (
    EnergyVoiceActivityDetector,
    UtteranceSegmenter,
    UtteranceSegmenterConfig,
)
from ai_character_engine.vision.models import VisionFrame

from .models import (
    LiveEventType, LiveInputKind, LiveRuntimeConfig, LiveRuntimeEvent, LiveTurnInput,
    MouthActivityCue,
)


class LiveRuntimeError(RuntimeError):
    """Base error for the live orchestration boundary."""


class LiveRuntimeBackpressureError(LiveRuntimeError):
    """Raised when the bounded foreground input queue is full."""


@dataclass(slots=True)
class _ObservedFrame:
    observed_at: float
    frame: VisionFrame


class LiveCharacterOrchestrator:
    """Arbitrate foreground input, recent vision, autonomy and duplex voice.

    v0.25 keeps device ownership in the host. A microphone task may call
    :meth:`ingest_audio_chunk` concurrently while ``run()`` is generating or
    playing a reply. VAD can therefore stop a cooperative playback sink and/or
    cancel the current character generation without creating a second Runtime
    owner.

    Automatic acoustic barge-in is opt-in. It requires the host to explicitly
    confirm echo cancellation, and TTS playback additionally requires an async
    ``stop()`` hook on the sink. The default remains safe half-duplex behavior.
    """

    def __init__(
        self,
        bridge: CharacterHostBridge,
        *,
        scheduler: AutonomyScheduler | None = None,
        autonomy: AutonomyController | None = None,
        stt: SpeechToTextProvider | None = None,
        tts: TextToSpeechProvider | None = None,
        audio_sink: AudioOutputSink | None = None,
        avatar_runtime: AvatarRuntime | None = None,
        behavior_runtime: AvatarBehaviorRuntime | None = None,
        config: LiveRuntimeConfig | None = None,
        duplex: DuplexVoiceConfig | None = None,
        utterance_segmenter: UtteranceSegmenter | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bridge = bridge
        self.config = config or LiveRuntimeConfig()
        self._duplex_enabled = duplex is not None
        self.duplex = duplex or DuplexVoiceConfig()
        self.scheduler = scheduler or AutonomyScheduler()
        self.autonomy = autonomy or AutonomyController(bridge.runtime, self.scheduler)
        if self.autonomy.runtime is not bridge.runtime:
            raise ValueError("autonomy and bridge must share the same CharacterRuntime")
        if self.autonomy.scheduler is not self.scheduler:
            raise ValueError("autonomy must use the supplied scheduler")
        self.stt = stt
        self.tts = tts
        self.audio_sink = audio_sink
        self.avatar_runtime = avatar_runtime
        self.behavior_runtime = behavior_runtime
        if self.duplex.automatic_barge_in and self.tts is not None and self.audio_sink is not None:
            if not supports_interruptible_playback(self.audio_sink):
                raise DuplexVoiceError(
                    "automatic barge-in with speaker playback requires an async audio_sink.stop()"
                )
        self._utterance_segmenter = utterance_segmenter
        if self._utterance_segmenter is None and self.stt is not None:
            self._utterance_segmenter = UtteranceSegmenter(
                EnergyVoiceActivityDetector(rms_threshold=self.duplex.vad_rms_threshold),
                config=UtteranceSegmenterConfig(
                    min_speech_chunks=self.duplex.min_speech_chunks,
                    end_silence_chunks=self.duplex.end_silence_chunks,
                    max_chunks=self.duplex.max_utterance_chunks,
                ),
            )
        self._monotonic = monotonic
        self._inputs: asyncio.Queue[LiveTurnInput] = asyncio.Queue(self.config.max_pending_inputs)
        self._frames: deque[_ObservedFrame] = deque(maxlen=self.config.max_context_frames)
        self._closed = False
        self._runner_active = False

        self._active_input_id: str | None = None
        self._voice_phase = LiveVoicePhase.IDLE
        self._generation_interrupts: set[str] = set()
        self._barge_in_triggered = False
        self._playback_task: asyncio.Task[None] | None = None
        self._playback_interrupted = False
        self._played_segments: list[str] = []
        self._reply_text = ""
        self._active_user_text = ""
        self._stream_coordinator_task: asyncio.Task[None] | None = None
        self._tts_worker_task: asyncio.Task[None] | None = None
        self._interrupt_reasons: dict[str, str] = {}
        if self.behavior_runtime is not None and not self.behavior_runtime.active:
            self.behavior_runtime.start(now_ms=self._behavior_now_ms())

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_inputs(self) -> int:
        return self._inputs.qsize()

    @property
    def recent_frame_count(self) -> int:
        self._prune_frames()
        return len(self._frames)

    @property
    def voice_phase(self) -> LiveVoicePhase:
        return self._voice_phase

    @property
    def played_text(self) -> str:
        return "".join(self._played_segments)

    @property
    def playback_active(self) -> bool:
        return self._voice_phase is LiveVoicePhase.PLAYBACK and not self._playback_interrupted

    def _behavior_now_ms(self) -> float:
        return self._monotonic() * 1000.0

    def poll_behavior(self, *, force: bool = False, now_ms: float | None = None) -> tuple[LiveRuntimeEvent, ...]:
        """Poll the optional session-scoped avatar behavior runtime.

        Hosts that render at 30/60 FPS may call this independently of ``run``.
        The orchestrator also polls opportunistically on idle ticks, LLM deltas,
        and playback chunks so simple hosts do not need a second loop.
        """
        if self.behavior_runtime is None or self._closed:
            return ()
        now = self._behavior_now_ms() if now_ms is None else now_ms
        bundle = self.behavior_runtime.tick(now_ms=now, force=force)
        if bundle is None:
            return ()
        return (
            LiveRuntimeEvent(
                LiveEventType.AVATAR_BEHAVIOR_CUE,
                input_id=self._active_input_id,
                data=bundle.to_dict(),
            ),
        )

    def schedule_gaze(self, request: GazeRequest) -> str:
        if self.behavior_runtime is None:
            raise LiveRuntimeError("gaze scheduling requires an AvatarBehaviorRuntime")
        return self.behavior_runtime.schedule_gaze(request, now_ms=self._behavior_now_ms())

    def schedule_gesture(self, request: GestureRequest) -> str:
        if self.behavior_runtime is None:
            raise LiveRuntimeError("gesture scheduling requires an AvatarBehaviorRuntime")
        return self.behavior_runtime.schedule_gesture(request, now_ms=self._behavior_now_ms())

    def reset_behavior(self, *, reason: str = "host") -> LiveRuntimeEvent | None:
        """Stop and reset the optional session-scoped behavior timeline.

        A later :meth:`poll_behavior`, gaze request, gesture request, or turn can
        start it again. This is useful when a renderer is being unloaded or the
        avatar is temporarily hidden without closing the CharacterRuntime.
        """
        if self.behavior_runtime is None or not self.behavior_runtime.active:
            return None
        data = self.behavior_runtime.stop(now_ms=self._behavior_now_ms(), reason=reason)
        return LiveRuntimeEvent(
            LiveEventType.AVATAR_BEHAVIOR_RESET,
            input_id=self._active_input_id,
            data=data,
        )

    def _set_behavior_phase(self, phase: str, *, force: bool = True) -> tuple[LiveRuntimeEvent, ...]:
        if self.behavior_runtime is None:
            return ()
        now = self._behavior_now_ms()
        changed = self.behavior_runtime.phase.value != phase
        self.behavior_runtime.set_phase(phase, now_ms=now)
        return self.poll_behavior(force=force and changed, now_ms=now)

    def _begin_behavior_turn(self, turn_id: str) -> tuple[LiveRuntimeEvent, ...]:
        if self.behavior_runtime is None:
            return ()
        now = self._behavior_now_ms()
        self.behavior_runtime.begin_turn(turn_id, now_ms=now)
        return self.poll_behavior(force=True, now_ms=now)

    def _end_behavior_turn(self, *, reason: str) -> tuple[LiveRuntimeEvent, ...]:
        if self.behavior_runtime is None or self.behavior_runtime.turn_id is None:
            return ()
        now = self._behavior_now_ms()
        self.behavior_runtime.end_turn(now_ms=now, reason=reason)
        return self.poll_behavior(force=True, now_ms=now)

    def submit_text(self, text: str, *, frames: tuple[VisionFrame, ...] = ()) -> str:
        item = LiveTurnInput(LiveInputKind.TEXT, text=text, frames=tuple(frames))
        self._enqueue(item)
        return item.id

    def submit_audio(
        self,
        audio: bytes,
        *,
        audio_format: AudioFormat | None = None,
        frames: tuple[VisionFrame, ...] = (),
    ) -> str:
        if self.stt is None:
            raise LiveRuntimeError("audio input requires an STT provider")
        if len(audio) > self.config.max_audio_bytes:
            raise ValueError(f"audio exceeds max_audio_bytes={self.config.max_audio_bytes}")
        item = LiveTurnInput(
            LiveInputKind.AUDIO,
            audio=audio,
            audio_format=audio_format or AudioFormat(),
            frames=tuple(frames),
        )
        self._enqueue(item)
        return item.id

    async def ingest_audio_chunk(self, chunk: AudioChunk) -> tuple[LiveRuntimeEvent, ...]:
        """Feed one microphone chunk through VAD and queue completed utterances.

        This method is intended to run from a capture task independently of
        :meth:`run`. It never starts another character turn directly. A complete
        utterance becomes a normal bounded foreground audio item.
        """

        if self._closed:
            raise LiveRuntimeError("live runtime is closed")
        if self.stt is None or self._utterance_segmenter is None:
            raise LiveRuntimeError("continuous audio ingestion requires an STT provider")
        if (
            self._voice_phase is LiveVoicePhase.PLAYBACK
            and self.audio_sink is not None
            and not self.duplex.echo_cancellation_confirmed
        ):
            # Conservative half-duplex fallback: do not feed speaker echo into VAD/STT.
            # Push-to-talk hosts can call interrupt_output() first, then resume capture.
            self._utterance_segmenter.reset()
            return ()

        events: list[LiveRuntimeEvent] = []
        was_started = self._utterance_segmenter.started
        utterance = self._utterance_segmenter.push(chunk)
        started_now = not was_started and self._utterance_segmenter.started
        if started_now:
            self._barge_in_triggered = False
            events.append(
                LiveRuntimeEvent(
                    LiveEventType.SPEECH_STARTED,
                    data={"sequence": chunk.sequence},
                )
            )
            events.extend(self._set_behavior_phase("listening"))

        if (
            self.duplex.automatic_barge_in
            and not self._barge_in_triggered
            and self._voice_phase in (LiveVoicePhase.GENERATING, LiveVoicePhase.PLAYBACK)
            and self._utterance_segmenter.speech_chunks >= self.duplex.barge_in_min_speech_chunks
        ):
            self._barge_in_triggered = True
            phase = self._voice_phase
            events.append(
                LiveRuntimeEvent(
                    LiveEventType.BARGE_IN_DETECTED,
                    input_id=self._active_input_id,
                    data={"phase": phase.value, "speech_chunks": self._utterance_segmenter.speech_chunks},
                )
            )
            interrupted = await self.interrupt_output(reason="vad_barge_in")
            if interrupted is not None:
                events.append(interrupted)

        if utterance is not None:
            if any(part.format != utterance[0].format for part in utterance):
                raise ValueError("all chunks in an utterance must use the same AudioFormat")
            audio = b"".join(part.data for part in utterance)
            input_id = self.submit_audio(audio, audio_format=utterance[0].format)
            events.append(
                LiveRuntimeEvent(
                    LiveEventType.SPEECH_ENDED,
                    input_id=input_id,
                    data={"chunks": len(utterance), "audio_bytes": len(audio)},
                )
            )
            events.extend(self._set_behavior_phase("thinking"))
            self._barge_in_triggered = False
        return tuple(events)

    async def flush_audio_input(self) -> tuple[LiveRuntimeEvent, ...]:
        """Flush a trailing VAD utterance when a microphone source ends."""

        if self._utterance_segmenter is None:
            return ()
        utterance = self._utterance_segmenter.flush()
        if utterance is None:
            return ()
        if any(part.format != utterance[0].format for part in utterance):
            raise ValueError("all chunks in an utterance must use the same AudioFormat")
        audio = b"".join(part.data for part in utterance)
        input_id = self.submit_audio(audio, audio_format=utterance[0].format)
        self._barge_in_triggered = False
        events = [
            LiveRuntimeEvent(
                LiveEventType.SPEECH_ENDED,
                input_id=input_id,
                data={"chunks": len(utterance), "audio_bytes": len(audio), "flushed": True},
            )
        ]
        events.extend(self._set_behavior_phase("thinking"))
        return tuple(events)

    def observe_frame(self, frame: VisionFrame) -> None:
        if self._closed:
            raise LiveRuntimeError("live runtime is closed")
        self._prune_frames()
        self._frames.append(_ObservedFrame(self._monotonic(), frame))

    async def submit_visual_candidate(
        self,
        frame: VisionFrame,
        *,
        prompt: str | None = None,
        source: str = "live_vision",
        priority: int = 50,
        dedupe_key: str | None = None,
        cooldown_key: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AdmissionResult | None:
        if self._closed:
            raise LiveRuntimeError("live runtime is closed")
        if self.bridge.vision is None:
            raise LiveRuntimeError("visual candidates require a VisionPipeline")
        result = await self.bridge.vision.analyze_frame(frame, prompt=prompt)
        if result is None:
            return None
        candidate_payload = dict(payload or {})
        candidate_payload.setdefault("memory_importance", 0.0)
        candidate_payload["vision"] = result.event_payload["vision"]
        candidate = ProactiveCandidate(
            content=result.event_content,
            source=source,
            priority=priority,
            dedupe_key=dedupe_key,
            cooldown_key=cooldown_key,
            payload=candidate_payload,
        )
        return self.scheduler.submit(candidate)

    def submit_candidate(self, candidate: ProactiveCandidate) -> AdmissionResult:
        if self._closed:
            raise LiveRuntimeError("live runtime is closed")
        return self.scheduler.submit(candidate)

    def schedule_expression(self, request: ExpressionRequest) -> str:
        """Schedule a transient avatar expression during the active foreground turn."""
        if self.avatar_runtime is None:
            raise LiveRuntimeError("expression scheduling requires an AvatarRuntime")
        return self.avatar_runtime.schedule_expression(request)

    async def run_once(self) -> tuple[LiveRuntimeEvent, ...]:
        """Process one foreground item, otherwise make one autonomy attempt."""
        if self._closed:
            return ()
        try:
            item = self._inputs.get_nowait()
        except asyncio.QueueEmpty:
            return self.poll_behavior() + await self._run_autonomy_once()
        try:
            return tuple([event async for event in self._process_input_stream(item)])
        finally:
            self._inputs.task_done()

    async def run(self) -> AsyncIterator[LiveRuntimeEvent]:
        """Continuously yield meaningful host events until close() is called."""
        if self._runner_active:
            raise LiveRuntimeError("run() is already active")
        if self._closed:
            return
        self._runner_active = True
        try:
            while not self._closed:
                try:
                    item = await asyncio.wait_for(
                        self._inputs.get(), timeout=self.config.poll_interval_seconds
                    )
                except TimeoutError:
                    events = self.poll_behavior() + await self._run_autonomy_once()
                else:
                    try:
                        async for event in self._process_input_stream(item):
                            yield event
                    finally:
                        self._inputs.task_done()
                    continue
                for event in events:
                    yield event
        finally:
            self._runner_active = False

    def interrupt(self, heard_response: str = "") -> None:
        """Compatibility hook for history-only interruption.

        New duplex hosts should use :meth:`interrupt_output`, which also stops a
        cooperative playback sink and derives heard text from completed segments.
        """
        self.bridge.interrupt(heard_response)

    async def interrupt_output(self, *, reason: str = "host") -> LiveRuntimeEvent | None:
        """Interrupt active generation or playback without inventing heard text."""

        input_id = self._active_input_id
        if input_id is None or self._voice_phase is LiveVoicePhase.IDLE:
            return None

        if self.behavior_runtime is not None:
            self.behavior_runtime.interrupt_turn(now_ms=self._behavior_now_ms())

        if self._voice_phase is LiveVoicePhase.GENERATING:
            self._generation_interrupts.add(input_id)
            self._interrupt_reasons[input_id] = reason
            self.bridge.cancel()
            coordinator = self._stream_coordinator_task
            if not self.bridge.busy and coordinator is not None and not coordinator.done():
                coordinator.cancel()
            heard = self.played_text
            return LiveRuntimeEvent(
                LiveEventType.TURN_INTERRUPTED,
                input_id=input_id,
                text=heard or None,
                data={
                    "phase": LiveVoicePhase.GENERATING.value,
                    "reason": reason,
                    "played_text": heard,
                },
            )

        self._playback_interrupted = True
        self._interrupt_reasons[input_id] = reason
        stop = getattr(self.audio_sink, "stop", None)
        if callable(stop):
            await stop()
        task = self._playback_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        heard = self.played_text
        self.bridge.interrupt(heard)
        return LiveRuntimeEvent(
            LiveEventType.PLAYBACK_INTERRUPTED,
            input_id=input_id,
            text=heard,
            data={"phase": LiveVoicePhase.PLAYBACK.value, "reason": reason, "played_text": heard},
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.interrupt_output(reason="close")
        if self.behavior_runtime is not None and self.behavior_runtime.active:
            self.behavior_runtime.stop(now_ms=self._behavior_now_ms(), reason="close")
        await self.bridge.close()

    def _enqueue(self, item: LiveTurnInput) -> None:
        if self._closed:
            raise LiveRuntimeError("live runtime is closed")
        try:
            self._inputs.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise LiveRuntimeBackpressureError("foreground input queue is full") from exc

    def _prune_frames(self) -> None:
        threshold = self._monotonic() - self.config.context_frame_ttl_seconds
        while self._frames and self._frames[0].observed_at < threshold:
            self._frames.popleft()

    def _context_frames(self, explicit: tuple[VisionFrame, ...]) -> tuple[VisionFrame, ...]:
        self._prune_frames()
        combined: list[VisionFrame] = []
        seen: set[str] = set()
        for frame in (*explicit, *(entry.frame for entry in self._frames)):
            if frame.id in seen:
                continue
            combined.append(frame)
            seen.add(frame.id)
        limit = self.bridge.config.max_images
        return tuple(combined[-limit:])

    def _tts_segments(self, text: str) -> tuple[str, ...]:
        segmenter = SentenceSegmenter(
            min_chars=self.duplex.sentence_min_chars,
            max_chars=self.duplex.sentence_max_chars,
        )
        items = list(segmenter.push(text))
        tail = segmenter.flush()
        if tail:
            items.append(tail)
        return tuple(items)

    async def _process_input_stream(self, item: LiveTurnInput):
        """Yield one foreground turn, using v0.26 streaming only when opted in."""
        if not self.config.streaming_output:
            for event in await self._process_input(item):
                yield event
            return
        async for event in self._process_input_streaming(item):
            yield event

    @staticmethod
    def _audio_duration_ms(chunk: AudioChunk) -> float | None:
        fmt = chunk.format
        if fmt.encoding != "pcm_s16le":
            return None
        bytes_per_second = fmt.sample_rate_hz * fmt.channels * fmt.sample_width_bytes
        if bytes_per_second <= 0:
            return None
        return len(chunk.data) * 1000.0 / bytes_per_second

    async def _process_input_streaming(self, item: LiveTurnInput):
        """True LLM -> segmenter -> TTS -> playback streaming pipeline.

        The runtime remains transactional: generation commits state/history only
        after a final LLM response. Text/audio may be delivered earlier, so a
        cancellation records only fully played text as conversational continuity;
        state and long-term memory remain rolled back by CharacterRuntime.
        """
        self._active_input_id = item.id
        self._voice_phase = LiveVoicePhase.GENERATING
        self._playback_interrupted = False
        self._played_segments = []
        self._reply_text = ""
        self._active_user_text = ""
        self._stream_coordinator_task: asyncio.Task[None] | None = None
        self._tts_worker_task: asyncio.Task[None] | None = None
        self._interrupt_reasons: dict[str, str] = {}
        self._generation_interrupts.discard(item.id)

        yield LiveRuntimeEvent(
            LiveEventType.TURN_STARTED,
            input_id=item.id,
            data={"kind": item.kind.value, "streaming": True},
        )
        for behavior_event in self._begin_behavior_turn(item.id):
            yield behavior_event
        if self.avatar_runtime is not None:
            self.avatar_runtime.begin_turn(item.id)

        try:
            text = item.text
            if item.kind is LiveInputKind.AUDIO:
                assert self.stt is not None
                transcript = await asyncio.wait_for(
                    self.stt.transcribe(item.audio, audio_format=item.audio_format),
                    timeout=self.config.stt_timeout_seconds,
                )
                text = transcript.text.strip()
                if not text:
                    raise LiveRuntimeError("STT returned an empty transcript")
                yield LiveRuntimeEvent(
                    LiveEventType.STT_FINAL,
                    input_id=item.id,
                    text=text,
                    data={"language": transcript.language, "confidence": transcript.confidence},
                )
            self._active_user_text = text
            frames = self._context_frames(item.frames)

            event_queue: asyncio.Queue[Any] = asyncio.Queue(self.config.stream_queue_size * 4)
            tts_queue: asyncio.Queue[Any] = asyncio.Queue(self.config.stream_queue_size)
            done = object()
            tts_done = object()
            segmenter = SentenceSegmenter(
                min_chars=self.duplex.sentence_min_chars,
                max_chars=self.duplex.sentence_max_chars,
            )
            segment_sequence = 0
            audio_sequence = 0
            started = time.perf_counter()
            first_delta_ms: float | None = None
            first_audio_ms: float | None = None
            first_playback_ms: float | None = None
            tts_chunk_latencies: list[float] = []
            playback_started = False

            avatar_reset_sent = False

            async def emit(event: LiveRuntimeEvent) -> None:
                await event_queue.put(event)

            async def reset_avatar(reason: str) -> None:
                nonlocal avatar_reset_sent
                if avatar_reset_sent or self.avatar_runtime is None or not self.avatar_runtime.active:
                    return
                data = self.avatar_runtime.end_turn()
                data["reason"] = reason
                avatar_reset_sent = True
                await emit(
                    LiveRuntimeEvent(
                        LiveEventType.AVATAR_RESET,
                        input_id=item.id,
                        data=data,
                    )
                )

            async def on_delta(delta: str) -> None:
                nonlocal segment_sequence, first_delta_ms
                if not delta:
                    return
                if first_delta_ms is None:
                    first_delta_ms = (time.perf_counter() - started) * 1000
                await emit(
                    LiveRuntimeEvent(
                        LiveEventType.CHARACTER_DELTA,
                        input_id=item.id,
                        text=delta,
                        data={"elapsed_ms": (time.perf_counter() - started) * 1000},
                    )
                )
                for behavior_event in self.poll_behavior():
                    await emit(behavior_event)
                for sentence in segmenter.push(delta):
                    await tts_queue.put((segment_sequence, sentence))
                    segment_sequence += 1

            async def tts_worker() -> None:
                nonlocal audio_sequence, first_audio_ms, first_playback_ms, playback_started
                while True:
                    queued = await tts_queue.get()
                    try:
                        if queued is tts_done:
                            return
                        sequence, sentence = queued
                        if self._playback_interrupted or self.tts is None:
                            if self._playback_interrupted:
                                return
                            continue

                        provider_stream = getattr(self.tts, "synthesize_stream", None)
                        use_stream = self.config.prefer_streaming_tts and callable(provider_stream)
                        segment_completed = True
                        chunk_index = 0
                        segment_tts_started = time.perf_counter()

                        async def handle_chunk(chunk: AudioChunk, *, duration_ms: float | None, provider_latency_ms: float | None, streaming_tts: bool) -> bool:
                            nonlocal audio_sequence, first_audio_ms, first_playback_ms, playback_started
                            if self._playback_interrupted:
                                return False
                            ready_elapsed = (time.perf_counter() - started) * 1000
                            if first_audio_ms is None:
                                first_audio_ms = ready_elapsed
                            if provider_latency_ms is not None:
                                tts_chunk_latencies.append(provider_latency_ms)
                            for behavior_event in self._set_behavior_phase("speaking"):
                                await emit(behavior_event)
                            if self.audio_sink is not None and not playback_started:
                                playback_started = True
                                self._voice_phase = LiveVoicePhase.PLAYBACK
                                first_playback_ms = ready_elapsed
                                await emit(
                                    LiveRuntimeEvent(
                                        LiveEventType.PLAYBACK_STARTED,
                                        input_id=item.id,
                                        text=sentence,
                                        data={"streaming": True},
                                    )
                                )

                            cue_duration = duration_ms if duration_ms is not None else self._audio_duration_ms(chunk)
                            if self.config.emit_mouth_cues:
                                cue = MouthActivityCue(
                                    segment_sequence=sequence,
                                    chunk_index=chunk_index,
                                    duration_ms=cue_duration,
                                )
                                await emit(
                                    LiveRuntimeEvent(
                                        LiveEventType.MOUTH_CUE,
                                        input_id=item.id,
                                        text=sentence,
                                        data=cue.to_dict(),
                                    )
                                )

                            if self.avatar_runtime is not None:
                                bundle = self.avatar_runtime.feed_audio(
                                    text=sentence,
                                    chunk=chunk,
                                    segment_sequence=sequence,
                                    chunk_index=chunk_index,
                                    duration_ms=cue_duration,
                                    emotion=self.bridge.runtime.state.emotion,
                                    metadata=chunk.metadata,
                                    allow_text_visemes=not streaming_tts,
                                )
                                await emit(
                                    LiveRuntimeEvent(
                                        LiveEventType.AVATAR_CUE,
                                        input_id=item.id,
                                        text=sentence,
                                        data=bundle.to_dict(),
                                    )
                                )
                            for behavior_event in self.poll_behavior():
                                await emit(behavior_event)

                            played = False
                            if self.audio_sink is not None:
                                self._voice_phase = LiveVoicePhase.PLAYBACK
                                self._playback_task = asyncio.create_task(self.audio_sink.play(chunk))
                                try:
                                    await self._playback_task
                                except asyncio.CancelledError:
                                    if self._playback_interrupted:
                                        return False
                                    raise
                                finally:
                                    self._playback_task = None
                                played = not self._playback_interrupted
                            await emit(
                                LiveRuntimeEvent(
                                    LiveEventType.TTS_AUDIO,
                                    input_id=item.id,
                                    text=sentence,
                                    audio=AudioChunk(
                                        chunk.data,
                                        format=chunk.format,
                                        sequence=audio_sequence,
                                        timestamp_ms=chunk.timestamp_ms,
                                        metadata=dict(chunk.metadata),
                                    ),
                                    data={
                                        "segment_sequence": sequence,
                                        "chunk_index": chunk_index,
                                        "sequence": audio_sequence,
                                        "duration_ms": cue_duration,
                                        "latency_ms": provider_latency_ms,
                                        "streaming_tts": streaming_tts,
                                        "played": played,
                                    },
                                )
                            )
                            audio_sequence += 1
                            return not self._playback_interrupted

                        if use_stream:
                            first_chunk = True
                            async for raw in provider_stream(sentence, voice=self.config.tts_voice):
                                if not isinstance(raw, AudioChunk):
                                    raise LiveRuntimeError("streaming TTS must yield AudioChunk values")
                                latency = ((time.perf_counter() - segment_tts_started) * 1000) if first_chunk else None
                                first_chunk = False
                                ok = await handle_chunk(
                                    raw,
                                    duration_ms=self._audio_duration_ms(raw),
                                    provider_latency_ms=latency,
                                    streaming_tts=True,
                                )
                                chunk_index += 1
                                if not ok:
                                    segment_completed = False
                                    break
                        else:
                            synthesized = await asyncio.wait_for(
                                self.tts.synthesize(sentence, voice=self.config.tts_voice),
                                timeout=self.config.tts_timeout_seconds,
                            )
                            latency = (time.perf_counter() - segment_tts_started) * 1000
                            chunk = AudioChunk(
                                synthesized.data,
                                format=synthesized.format,
                                metadata=dict(synthesized.metadata),
                            )
                            segment_completed = await handle_chunk(
                                chunk,
                                duration_ms=synthesized.duration_ms,
                                provider_latency_ms=latency,
                                streaming_tts=False,
                            )
                            chunk_index += 1

                        if segment_completed and self.audio_sink is not None and not self._playback_interrupted:
                            self._played_segments.append(sentence)
                        if self.bridge.busy and not self._playback_interrupted:
                            self._voice_phase = LiveVoicePhase.GENERATING
                        if self._playback_interrupted:
                            return
                    finally:
                        tts_queue.task_done()

            async def coordinator() -> None:
                worker = asyncio.create_task(tts_worker())
                self._tts_worker_task = worker
                interrupted = False
                generation_committed = False
                try:
                    result = await self.bridge.process(
                        text,
                        frames=frames,
                        on_text_delta=on_delta,
                    )
                    generation_committed = True
                    self._reply_text = result.text
                    await emit(
                        LiveRuntimeEvent(
                            LiveEventType.REPLY,
                            input_id=item.id,
                            text=result.text,
                            run_result=result,
                            data={"frames": len(frames), "streaming": True},
                        )
                    )
                    tail = segmenter.flush()
                    if tail:
                        await tts_queue.put((segment_sequence, tail))
                    await tts_queue.put(tts_done)
                    await worker
                    if self.audio_sink is not None and playback_started and not self._playback_interrupted:
                        await emit(
                            LiveRuntimeEvent(
                                LiveEventType.PLAYBACK_FINISHED,
                                input_id=item.id,
                                text=self.played_text,
                                data={
                                    "played_text": self.played_text,
                                    "segments": len(self._played_segments),
                                    "streaming": True,
                                },
                            )
                        )
                    await reset_avatar("completed")
                    for behavior_event in self._end_behavior_turn(reason="completed"):
                        await emit(behavior_event)
                except asyncio.CancelledError:
                    interrupted = True
                    if not worker.done():
                        worker.cancel()
                        try:
                            await worker
                        except asyncio.CancelledError:
                            pass
                    heard = self.played_text
                    if not self.bridge.busy:
                        if generation_committed:
                            # Even zero fully played segments must remove the
                            # committed but unheard answer from history.
                            self.bridge.interrupt(heard)
                        elif heard:
                            self.bridge.record_interrupted_turn(text, heard)
                    if item.id in self._generation_interrupts:
                        self._generation_interrupts.discard(item.id)
                        reason = self._interrupt_reasons.pop(item.id, "barge_in")
                        await emit(
                            LiveRuntimeEvent(
                                LiveEventType.TURN_INTERRUPTED,
                                input_id=item.id,
                                text=heard or None,
                                data={
                                    "phase": LiveVoicePhase.GENERATING.value,
                                    "reason": reason,
                                    "played_text": heard,
                                },
                            )
                        )
                    await reset_avatar("interrupted")
                    for behavior_event in self._end_behavior_turn(reason="interrupted"):
                        await emit(behavior_event)
                except Exception as exc:
                    if not worker.done():
                        worker.cancel()
                        try:
                            await worker
                        except asyncio.CancelledError:
                            pass
                    await reset_avatar("error")
                    for behavior_event in self._end_behavior_turn(reason="error"):
                        await emit(behavior_event)
                    await emit(
                        LiveRuntimeEvent(
                            LiveEventType.ERROR,
                            input_id=item.id,
                            data={
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                                "requires_review": bool(getattr(exc, "requires_review", False)),
                            },
                        )
                    )
                finally:
                    if self.avatar_runtime is not None and self.avatar_runtime.active and not avatar_reset_sent:
                        await reset_avatar("finalize")
                    if self.behavior_runtime is not None and self.behavior_runtime.turn_id is not None:
                        for behavior_event in self._end_behavior_turn(reason="finalize"):
                            await emit(behavior_event)
                    total_ms = (time.perf_counter() - started) * 1000
                    await emit(
                        LiveRuntimeEvent(
                            LiveEventType.STREAM_METRICS,
                            input_id=item.id,
                            data={
                                "llm_ttft_ms": first_delta_ms,
                                "ttfa_ms": first_audio_ms,
                                "first_playback_ms": first_playback_ms,
                                "total_ms": total_ms,
                                "audio_chunks": audio_sequence,
                                "played_segments": len(self._played_segments),
                                "tts_chunk_latency_ms": tuple(tts_chunk_latencies),
                                "interrupted": interrupted or self._playback_interrupted,
                            },
                        )
                    )
                    self._tts_worker_task = None
                    await event_queue.put(done)

            producer = asyncio.create_task(coordinator())
            self._stream_coordinator_task = producer
            try:
                while True:
                    event = await event_queue.get()
                    try:
                        if event is done:
                            break
                        yield event
                    finally:
                        event_queue.task_done()
                await producer
            finally:
                if not producer.done():
                    producer.cancel()
                    try:
                        await producer
                    except asyncio.CancelledError:
                        pass
                if self._stream_coordinator_task is producer:
                    self._stream_coordinator_task = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield LiveRuntimeEvent(
                LiveEventType.ERROR,
                input_id=item.id,
                data={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "requires_review": bool(getattr(exc, "requires_review", False)),
                },
            )
            if self.avatar_runtime is not None and self.avatar_runtime.active:
                data = self.avatar_runtime.end_turn()
                data["reason"] = "error"
                yield LiveRuntimeEvent(
                    LiveEventType.AVATAR_RESET,
                    input_id=item.id,
                    data=data,
                )
            for behavior_event in self._end_behavior_turn(reason="error"):
                yield behavior_event
        finally:
            if self._active_input_id == item.id:
                self._active_input_id = None
                self._voice_phase = LiveVoicePhase.IDLE
                self._playback_task = None
                self._reply_text = ""
                self._active_user_text = ""
                self._stream_coordinator_task = None
                self._tts_worker_task = None
                self._interrupt_reasons.pop(item.id, None)
            if self.avatar_runtime is not None and self.avatar_runtime.active:
                self.avatar_runtime.end_turn()

    async def _process_input(self, item: LiveTurnInput) -> tuple[LiveRuntimeEvent, ...]:
        events: list[LiveRuntimeEvent] = [
            LiveRuntimeEvent(LiveEventType.TURN_STARTED, input_id=item.id, data={"kind": item.kind.value})
        ]
        self._active_input_id = item.id
        self._voice_phase = LiveVoicePhase.GENERATING
        self._playback_interrupted = False
        self._played_segments = []
        self._reply_text = ""
        events.extend(self._begin_behavior_turn(item.id))
        if self.avatar_runtime is not None:
            self.avatar_runtime.begin_turn(item.id)

        def append_avatar_reset(reason: str) -> None:
            if self.avatar_runtime is None or not self.avatar_runtime.active:
                return
            data = self.avatar_runtime.end_turn()
            data["reason"] = reason
            events.append(
                LiveRuntimeEvent(
                    LiveEventType.AVATAR_RESET,
                    input_id=item.id,
                    data=data,
                )
            )

        try:
            text = item.text
            if item.kind is LiveInputKind.AUDIO:
                assert self.stt is not None
                transcript = await asyncio.wait_for(
                    self.stt.transcribe(item.audio, audio_format=item.audio_format),
                    timeout=self.config.stt_timeout_seconds,
                )
                text = transcript.text.strip()
                if not text:
                    raise LiveRuntimeError("STT returned an empty transcript")
                events.append(
                    LiveRuntimeEvent(
                        LiveEventType.STT_FINAL,
                        input_id=item.id,
                        text=text,
                        data={"language": transcript.language, "confidence": transcript.confidence},
                    )
                )

            frames = self._context_frames(item.frames)
            result = await self.bridge.process(text, frames=frames)
            self._reply_text = result.text
            events.append(
                LiveRuntimeEvent(
                    LiveEventType.REPLY,
                    input_id=item.id,
                    text=result.text,
                    run_result=result,
                    data={"frames": len(frames)},
                )
            )
            if self.tts is not None and result.text.strip():
                segments = self._tts_segments(result.text) if self._duplex_enabled else (result.text,)
                events.extend(self._set_behavior_phase("speaking"))
                if self.audio_sink is not None:
                    self._voice_phase = LiveVoicePhase.PLAYBACK
                    if self._duplex_enabled:
                        events.append(
                            LiveRuntimeEvent(
                                LiveEventType.PLAYBACK_STARTED,
                                input_id=item.id,
                                text=result.text,
                                data={"segments": len(segments)},
                            )
                        )
                for sequence, sentence in enumerate(segments):
                    if self._playback_interrupted:
                        break
                    synthesized = await asyncio.wait_for(
                        self.tts.synthesize(sentence, voice=self.config.tts_voice),
                        timeout=self.config.tts_timeout_seconds,
                    )
                    chunk = AudioChunk(
                        synthesized.data,
                        format=synthesized.format,
                        sequence=sequence,
                        metadata=dict(synthesized.metadata),
                    )
                    if self.avatar_runtime is not None:
                        emotion = result.state_after.emotion if result.state_after is not None else self.bridge.runtime.state.emotion
                        bundle = self.avatar_runtime.feed_audio(
                            text=sentence,
                            chunk=chunk,
                            segment_sequence=sequence,
                            chunk_index=0,
                            duration_ms=synthesized.duration_ms,
                            emotion=emotion,
                            metadata=chunk.metadata,
                            allow_text_visemes=True,
                        )
                        events.append(
                            LiveRuntimeEvent(
                                LiveEventType.AVATAR_CUE,
                                input_id=item.id,
                                text=sentence,
                                data=bundle.to_dict(),
                            )
                        )
                    events.extend(self.poll_behavior())
                    played = False
                    if self.audio_sink is not None:
                        self._playback_task = asyncio.create_task(self.audio_sink.play(chunk))
                        try:
                            await self._playback_task
                        except asyncio.CancelledError:
                            if self._playback_interrupted:
                                break
                            raise
                        finally:
                            self._playback_task = None
                        if not self._playback_interrupted:
                            self._played_segments.append(sentence)
                            played = True
                    events.append(
                        LiveRuntimeEvent(
                            LiveEventType.TTS_AUDIO,
                            input_id=item.id,
                            text=sentence,
                            audio=chunk,
                            data={
                                "duration_ms": synthesized.duration_ms,
                                "latency_ms": synthesized.latency_ms,
                                "sequence": sequence,
                                "played": played,
                            },
                        )
                    )
                if self.audio_sink is not None and not self._playback_interrupted and self._duplex_enabled:
                    events.append(
                        LiveRuntimeEvent(
                            LiveEventType.PLAYBACK_FINISHED,
                            input_id=item.id,
                            text=self.played_text,
                            data={"played_text": self.played_text, "segments": len(self._played_segments)},
                        )
                    )
            append_avatar_reset("interrupted" if self._playback_interrupted else "completed")
            events.extend(
                self._end_behavior_turn(
                    reason="interrupted" if self._playback_interrupted else "completed"
                )
            )
            return tuple(events)
        except asyncio.CancelledError:
            if item.id in self._generation_interrupts:
                self._generation_interrupts.discard(item.id)
                if not any(event.type is LiveEventType.TURN_INTERRUPTED for event in events):
                    events.append(
                        LiveRuntimeEvent(
                            LiveEventType.TURN_INTERRUPTED,
                            input_id=item.id,
                            data={"phase": LiveVoicePhase.GENERATING.value, "reason": "barge_in", "played_text": ""},
                        )
                    )
                append_avatar_reset("interrupted")
                events.extend(self._end_behavior_turn(reason="interrupted"))
                return tuple(events)
            raise
        except Exception as exc:
            events.append(
                LiveRuntimeEvent(
                    LiveEventType.ERROR,
                    input_id=item.id,
                    data={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "requires_review": bool(getattr(exc, "requires_review", False)),
                    },
                )
            )
            append_avatar_reset("error")
            events.extend(self._end_behavior_turn(reason="error"))
            return tuple(events)
        finally:
            if self._active_input_id == item.id:
                self._active_input_id = None
                self._voice_phase = LiveVoicePhase.IDLE
                self._playback_task = None
                self._reply_text = ""
            if self.avatar_runtime is not None and self.avatar_runtime.active:
                self.avatar_runtime.end_turn()
            if self.behavior_runtime is not None and self.behavior_runtime.turn_id is not None:
                self.behavior_runtime.end_turn(now_ms=self._behavior_now_ms(), reason="finalize")

    async def _run_autonomy_once(self) -> tuple[LiveRuntimeEvent, ...]:
        result = await self.autonomy.run_once()
        if result.status in (DispatchStatus.EMPTY, DispatchStatus.BLOCKED):
            return ()
        if result.status is DispatchStatus.DELIVERED:
            run_result = result.run_result
            text = getattr(run_result, "text", None)
            return (
                LiveRuntimeEvent(
                    LiveEventType.AUTONOMY_DELIVERED,
                    text=text,
                    run_result=run_result,
                    data={"candidate_id": result.candidate.id if result.candidate else None},
                ),
            )
        return (
            LiveRuntimeEvent(
                LiveEventType.AUTONOMY_FAILED,
                data={
                    "candidate_id": result.candidate.id if result.candidate else None,
                    "reason": result.reason,
                    "requires_review": result.requires_review,
                },
            ),
        )
