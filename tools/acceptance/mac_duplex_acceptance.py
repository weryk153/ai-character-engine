"""Opt-in real microphone duplex diagnostic; never fabricates human acceptance.

Requires headphones/acoustic isolation confirmed interactively. Audio stays in
memory. macOS say is sentence-buffered TTS, not a native streaming TTS provider.
"""
from __future__ import annotations
import argparse, asyncio, json, time
from contextlib import aclosing
from dataclasses import asdict
from pathlib import Path
from ai_character_engine.voice.devices import SoundDeviceInput, SoundDeviceOutput
from ai_character_engine.voice.mac_providers import MacOSSayTTS, SherpaOnnxSTT
from ai_character_engine.voice.live import make_session
from ai_character_engine.voice.vad import EnergyVoiceActivityDetector
from ai_character_engine.llm.voice_stream import VoiceStreamingChatClient
from ai_character_engine.host import CharacterHostBridge
from ai_character_engine.live import LiveCharacterOrchestrator, LiveRuntimeConfig, DuplexVoiceConfig
from ai_character_engine.memory.manager import MemoryManager

async def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    provider = VoiceStreamingChatClient(model=args.model, base_url=args.base_url,
        backend='lmstudio', max_tokens=1024, extra_body={'reasoning_effort':'none'}, require_time_tool=True)
    session = make_session(provider)
    session.runtime.memory_manager = MemoryManager()
    memory = session.runtime.memory_manager
    stt = SherpaOnnxSTT(args.stt_model, language='zh', local_files_only=True)
    sink = SoundDeviceOutput(args.output_device)
    live = LiveCharacterOrchestrator(CharacterHostBridge(session.runtime), stt=stt,
        tts=MacOSSayTTS(), audio_sink=sink,
        config=LiveRuntimeConfig(streaming_output=True, tts_voice='Meijia'),
        duplex=DuplexVoiceConfig(automatic_barge_in=True, echo_cancellation_confirmed=True,
            vad_rms_threshold=args.vad_threshold, min_speech_chunks=5, end_silence_chunks=30,
            barge_in_min_speech_chunks=5, max_utterance_chunks=1000))
    detector=EnergyVoiceActivityDetector(rms_threshold=args.vad_threshold)
    starts={}; metrics=[]; event_counts={}; last_speech=None; errors=[]
    begun=time.perf_counter()
    def log(event):
        now=time.perf_counter();kind=event.type.value
        event_counts[kind]=event_counts.get(kind,0)+1
        if kind=='speech_ended': starts[event.input_id]={'speech_end':last_speech,'endpoint':now}
        timing=starts.get(event.input_id,{})
        if kind=='stt_final': timing['stt_result']=now;print('You:',event.text,flush=True)
        if kind=='reply':print('Character:',event.text,flush=True)
        if kind=='error':errors.append(event.data)
        if kind in ('barge_in_detected','playback_interrupted','turn_interrupted'):print(kind,event.data,flush=True)
        if kind=='stream_metrics':
            row={'input_id':event.input_id,**event.data,'speech_end_to_stt_ms':(timing['stt_result']-timing['speech_end'])*1000 if timing.get('speech_end') and timing.get('stt_result') else None,
                'complete_response_ms':(now-timing['speech_end'])*1000 if timing.get('speech_end') else None,
                'first_playback_metric_kind':'orchestrator playback request; not acoustic onset',
                'streaming_tts':False,'tts_mode':'macOS sentence buffered fallback'}
            metrics.append(row)
            with (args.out/'duplex_latency.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        row={'at_s':now-begun,'type':kind,'input_id':event.input_id,'text':event.text,'data':event.data,
             'history_messages':len(session.runtime.history),'state':asdict(session.runtime.state),
             'memory_records':len(memory.store.list_for_character(session.runtime.memory_scope_id)),
             'ledger_entries':len(memory.ledger.list_for_character(session.runtime.memory_scope_id)),
             'pending_inputs':live.pending_inputs,'tasks':len(asyncio.all_tasks())}
        with (args.out/'duplex_events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    async def capture():
        nonlocal last_speech
        async with aclosing(SoundDeviceInput(args.input_device).chunks()) as chunks:
            async for chunk in chunks:
                if detector.is_speech(chunk):last_speech=chunk.timestamp_ms/1000
                for event in await live.ingest_audio_chunk(chunk):log(event)
    async def consume():
        async with aclosing(live.run()) as events:
            async for event in events:log(event)
    tasks=[]
    try:
        await stt.start()
        print('Listening: talk normally first, then interrupt while the AI is speaking. Ctrl-C to stop.',flush=True)
        tasks=[asyncio.create_task(capture()),asyncio.create_task(consume())]
        done,_=await asyncio.wait(tasks,timeout=args.duration,return_when=asyncio.FIRST_COMPLETED)
        for task in done:task.result()
    finally:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        await live.close();await stt.aclose();await provider.client.close();sink.cancel()
        report={'status':'pending_human_review','elapsed_s':time.perf_counter()-begun,'event_counts':event_counts,'errors':errors,
            'history':[{'role':m.role,'content':m.content} for m in session.runtime.history],
            'memory_records':len(memory.store.list_for_character(session.runtime.memory_scope_id)),
            'ledger_ids':[e.id for e in memory.ledger.list_for_character(session.runtime.memory_scope_id)],
            'state':asdict(session.runtime.state),'raw_audio_saved':False,'streaming_tts_verified':False,
            'human_confirmation':{},'output_stream_closed':sink.stream is None,'stt_worker_closed':stt.process is None}
        (args.out/'duplex_acceptance_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stt-model',required=True);p.add_argument('--model',default='qwen/qwen3.5-9b')
    p.add_argument('--base-url',default='http://127.0.0.1:1234/v1');p.add_argument('--input-device',type=int,default=1)
    p.add_argument('--output-device',type=int,default=0);p.add_argument('--vad-threshold',type=float,default=500)
    p.add_argument('--duration',type=float,default=1200);p.add_argument('--out',type=Path,default=Path('benchmarks/duplex'))
    args=p.parse_args()
    if args.duration<=0:p.error('--duration must be positive')
    answer=input('Confirm you are wearing headphones and the microphone cannot pick up the AI playback; type yes to continue: ').strip().lower()
    if answer!='yes':raise SystemExit('Acoustic isolation not confirmed; the automatic barge-in test was not started.')
    try:asyncio.run(run(args))
    except KeyboardInterrupt:print('Stopped; check the report and confirm by hand.')
if __name__=='__main__':main()
