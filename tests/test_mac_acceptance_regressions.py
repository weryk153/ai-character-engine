import importlib.util
import subprocess
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from ai_character_engine.voice.acceptance import evaluate_hardware_acceptance
from ai_character_engine.voice.live import LiveVoiceRunner, CapturedUtterance
from ai_character_engine.voice.models import AudioFormat, TranscriptResult

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1,True,'100'])
def test_invalid_latency_cannot_pass_hardware_acceptance(bad):
 row=dict(input_source='microphone',stt_latency_ms=bad,llm_ttft_ms=1,llm_total_latency_ms=1,tts_ttfa_ms=1,end_to_end_latency_ms=1,tool_calls=1,tool_results_successful=1)
 r=evaluate_hardware_acceptance([row]*5,human_confirmation=dict(live_microphone=True,heard_speaker_audio=True,no_severe_audio_issue=True))
 assert r['status']=='failed'
 assert r['latency_ms']['stt_latency_ms']['n']==0

async def test_speech_end_latency_includes_endpointing_without_relabeling_inference():
 now=time.perf_counter()
 utterance=CapturedUtterance(b'\0\0',AudioFormat(),now-.6,now)
 llm=NS(reset=lambda:None,metrics=lambda:{},calls=[])
 result=NS(response=NS(text='測試'),tool_results=[])
 runner=LiveVoiceRunner(session=NS(process_event=AsyncMock(return_value=result),record=NS(id='s',user_id='u',character_id='c')),llm=llm,stt=NS(transcribe=AsyncMock(return_value=TranscriptResult('hello'))),tts=None,sink=None)
 r=await runner.run_turn(utterance)
 assert r['speech_end_to_stt_ms']>=600
 assert r['speech_end_to_stt_ms']-r['stt_latency_ms']>=590
 assert r['first_playback_ms'] is None
 assert r['output_mode']=='buffered_half_duplex' and r['streaming_tts'] is False

def test_installed_mac_sherpa_native_library_is_loadable():
 # Exercise the native import, not find_spec: the release lock originally
 # installed the wrapper while omitting its required ONNX runtime dylib.
 if sys.platform=='darwin' and importlib.util.find_spec('sherpa_onnx'):
  subprocess.run([sys.executable,'-c','import sherpa_onnx; assert sherpa_onnx.OfflineRecognizer'],check=True,capture_output=True)
