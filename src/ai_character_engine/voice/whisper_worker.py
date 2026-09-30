"""Private persistent STT subprocess. PCM stays in pipes/memory, never log files."""
from __future__ import annotations
import base64
import contextlib
import json
import sys
from pathlib import Path


def main():
    config = json.loads(sys.stdin.readline())
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np
            sherpa = config['backend'] == 'sherpa-onnx'
            if sherpa:
                import sherpa_onnx
                folder = Path(config['model'])
                model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(folder / 'model.int8.onnx'), tokens=str(folder / 'tokens.txt'),
                    num_threads=config['threads'], use_itn=True,
                    language=config['language'] or 'auto', debug=False, provider='cpu')
            else:
                from faster_whisper import WhisperModel
                model = WhisperModel(config['model'], device='cpu', compute_type='int8',
                                     cpu_threads=config['threads'],
                                     local_files_only=config['local_files_only'])
        print(json.dumps({'ready': True}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            audio = np.frombuffer(base64.b64decode(request['pcm']), dtype='<i2').astype(np.float32) / 32768
            with contextlib.redirect_stdout(sys.stderr):
                if sherpa:
                    stream = model.create_stream()
                    stream.accept_waveform(16000, audio)
                    model.decode_stream(stream)
                    text, language = stream.result.text.strip(), config['language']
                else:
                    segments, info = model.transcribe(audio, language=config['language'],
                        beam_size=1, vad_filter=True, condition_on_previous_text=False)
                    text, language = ''.join(s.text for s in segments).strip(), info.language
            print(json.dumps({'text': text, 'language': language}), flush=True)
    except Exception as exc:
        # Avoid echoing audio, tokens, filesystem paths or arbitrary backend errors.
        print(json.dumps({'error': type(exc).__name__}), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
