"""Kokoro-82M speech for every cue sentence x two voices. 16 kHz mono wav, RMS loudness-normalized, cached by hash.
Usage: python src/tts.py   (CPU is enough; resumable)"""
import sys, json, time
import numpy as np
import soundfile as sf
import librosa
from channels import CUES, CH, AUDIO, audio_path


def norm(x):
    a = CH['audio']
    x = x - x.mean()
    x = x * (10 ** (a['rms_dbfs'] / 20) / (np.sqrt((x ** 2).mean()) + 1e-9))
    pk = np.abs(x).max()
    return (x * min(1.0, a['peak_limit'] / pk)).astype(np.float32)


def main():
    from kokoro import KPipeline
    AUDIO.mkdir(parents=True, exist_ok=True)
    pipe = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    man, t0 = {}, time.time()
    for cond, text in CUES.items():
        if not text or cond.startswith('val_'):
            continue
        for voice, vid in CH['voices'].items():
            p = audio_path(text, voice)
            if not p.exists():
                audio = np.concatenate([np.asarray(r.audio) for r in pipe(text, voice=vid, speed=1.0)])
                y = norm(librosa.resample(audio.astype(np.float32), orig_sr=24000, target_sr=16000))
                sf.write(str(p) + '.part.wav', y, 16000, subtype='PCM_16')
                (p.parent / (p.name + '.part.wav')).replace(p)
            info = sf.info(str(p))
            man[f'{cond}|{voice}'] = {'file': p.name, 'sec': round(info.duration, 2), 'text': text}
    (AUDIO / 'manifest.json').write_text(json.dumps(man, indent=1))
    print(len(man), 'clips', f'{time.time() - t0:.0f}s', 'duration range', min(m['sec'] for m in man.values()), max(m['sec'] for m in man.values()))


if __name__ == '__main__':
    main()
