'''
Kokoro-82M local neural TTS engine.

Why this exists:
  Microsoft tightened the Edge Read Aloud endpoint (Sec-MS-GEC anti-abuse
  token), so edge_tts started failing with:
      403, message='Invalid response status', url='wss://speech.platform.bing.com/...'
  That endpoint is undocumented and can break again at any time.

Kokoro-82M is an Apache-2.0 open-weight TTS model (82M params, ~350MB) that
runs fully offline on CPU or GPU. It is the highest ranked open-weight model
on public TTS preference leaderboards, so it does not sound robotic like
Piper / eSpeak. It needs no API key and makes no network calls after the
first model download.

This module is dependency-optional: if kokoro is not installed it installs
itself on first use (Colab friendly) and, if that fails, returns None so the
caller can fall back to Edge TTS.
'''

import os
import shutil
import subprocess
import sys
import threading

from loguru import logger

from app.utils import utils

SAMPLE_RATE = 24000
# edge_tts SubMaker offsets are in 100-nanosecond units; match that so the
# existing subtitle + duration code keeps working unchanged.
HUNDRED_NS = 10000000
# Small breath between sentences so narration does not sound machine-gunned.
PAUSE_SECONDS = 0.14

_LOCK = threading.Lock()
_PIPELINES = {}
_STATE = {'ready': False, 'failed': False}


class KokoroSubMaker:
    '''Duck-typed stand-in for edge_tts.SubMaker.

    voice.create_subtitle() and voice.get_audio_duration() only touch .subs
    and .offset, so this is all we need to stay compatible.
    '''

    def __init__(self):
        self.subs = []
        self.offset = []


# lang_code -> ordered voice list, best sounding first.
# Grades are from the official Kokoro voice quality table.
VOICES = {
    'a': {
        'female': ['af_heart', 'af_bella', 'af_nicole', 'af_aoede', 'af_kore', 'af_sarah', 'af_nova'],
        'male': ['am_michael', 'am_fenrir', 'am_puck', 'am_eric', 'am_liam', 'am_adam'],
    },
    'b': {
        'female': ['bf_emma', 'bf_isabella', 'bf_alice', 'bf_lily'],
        'male': ['bm_george', 'bm_fable', 'bm_lewis', 'bm_daniel'],
    },
    'e': {'female': ['ef_dora'], 'male': ['em_alex']},
    'f': {'female': ['ff_siwis'], 'male': ['ff_siwis']},
    'h': {'female': ['hf_alpha', 'hf_beta'], 'male': ['hm_omega', 'hm_psi']},
    'i': {'female': ['if_sara'], 'male': ['im_nicola']},
    'j': {'female': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi'], 'male': ['jm_kumo']},
    'p': {'female': ['pf_dora'], 'male': ['pm_alex']},
    'z': {
        'female': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao'],
        'male': ['zm_yunjian', 'zm_yunxi', 'zm_yunyang'],
    },
}

# ISO language prefix -> Kokoro lang_code.
# Anything not listed here is NOT supported by Kokoro and must fall back to
# Edge TTS (Telugu, Tamil, Kannada, Arabic, Russian, ... ).
LANG_MAP = {
    'en': 'a',
    'es': 'e',
    'fr': 'f',
    'hi': 'h',
    'it': 'i',
    'ja': 'j',
    'pt': 'p',
    'zh': 'z',
}

# English regions that should use the British voice pack.
BRITISH_REGIONS = ('GB', 'IE', 'AU', 'NZ', 'ZA', 'IN')


def _run(cmd, timeout=900):
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        logger.warning(f'kokoro: command failed {cmd}: {e}')
        return False


def is_kokoro_voice(voice_name) -> bool:
    if not voice_name:
        return False
    name = str(voice_name).strip().lower()
    return name.startswith('kokoro:') or name.startswith('kokoro-')


def list_voices() -> list:
    '''Display names for a UI dropdown, e.g. kokoro:af_heart-Female.'''
    out = []
    for lang_code in VOICES:
        for gender in ('female', 'male'):
            for voice_id in VOICES[lang_code][gender]:
                label = 'Female' if gender == 'female' else 'Male'
                entry = 'kokoro:' + voice_id + '-' + label
                if entry not in out:
                    out.append(entry)
    return out


def resolve_voice(voice_name, language=None):
    '''Map any voice name to (lang_code, voice_id).

    Accepts native Kokoro ids (kokoro:af_heart-Female) and also the existing
    Edge voice names already stored in user configs
    (en-AU-NatashaNeural-Female), so switching engines needs no UI change.

    Returns (None, None) when Kokoro cannot serve the requested language.
    '''
    raw = str(voice_name or '').strip()

    if is_kokoro_voice(raw):
        voice_id = raw.split(':', 1)[-1] if ':' in raw else raw.split('-', 1)[-1]
        voice_id = voice_id.split('-')[0].strip().lower()
        lang_code = voice_id[0] if voice_id else 'a'
        if lang_code not in VOICES:
            lang_code = 'a'
        return lang_code, voice_id

    gender = 'male' if raw.lower().endswith('-male') else 'female'

    locale = language or raw
    parts = str(locale).replace('_', '-').split('-')
    prefix = parts[0].lower() if parts and parts[0] else ''
    region = parts[1].upper() if len(parts) > 1 else ''

    lang_code = LANG_MAP.get(prefix)
    if not lang_code:
        return None, None

    if lang_code == 'a' and region in BRITISH_REGIONS:
        lang_code = 'b'

    candidates = VOICES[lang_code].get(gender) or VOICES[lang_code].get('female')
    if not candidates:
        return None, None
    return lang_code, candidates[0]


def ensure_ready(auto_install=True) -> bool:
    '''Import kokoro, installing it and espeak-ng on first use if needed.'''
    if _STATE['ready']:
        return True
    if _STATE['failed'] and not auto_install:
        return False

    with _LOCK:
        if _STATE['ready']:
            return True

        try:
            import kokoro  # noqa: F401
            import soundfile  # noqa: F401
            _STATE['ready'] = True
            return True
        except Exception:
            pass

        if not auto_install:
            return False

        logger.info('kokoro: first-time setup, installing engine (one time only)')

        if shutil.which('espeak-ng') is None and shutil.which('apt-get'):
            _run(['apt-get', '-qq', '-y', 'install', 'espeak-ng'], timeout=600)

        _run(
            [
                sys.executable,
                '-m',
                'pip',
                'install',
                '-q',
                'kokoro>=0.9.4',
                'soundfile',
                'misaki[en]',
            ]
        )

        try:
            import kokoro  # noqa: F401
            import soundfile  # noqa: F401
            _STATE['ready'] = True
            logger.success('kokoro: engine ready')
            return True
        except Exception as e:
            _STATE['failed'] = True
            logger.warning(f'kokoro: unavailable, will use edge tts instead ({e})')
            return False


def _get_pipeline(lang_code):
    if lang_code in _PIPELINES:
        return _PIPELINES[lang_code]
    with _LOCK:
        if lang_code in _PIPELINES:
            return _PIPELINES[lang_code]
        from kokoro import KPipeline

        device = None
        try:
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = None

        try:
            pipeline = KPipeline(lang_code=lang_code, device=device)
        except TypeError:
            pipeline = KPipeline(lang_code=lang_code)

        logger.info(f'kokoro: pipeline loaded (lang={lang_code}, device={device})')
        _PIPELINES[lang_code] = pipeline
        return pipeline


def _render(pipeline, text, voice_id, speed):
    import numpy as np

    segments = []
    for item in pipeline(text, voice=voice_id, speed=speed):
        audio = None
        if isinstance(item, (tuple, list)) and len(item) >= 3:
            audio = item[2]
        else:
            audio = getattr(item, 'audio', None)
        if audio is None:
            continue
        if hasattr(audio, 'detach'):
            audio = audio.detach().cpu().numpy()
        arr = np.asarray(audio, dtype='float32').reshape(-1)
        if arr.size:
            segments.append(arr)

    if not segments:
        return None
    return np.concatenate(segments)


def _write_output(samples, voice_file):
    import soundfile as sf

    base, ext = os.path.splitext(voice_file)
    ext = ext.lower()

    if ext == '.wav':
        sf.write(voice_file, samples, SAMPLE_RATE)
        return True

    wav_path = base + '.kokoro.wav'
    sf.write(wav_path, samples, SAMPLE_RATE)

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        ok = _run(
            [
                ffmpeg,
                '-y',
                '-loglevel',
                'error',
                '-i',
                wav_path,
                '-codec:a',
                'libmp3lame',
                '-b:a',
                '192k',
                voice_file,
            ],
            timeout=600,
        )
        if ok and os.path.exists(voice_file) and os.path.getsize(voice_file) > 0:
            try:
                os.remove(wav_path)
            except Exception:
                pass
            return True

    try:
        from moviepy import AudioFileClip

        clip = AudioFileClip(wav_path)
        clip.write_audiofile(voice_file, logger=None)
        clip.close()
        os.remove(wav_path)
        return True
    except Exception as e:
        logger.warning(f'kokoro: mp3 encode failed, keeping wav data ({e})')

    try:
        shutil.move(wav_path, voice_file)
        return True
    except Exception as e:
        logger.error(f'kokoro: could not write {voice_file}: {e}')
        return False


def synthesize(text, voice_name, voice_rate, voice_file, language=None):
    '''Generate narration with Kokoro.

    Returns a SubMaker-compatible object, or None so the caller can fall back
    to another engine. Never raises.
    '''
    try:
        text = (text or '').strip()
        if not text:
            return None

        lang_code, voice_id = resolve_voice(voice_name, language)
        if not lang_code:
            logger.info(
                f'kokoro: no voice for {voice_name} / {language}, using edge tts'
            )
            return None

        if not ensure_ready():
            return None

        import numpy as np

        pipeline = _get_pipeline(lang_code)

        try:
            speed = float(voice_rate) if voice_rate else 1.0
        except Exception:
            speed = 1.0
        speed = max(0.5, min(2.0, speed))

        chunks = [c.strip() for c in utils.split_string_by_punctuations(text) if c and c.strip()]
        if not chunks:
            chunks = [text]

        logger.info(
            f'kokoro: synthesizing {len(chunks)} segments '
            f'(voice={voice_id}, lang={lang_code}, speed={speed})'
        )

        pause = np.zeros(int(SAMPLE_RATE * PAUSE_SECONDS), dtype='float32')
        pause_ns = int(round(PAUSE_SECONDS * HUNDRED_NS))

        sub_maker = KokoroSubMaker()
        pieces = []
        cursor = 0

        for index, chunk in enumerate(chunks):
            audio = _render(pipeline, chunk, voice_id, speed)
            if audio is None or audio.size == 0:
                logger.warning(f'kokoro: empty audio for segment {index + 1}, skipped')
                continue

            duration_ns = int(round(audio.size / float(SAMPLE_RATE) * HUNDRED_NS))

            # Real measured timings -> subtitles line up with the audio.
            sub_maker.subs.append(chunk)
            sub_maker.offset.append((cursor, cursor + duration_ns))

            pieces.append(audio)
            cursor += duration_ns

            if index < len(chunks) - 1:
                pieces.append(pause)
                cursor += pause_ns

        if not pieces or not sub_maker.subs:
            logger.warning('kokoro: no audio produced')
            return None

        samples = np.concatenate(pieces)

        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples = (samples / peak) * 0.95

        if not _write_output(samples, voice_file):
            return None

        total = cursor / float(HUNDRED_NS)
        logger.success(
            f'kokoro: completed, {total:.1f}s of audio -> {voice_file}'
        )
        return sub_maker

    except Exception as e:
        logger.error(f'kokoro: failed, falling back to edge tts ({e})')
        return None


if __name__ == '__main__':
    maker = synthesize(
        text='Space is not empty nothingness. It is a dynamic fabric where gravity warps time.',
        voice_name='en-US-AriaNeural-Female',
        voice_rate=1.0,
        voice_file='kokoro_demo.mp3',
    )
    print(maker.subs if maker else 'failed')
