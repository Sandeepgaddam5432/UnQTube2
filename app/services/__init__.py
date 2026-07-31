'''
Compatibility shim for the app.services package.

edge-tts 7.x is a breaking rewrite of the 6.x API:
  - edge_tts.submaker.mktimestamp was removed
  - SubMaker.create_sub() / .subs / .offset were replaced by .cues

app/services/voice.py is written against the 6.x API. Downgrading is not an
option: 6.x is permanently blocked by Microsoft with
  403, message='Invalid response status'
because it cannot produce a valid Sec-MS-GEC anti-abuse token.

So instead of rewriting voice.py we restore the 6.x surface on top of 7.x.
Python always imports this package before app.services.voice, so the patch
is guaranteed to be in place before voice.py runs its own imports.

Kokoro (app/services/kokoro_tts.py) is the primary engine; this shim is what
keeps Edge TTS usable as a fallback.
'''


def _mktimestamp(time_unit) -> str:
    '''Convert 100-nanosecond units to HH:MM:SS.mmm (edge-tts 6.x format).'''
    hour = int(time_unit // 36000000000)
    minute = int((time_unit % 36000000000) // 600000000)
    seconds = (time_unit % 600000000) / 10000000
    return f'{hour:02d}:{minute:02d}:{seconds:06.3f}'


class _CompatSubMaker:
    '''edge-tts 6.x style SubMaker backed by the 7.x streaming API.

    Communicate.stream() still yields the same WordBoundary chunks in 7.x
    (type / offset / duration / text), only the collector class changed, so
    re-implementing the collector is enough.
    '''

    def __init__(self):
        self.subs = []
        self.offset = []

    def create_sub(self, timestamp, text):
        start, duration = timestamp[0], timestamp[1]
        self.offset.append((int(start), int(start) + int(duration)))
        self.subs.append(text)

    def feed(self, msg):
        '''7.x entry point, kept so newer callers keep working too.'''
        if not isinstance(msg, dict):
            return
        if msg.get('type') != 'WordBoundary':
            return
        self.create_sub(
            (msg.get('offset', 0), msg.get('duration', 0)),
            msg.get('text', ''),
        )

    def merge_cues(self, words=None):
        return None

    def get_srt(self) -> str:
        lines = []
        for index, (pair, text) in enumerate(zip(self.offset, self.subs), start=1):
            start, end = pair
            lines.append(str(index))
            lines.append(
                _mktimestamp(start).replace('.', ',')
                + ' --> '
                + _mktimestamp(end).replace('.', ',')
            )
            lines.append(text)
            lines.append('')
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.get_srt()


def _patch_edge_tts():
    try:
        import edge_tts
        from edge_tts import submaker as _submaker
    except Exception:
        # edge-tts not installed at all is fine, Kokoro is the primary engine.
        return

    if not hasattr(_submaker, 'mktimestamp'):
        _submaker.mktimestamp = _mktimestamp

    existing = getattr(_submaker, 'SubMaker', None)
    if existing is not None and hasattr(existing, 'create_sub'):
        # Already the 6.x API, nothing to do.
        return

    _CompatSubMaker.__name__ = 'SubMaker'
    _submaker.SubMaker = _CompatSubMaker
    edge_tts.SubMaker = _CompatSubMaker


_patch_edge_tts()
