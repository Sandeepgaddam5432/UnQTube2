# -*- coding: utf-8 -*-
# Apple-styled entrypoint for the UnQTube2 web UI.
#
# Runs webui/Main.py exactly as-is and only changes presentation:
#   1. injects the theme immediately after set_page_config
#   2. suppresses the legacy neon stylesheet that ships inside Main.py
#   3. replaces the duplicated plain title with a real hero header
#
# Main.py stays the single source of truth for behaviour.

import os
import runpy
import sys

import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from apple_theme import APPLE_CSS  # noqa: E402

# Fragments that identify the old cyberpunk stylesheet inside Main.py.
_LEGACY_MARKERS = ('CYBERPUNK FUTURISTIC THEME', '--neon-blue', '--neon-purple')

# Main.py prints a plain title and then a second heading with the same name.
# The hero replaces both.
_DUPLICATE_HEADING = 'UnQTube2 Video Creator'

_HERO_TEMPLATE = '''
<div class='uq-hero'>
  <div class='uq-hero-badge'>__BADGE__</div>
  <h1 class='uq-hero-title'>UnQTube2</h1>
  <p class='uq-hero-sub'>Give it a topic. It writes the script, records the voice-over, finds the footage, burns in subtitles and renders a finished short video.</p>
</div>
'''

_original_markdown = st.markdown
_original_title = st.title
_original_set_page_config = st.set_page_config


def _hero_html(title_text):
    badge = 'AI Video Studio'
    for token in str(title_text).split():
        if token.startswith('v') and any(ch.isdigit() for ch in token):
            badge = 'AI Video Studio  -  ' + token
            break
    return _HERO_TEMPLATE.replace('__BADGE__', badge)


def _themed_set_page_config(*args, **kwargs):
    result = _original_set_page_config(*args, **kwargs)
    # set_page_config must be the first Streamlit call, so the theme goes in
    # right after it and therefore wins the cascade.
    _original_markdown(APPLE_CSS, unsafe_allow_html=True)
    return result


def _themed_markdown(body='', *args, **kwargs):
    if isinstance(body, str):
        if any(marker in body for marker in _LEGACY_MARKERS):
            return None
        if _DUPLICATE_HEADING in body:
            return None
    return _original_markdown(body, *args, **kwargs)


def _themed_title(body='', *args, **kwargs):
    return _original_markdown(_hero_html(body), unsafe_allow_html=True)


st.set_page_config = _themed_set_page_config
st.markdown = _themed_markdown
st.title = _themed_title

runpy.run_path(os.path.join(_HERE, 'Main.py'), run_name='__main__')
