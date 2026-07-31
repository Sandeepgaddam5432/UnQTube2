# -*- coding: utf-8 -*-
# Apple-styled entrypoint for the UnQTube2 web UI.
#
# It runs webui/Main.py exactly as-is and only swaps the stylesheet:
#   1. the Apple glassmorphism CSS is injected right after set_page_config
#   2. the legacy neon stylesheet inside Main.py is suppressed
#
# No application logic is duplicated or modified here, so Main.py stays the
# single source of truth for behaviour.

import os
import runpy
import sys

import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from apple_theme import APPLE_CSS  # noqa: E402

# Markers that identify the old cyberpunk stylesheet shipped inside Main.py.
_LEGACY_MARKERS = ('CYBERPUNK FUTURISTIC THEME', '--neon-blue', '--neon-purple')

_original_markdown = st.markdown
_original_set_page_config = st.set_page_config


def _themed_set_page_config(*args, **kwargs):
    result = _original_set_page_config(*args, **kwargs)
    # set_page_config must be the first Streamlit call, so the theme goes in
    # immediately after it and therefore wins the cascade.
    _original_markdown(APPLE_CSS, unsafe_allow_html=True)
    return result


def _themed_markdown(body='', *args, **kwargs):
    if isinstance(body, str) and any(marker in body for marker in _LEGACY_MARKERS):
        return None
    return _original_markdown(body, *args, **kwargs)


st.set_page_config = _themed_set_page_config
st.markdown = _themed_markdown

runpy.run_path(os.path.join(_HERE, 'Main.py'), run_name='__main__')
