import os
import platform
import sys
from uuid import uuid4
from datetime import datetime
import json

import streamlit as st
from loguru import logger

# Add the root directory of the project to the system path to allow importing modules from the project
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
    print("******** sys.path ********")
    print(sys.path)
    print("")

from app.config import config
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
    VideoResolution,
)
from app.services import llm, voice
from app.services import task as tm
from app.utils import utils

st.set_page_config(
    page_title="UnQTube2",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://github.com/Sandeepgaddam5432/UnQTube2/issues",
        "About": "# UnQTube2\nSimply provide a topic or keyword for a video, and it will "
        "automatically generate the video copy, video materials, video subtitles, "
        "and video background music before synthesizing a high-definition short "
        "video.\n\nhttps://github.com/Sandeepgaddam5432/UnQTube2",
    },
)


streamlit_style = """
<style>
/* ============================================
   🎨 UnQTube2 - CYBERPUNK FUTURISTIC THEME
   ============================================ */

/* Google Fonts Import */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Roboto:wght@300;400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ============================================
   🌌 ROOT VARIABLES - Neon Color Palette
   ============================================ */
:root {
    --bg-primary: #0e1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #1a1f2e;
    --neon-blue: #00d4ff;
    --neon-purple: #bd00ff;
    --neon-pink: #ff006e;
    --neon-cyan: #00fff2;
    --neon-green: #39ff14;
    --text-primary: #ffffff;
    --text-secondary: #a0aec0;
    --text-muted: #6b7280;
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.08);
    --glass-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
    --gradient-neon: linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-purple) 50%, var(--neon-pink) 100%);
    --gradient-glow: linear-gradient(135deg, rgba(0, 212, 255, 0.3) 0%, rgba(189, 0, 255, 0.3) 100%);
}

/* ============================================
   🎭 ANIMATIONS - Keyframes
   ============================================ */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

@keyframes neonPulse {
    0%, 100% {
        box-shadow: 0 0 5px var(--neon-blue),
                    0 0 10px var(--neon-blue),
                    0 0 20px var(--neon-blue);
    }
    50% {
        box-shadow: 0 0 10px var(--neon-blue),
                    0 0 20px var(--neon-blue),
                    0 0 40px var(--neon-blue),
                    0 0 60px var(--neon-purple);
    }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes glowBorder {
    0%, 100% {
        border-color: var(--neon-blue);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    50% {
        border-color: var(--neon-purple);
        box-shadow: 0 0 25px rgba(189, 0, 255, 0.4);
    }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ============================================
   📱 GLOBAL STYLES
   ============================================ */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Roboto', sans-serif !important;
}

.stApp > header {
    background: transparent !important;
}

/* Main content area */
.main .block-container {
    padding: 2rem 3rem !important;
    max-width: 1400px !important;
    animation: fadeIn 0.8s ease-out;
}

/* ============================================
   📝 TYPOGRAPHY
   ============================================ */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif !important;
    background: var(--gradient-neon);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
    text-shadow: none !important;
    padding-top: 0 !important;
}

h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: 3px !important;
    margin-bottom: 1.5rem !important;
    animation: fadeInUp 0.8s ease-out, gradientShift 4s ease infinite;
}

h2 {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
}

h3 {
    font-size: 1.2rem !important;
    font-weight: 500 !important;
}

p, span, label, .stMarkdown {
    color: var(--text-secondary) !important;
    font-family: 'Roboto', sans-serif !important;
}

/* ============================================
   🔘 BUTTONS - Neon Glow Effect
   ============================================ */
.stButton > button {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(189, 0, 255, 0.1) 100%) !important;
    border: 1px solid var(--neon-blue) !important;
    border-radius: 12px !important;
    color: var(--neon-blue) !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.2) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-purple) 100%) !important;
    border-color: var(--neon-cyan) !important;
    color: var(--bg-primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.5),
                0 0 60px rgba(189, 0, 255, 0.3),
                inset 0 0 20px rgba(255, 255, 255, 0.1) !important;
    animation: neonPulse 1.5s ease-in-out infinite;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Primary Button - Extra Glow */
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-purple) 100%) !important;
    color: var(--bg-primary) !important;
    font-size: 1.1rem !important;
    padding: 1rem 2.5rem !important;
    animation: float 3s ease-in-out infinite;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-pink) 100%) !important;
    box-shadow: 0 0 40px rgba(0, 212, 255, 0.7),
                0 0 80px rgba(189, 0, 255, 0.5),
                0 0 120px rgba(255, 0, 110, 0.3) !important;
}

/* ============================================
   📋 INPUT FIELDS - Glassmorphism
   ============================================ */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Roboto', sans-serif !important;
    padding: 0.75rem 1rem !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--neon-blue) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3),
                inset 0 0 10px rgba(0, 212, 255, 0.1) !important;
    animation: glowBorder 2s ease-in-out infinite;
}

.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic;
}

/* ============================================
   🎚️ SLIDERS - Neon Track
   ============================================ */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, var(--neon-blue) 0%, var(--neon-purple) 100%) !important;
    height: 6px !important;
    border-radius: 3px !important;
}

.stSlider > div > div > div > div > div {
    background: var(--neon-cyan) !important;
    border: 2px solid var(--neon-blue) !important;
    box-shadow: 0 0 15px var(--neon-blue) !important;
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    transition: all 0.2s ease !important;
}

.stSlider > div > div > div > div > div:hover {
    transform: scale(1.2) !important;
    box-shadow: 0 0 25px var(--neon-blue), 0 0 40px var(--neon-purple) !important;
}

/* ============================================
   ☑️ CHECKBOXES - Custom Neon Style
   ============================================ */
.stCheckbox > label > div[data-testid="stCheckbox"] > div {
    border: 2px solid var(--neon-blue) !important;
    border-radius: 6px !important;
    background: var(--glass-bg) !important;
    transition: all 0.3s ease !important;
}

.stCheckbox > label > div[data-testid="stCheckbox"] > div:hover {
    box-shadow: 0 0 15px rgba(0, 212, 255, 0.4) !important;
}

/* ============================================
   📑 TABS - Futuristic Design
   ============================================ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--glass-bg) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
    border: 1px solid var(--glass-border) !important;
    gap: 0.5rem !important;
}

.stTabs [data-baseweb="tab-list"] button {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.3s ease !important;
    border: 1px solid transparent !important;
}

.stTabs [data-baseweb="tab-list"] button:hover {
    color: var(--neon-blue) !important;
    background: rgba(0, 212, 255, 0.1) !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(189, 0, 255, 0.2) 100%) !important;
    color: var(--neon-cyan) !important;
    border: 1px solid var(--neon-blue) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
}

.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 0.95rem !important;
}

/* ============================================
   📊 EXPANDERS - Glass Card Style
   ============================================ */
.streamlit-expanderHeader {
    font-family: 'Orbitron', sans-serif !important;
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--neon-blue) !important;
    transition: all 0.3s ease !important;
}

.streamlit-expanderHeader:hover {
    background: rgba(0, 212, 255, 0.1) !important;
    border-color: var(--neon-blue) !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.2) !important;
}

.streamlit-expanderContent {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    animation: fadeIn 0.3s ease-out;
}

/* ============================================
   📦 CONTAINERS - Glassmorphism Cards
   ============================================ */
[data-testid="stVerticalBlock"] > div:has(> .stMarkdown),
[data-testid="stHorizontalBlock"] {
    animation: fadeInUp 0.6s ease-out;
}

.stContainer, div[data-testid="stExpander"] {
    animation: slideInLeft 0.5s ease-out;
}

/* Status containers */
.stStatus {
    background: var(--glass-bg) !important;
    border: 1px solid var(--neon-blue) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.2) !important;
}

/* ============================================
   📊 SIDEBAR - Neon Panel
   ============================================ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
    animation: slideInLeft 0.6s ease-out;
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
    padding: 1.5rem !important;
}

[data-testid="stSidebar"] .stSubheader {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-cyan) !important;
    font-size: 0.9rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--glass-border) !important;
    padding-bottom: 0.5rem !important;
    margin-bottom: 1rem !important;
}

/* ============================================
   📋 SELECTBOX - Custom Dropdown
   ============================================ */
.stSelectbox [data-baseweb="select"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
}

.stSelectbox [data-baseweb="select"]:hover {
    border-color: var(--neon-blue) !important;
}

div[data-baseweb="popover"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--neon-blue) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5),
                0 0 30px rgba(0, 212, 255, 0.2) !important;
}

div[data-baseweb="popover"] li {
    color: var(--text-secondary) !important;
    transition: all 0.2s ease !important;
}

div[data-baseweb="popover"] li:hover {
    background: rgba(0, 212, 255, 0.1) !important;
    color: var(--neon-blue) !important;
}

/* ============================================
   ⚠️ ALERTS & MESSAGES
   ============================================ */
.stAlert, .stSuccess, .stWarning, .stError, .stInfo {
    border-radius: 12px !important;
    border: 1px solid !important;
    backdrop-filter: blur(10px) !important;
    animation: fadeInUp 0.4s ease-out;
}

div[data-testid="stNotification"] {
    background: var(--glass-bg) !important;
    border: 1px solid var(--neon-blue) !important;
    border-radius: 12px !important;
}

/* Success message */
.stSuccess {
    background: rgba(57, 255, 20, 0.1) !important;
    border-color: var(--neon-green) !important;
}

/* Warning message */
.stWarning {
    background: rgba(255, 165, 0, 0.1) !important;
    border-color: #ffa500 !important;
}

/* Error message */
.stError {
    background: rgba(255, 0, 110, 0.1) !important;
    border-color: var(--neon-pink) !important;
}

/* Info message */
.stInfo {
    background: rgba(0, 212, 255, 0.1) !important;
    border-color: var(--neon-blue) !important;
}

/* ============================================
   📊 PROGRESS BAR - Neon Glow
   ============================================ */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--neon-blue) 0%, var(--neon-purple) 50%, var(--neon-pink) 100%) !important;
    background-size: 200% 100% !important;
    animation: shimmer 2s linear infinite !important;
    border-radius: 10px !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
}

.stProgress > div > div {
    background: var(--glass-bg) !important;
    border-radius: 10px !important;
}

/* ============================================
   🎥 VIDEO PLAYER
   ============================================ */
.stVideo {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 2px solid var(--glass-border) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3),
                0 0 30px rgba(0, 212, 255, 0.1) !important;
    transition: all 0.3s ease !important;
}

.stVideo:hover {
    border-color: var(--neon-blue) !important;
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4),
                0 0 50px rgba(0, 212, 255, 0.2) !important;
    transform: scale(1.02);
}

/* ============================================
   📝 CODE BLOCKS
   ============================================ */
.stCodeBlock, code {
    font-family: 'JetBrains Mono', monospace !important;
    background: var(--bg-secondary) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
}

/* ============================================
   🎨 CUSTOM CLASSES - Main Title & Generate Button
   ============================================ */
.main-title {
    text-align: center !important;
    margin-bottom: 2.5rem !important;
    font-size: 3rem !important;
    animation: fadeInUp 0.8s ease-out;
}

.generate-btn {
    margin-top: 2rem !important;
}

.generate-btn button {
    width: 100% !important;
    font-size: 1.3rem !important;
    padding: 1.2rem 2rem !important;
    background: linear-gradient(135deg, var(--neon-blue) 0%, var(--neon-purple) 50%, var(--neon-pink) 100%) !important;
    background-size: 200% 200% !important;
    animation: gradientShift 3s ease infinite, float 4s ease-in-out infinite !important;
}

/* ============================================
   🔧 SPINNER - Neon Loading
   ============================================ */
.stSpinner > div {
    border-color: var(--neon-blue) transparent transparent transparent !important;
}

/* ============================================
   🎨 SCROLLBAR - Custom Neon Style
   ============================================ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--neon-blue) 0%, var(--neon-purple) 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, var(--neon-cyan) 0%, var(--neon-pink) 100%);
}

/* ============================================
   🌟 SPECIAL EFFECTS - Particles Background
   ============================================ */
.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(ellipse at 20% 80%, rgba(0, 212, 255, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(189, 0, 255, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(255, 0, 110, 0.03) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ============================================
   📱 RESPONSIVE TWEAKS
   ============================================ */
@media (max-width: 768px) {
    h1 {
        font-size: 1.8rem !important;
    }
    
    .main .block-container {
        padding: 1rem !important;
    }
    
    .stButton > button {
        padding: 0.5rem 1rem !important;
        font-size: 0.9rem !important;
    }
}

/* ============================================
   ✨ MICRO-INTERACTIONS
   ============================================ */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--neon-green) 0%, var(--neon-cyan) 100%) !important;
    border: none !important;
    color: var(--bg-primary) !important;
}

.stDownloadButton > button:hover {
    box-shadow: 0 0 30px rgba(57, 255, 20, 0.5) !important;
    transform: translateY(-3px) !important;
}

/* Color picker styling */
.stColorPicker > div {
    border-radius: 10px !important;
    overflow: hidden;
}

/* File uploader */
.stFileUploader > div {
    background: var(--glass-bg) !important;
    border: 2px dashed var(--neon-blue) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

.stFileUploader > div:hover {
    border-color: var(--neon-purple) !important;
    background: rgba(0, 212, 255, 0.05) !important;
}

/* Toast notifications */
.stToast {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--neon-blue) !important;
    border-radius: 12px !important;
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.3) !important;
}

</style>
"""
st.markdown(streamlit_style, unsafe_allow_html=True)

# Define resource directory
font_dir = os.path.join(root_dir, "resource", "fonts")
song_dir = os.path.join(root_dir, "resource", "songs")
i18n_dir = os.path.join(root_dir, "webui", "i18n")
config_file = os.path.join(root_dir, "webui", ".streamlit", "webui.toml")
system_locale = utils.get_system_locale()


if "video_subject" not in st.session_state:
    st.session_state["video_subject"] = ""
if "video_script" not in st.session_state:
    st.session_state["video_script"] = ""
if "voice_over_script" not in st.session_state:
    st.session_state["voice_over_script"] = ""
if "video_terms" not in st.session_state:
    st.session_state["video_terms"] = ""
if "ui_language" not in st.session_state:
    st.session_state["ui_language"] = config.ui.get("language", system_locale)
if "gemini_models" not in st.session_state:
    st.session_state["gemini_models"] = []

# Load language files
locales = utils.load_locales(i18n_dir)

# Helper function to translate text
def tr(key):
    loc = locales.get(st.session_state["ui_language"], {})
    return loc.get("Translation", {}).get(key, key)

def get_all_fonts():
    fonts = []
    for root, dirs, files in os.walk(font_dir):
        for file in files:
            if file.endswith(".ttf") or file.endswith(".ttc"):
                fonts.append(file)
    fonts.sort()
    return fonts


def get_all_songs():
    songs = []
    for root, dirs, files in os.walk(song_dir):
        for file in files:
            if file.endswith(".mp3"):
                songs.append(file)
    return songs


def open_task_folder(task_id):
    try:
        sys = platform.system()
        path = os.path.join(root_dir, "storage", "tasks", task_id)
        if os.path.exists(path):
            if sys == "Windows":
                os.system(f"start {path}")
            if sys == "Darwin":
                os.system(f"open {path}")
    except Exception as e:
        logger.error(e)


def scroll_to_bottom():
    js = """
    <script>
        console.log("scroll_to_bottom");
        function scroll(dummy_var_to_force_repeat_execution){
            var sections = parent.document.querySelectorAll('section.main');
            console.log(sections);
            for(let index = 0; index<sections.length; index++) {
                sections[index].scrollTop = sections[index].scrollHeight;
            }
        }
        scroll(1);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)


def init_log():
    logger.remove()
    _lvl = "DEBUG"

    def format_record(record):
        # Get full path of the file in log record
        file_path = record["file"].path
        # Convert absolute path to path relative to project root directory
        relative_path = os.path.relpath(file_path, root_dir)
        # Update file path in the record
        record["file"].path = f"./{relative_path}"
        # Return modified format string
        # You can adjust the format here as needed
        record["message"] = record["message"].replace(root_dir, ".")

        _format = (
            "<green>{time:%Y-%m-%d %H:%M:%S}</> | "
            + "<level>{level}</> | "
            + '"{file.path}:{line}":<blue> {function}</> '
            + "- <level>{message}</>"
            + "\n"
        )
        return _format

    logger.add(
        sys.stdout,
        level=_lvl,
        format=format_record,
        colorize=True,
    )

# Initialize logging
init_log()

# Setup header with title and language selector
st.title(f"UnQTube2 v{config.project_version} 🎬")

# SIDEBAR CONFIGURATION
with st.sidebar:
    # Language selection in sidebar
    st.subheader("🌐 " + tr("Language"))
    display_languages = []
    selected_index = 0
    for i, code in enumerate(locales.keys()):
        display_languages.append(f"{code} - {locales[code].get('Language')}")
        if code == st.session_state.get("ui_language", ""):
            selected_index = i

    selected_language = st.selectbox(
        "Language / 语言",
        options=display_languages,
        index=selected_index,
        key="language_selector",
    )
    if selected_language:
        code = selected_language.split(" - ")[0].strip()
        st.session_state["ui_language"] = code
        config.ui["language"] = code

    # LLM Configuration
    st.subheader("🧠 " + tr("LLM Settings"))
    llm_providers = [
        "OpenAI",
        "Moonshot",
        "Azure",
        "Qwen",
        "DeepSeek",
        "Gemini",
        "Ollama",
        "G4f",
        "OneAPI",
        "Cloudflare",
        "ERNIE",
        "Pollinations",
    ]
    saved_llm_provider = config.app.get("llm_provider", "OpenAI").lower()
    saved_llm_provider_index = 0
    for i, provider in enumerate(llm_providers):
        if provider.lower() == saved_llm_provider:
            saved_llm_provider_index = i
            break

    llm_provider = st.selectbox(
        tr("LLM Provider"),
        options=llm_providers,
        index=saved_llm_provider_index,
    )

    llm_provider = llm_provider.lower()
    config.app["llm_provider"] = llm_provider

    llm_api_key = config.app.get(f"{llm_provider}_api_key", "")
    llm_secret_key = config.app.get(
        f"{llm_provider}_secret_key", ""
    )  # only for baidu ernie
    llm_base_url = config.app.get(f"{llm_provider}_base_url", "")
    llm_model_name = config.app.get(f"{llm_provider}_model_name", "")
    llm_account_id = config.app.get(f"{llm_provider}_account_id", "")

    st_llm_api_key = st.text_input(
        tr("API Key"), value=llm_api_key, type="password"
    )
    
    st_llm_base_url = st.text_input(tr("Base Url"), value=llm_base_url)
    
    # Dynamic model selection for Gemini
    if llm_provider == "gemini":
        if not llm_model_name:
            llm_model_name = "gemini-1.0-pro"

    # If API key is provided, try to get available models
    if st_llm_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=st_llm_api_key)
            
            # Only fetch models once or when API key changes
            if not st.session_state["gemini_models"] or st.session_state.get("last_gemini_api_key") != st_llm_api_key:
                with st.spinner(tr("Fetching available models...")):
                    models = genai.list_models()
                    # Filter for text generation models
                    text_models = [
                        model.name for model in models 
                        if hasattr(model, 'supported_generation_methods') 
                        and 'generateContent' in model.supported_generation_methods
                    ]
                    st.session_state["gemini_models"] = text_models
                    st.session_state["last_gemini_api_key"] = st_llm_api_key
            
            if st.session_state["gemini_models"]:
                # Extract just model names for the display
                model_names = [m.split('/')[-1] for m in st.session_state["gemini_models"]]
                # Use selectbox instead of text_input
                model_index = 0
                if llm_model_name in model_names:
                    model_index = model_names.index(llm_model_name)
                
                selected_model = st.selectbox(
                    tr("Model Name"), 
                    options=model_names,
                    index=model_index
                )
                st_llm_model_name = selected_model
            else:
                st_llm_model_name = st.text_input(
                    tr("Model Name"),
                    value=llm_model_name,
                    key=f"{llm_provider}_model_name_input"
                )
        except Exception as e:
            st.warning(f"Could not fetch models: {str(e)}")
            st_llm_model_name = st.text_input(
                tr("Model Name"),
                value=llm_model_name,
                key=f"{llm_provider}_model_name_input"
            )
    elif llm_provider == "ernie":
        st_llm_model_name = None
        st_llm_secret_key = st.text_input(
            tr("Secret Key"), value=llm_secret_key, type="password"
        )
        config.app[f"{llm_provider}_secret_key"] = st_llm_secret_key
    else:
        st_llm_model_name = st.text_input(
            tr("Model Name"),
            value=llm_model_name,
            key=f"{llm_provider}_model_name_input"
        )
    
    # Save config values
    if st_llm_api_key:
        config.app[f"{llm_provider}_api_key"] = st_llm_api_key
    if st_llm_base_url:
        config.app[f"{llm_provider}_base_url"] = st_llm_base_url
    if st_llm_model_name:
        config.app[f"{llm_provider}_model_name"] = st_llm_model_name

    if llm_provider == "cloudflare":
        st_llm_account_id = st.text_input(
            tr("Account ID"), value=llm_account_id
        )
        if st_llm_account_id:
            config.app[f"{llm_provider}_account_id"] = st_llm_account_id

    # API Keys for video sources
    st.subheader("🎬 " + tr("Video Source Settings"))

    def get_keys_from_config(cfg_key):
        api_keys = config.app.get(cfg_key, [])
        if isinstance(api_keys, str):
            api_keys = [api_keys]
        api_key = ", ".join(api_keys)
        return api_key

    def save_keys_to_config(cfg_key, value):
        value = value.replace(" ", "")
        if value:
            config.app[cfg_key] = value.split(",")

    pexels_api_key = get_keys_from_config("pexels_api_keys")
    pexels_api_key = st.text_input(
        tr("Pexels API Key"), value=pexels_api_key, type="password"
    )
    save_keys_to_config("pexels_api_keys", pexels_api_key)

    pixabay_api_key = get_keys_from_config("pixabay_api_keys")
    pixabay_api_key = st.text_input(
        tr("Pixabay API Key"), value=pixabay_api_key, type="password"
    )
    save_keys_to_config("pixabay_api_keys", pixabay_api_key)

    # Voice Settings
    st.subheader("🔊 " + tr("Voice Settings"))
    
    # TTS server selection
    tts_servers = [
        ("azure-tts-v1", "Azure TTS V1"),
        ("azure-tts-v2", "Azure TTS V2"),
        ("siliconflow", "SiliconFlow TTS"),
        ("google-gemini", "Google Gemini TTS"),
    ]

    # Get saved TTS server, default is v1
    saved_tts_server = config.ui.get("tts_server", "azure-tts-v1")
    saved_tts_server_index = 0
    for i, (server_value, _) in enumerate(tts_servers):
        if server_value == saved_tts_server:
            saved_tts_server_index = i
            break

    selected_tts_server_index = st.selectbox(
        tr("TTS Servers"),
        options=range(len(tts_servers)),
        format_func=lambda x: tts_servers[x][1],
        index=saved_tts_server_index,
        key="selected_tts_server"
    )

    selected_tts_server = tts_servers[selected_tts_server_index][0]
    config.ui["tts_server"] = selected_tts_server

    # Get voice list based on selected TTS server
    filtered_voices = []

    if selected_tts_server == "siliconflow":
        # Get Silicon Flow voice list
        filtered_voices = voice.get_siliconflow_voices()
    elif selected_tts_server == "google-gemini":
        # Get Google Gemini voice list
        filtered_voices = voice.get_google_gemini_voices()
    else:
        # Get Azure voice list
        all_voices = voice.get_all_azure_voices(filter_locals=None)

        # Load the mapping of Indian language codes to available voices
        indian_voices_map = {}
        try:
            with open('temp_voices.json', 'r') as f:
                indian_voices_map = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or has invalid JSON, use an empty map
            indian_voices_map = {}

        # Get the currently selected language from the script tab
        selected_language = params.video_language if 'params' in locals() else ""
        
        # Filter voices based on selected TTS server and language
        for v in all_voices:
            # First filter by TTS server version
            server_match = False
            if selected_tts_server == "azure-tts-v2":
                server_match = "V2" in v
            else:  # "azure-tts-v1"
                server_match = "V2" not in v
                
            # Then filter by language if an Indian language is selected
            language_match = True
            if selected_language and selected_language in indian_voices_map:
                # If it's an Indian language, only show voices for that language
                language_match = False
                for voice_name in indian_voices_map[selected_language]:
                    if voice_name.split('-')[0] in v:  # Check if the voice name matches
                        language_match = True
                        break
            
            # Add voice to filtered list if both filters pass
            if server_match and language_match:
                filtered_voices.append(v)

    friendly_names = {
        v: v.replace("Female", tr("Female"))
        .replace("Male", tr("Male"))
        .replace("Neural", "")
        for v in filtered_voices
    }

    saved_voice_name = config.ui.get("voice_name", "")
    saved_voice_name_index = 0

    # Check if saved voice is in current filtered voice list
    if saved_voice_name in friendly_names:
        saved_voice_name_index = list(friendly_names.keys()).index(saved_voice_name)
    else:
        # If not, select a default voice based on current UI language
        for i, v in enumerate(filtered_voices):
            if v.lower().startswith(st.session_state["ui_language"].lower()):
                saved_voice_name_index = i
                break

    # If no matching voice found, use the first voice
    if saved_voice_name_index >= len(friendly_names) and friendly_names:
        saved_voice_name_index = 0

    # Speech rate and volume - Define these BEFORE they're used in the voice preview
    voice_rate = st.select_slider(
        tr("Speech Rate"),
        options=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0],
        value=config.ui.get("voice_rate", 1.0)
    )
    config.ui["voice_rate"] = voice_rate
    
    voice_volume = st.select_slider(
        tr("Speech Volume"),
        options=[0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0],
        value=config.ui.get("voice_volume", 1.0)
    )
    config.ui["voice_volume"] = voice_volume

    # Ensure there are voices available
    if friendly_names:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_friendly_name = st.selectbox(
                tr("Voice"),
                options=list(friendly_names.keys()),
                format_func=lambda x: friendly_names[x],
                index=min(saved_voice_name_index, len(friendly_names) - 1) if friendly_names else 0,
                key="selected_voice"
            )
            # Store in both session state and config
            voice_name = st.session_state["selected_voice"]
            config.ui["voice_name"] = voice_name
            
        with col2:
            preview_button = st.button("🔊 " + tr("Preview"), use_container_width=True)
            
            # Handle voice preview functionality
            if preview_button:
                voice_to_preview = st.session_state.get("selected_voice")
                if voice_to_preview:
                    with st.spinner(f"Generating preview for {friendly_names.get(voice_to_preview, voice_to_preview)}..."):
                        try:
                            # Define sample text for preview
                            sample_text = tr("Hello, this is a preview of my voice for the UnQTube2 project.")
                            
                            # First try the API endpoint (faster)
                            try:
                                import requests
                                api_url = "http://localhost:8080/audio/preview"
                                
                                response = requests.post(
                                    api_url,
                                    json={
                                        "text": sample_text,
                                        "voice_name": voice_to_preview,
                                        "voice_rate": voice_rate
                                    },
                                    timeout=15
                                )
                                
                                if response.status_code == 200:
                                    # Display the audio preview from API
                                    st.audio(response.content, format="audio/mp3")
                                    # Using a flag instead of return statement
                                    preview_success = True
                                else:
                                    preview_success = False
                            except Exception as api_err:
                                logger.warning(f"API preview failed, falling back to local: {str(api_err)}")
                                preview_success = False
                                # Fall back to local generation if API fails
                            
                            # Only proceed with local generation if API call failed
                            if not preview_success:
                                # Fallback: Create a temporary file for the audio locally
                                preview_file = utils.storage_dir("temp", create=True)
                                preview_file = os.path.join(preview_file, f"voice_preview_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3")
                                
                                # Generate preview audio using the same TTS function used in video generation
                                sub_maker = voice.tts(
                                    text=sample_text,
                                    voice_name=voice_to_preview,
                                    voice_rate=voice_rate,
                                    voice_file=preview_file,
                                    voice_volume=voice_volume
                                )
                                
                                if os.path.exists(preview_file):
                                    # Display an audio player with the preview
                                    st.audio(preview_file)
                                else:
                                    st.error(tr("Failed to generate voice preview."))
                        except Exception as e:
                            st.error(f"Error generating voice preview: {str(e)}")
                else:
                    st.warning(tr("Please select a voice to preview."))
    else:
        # If no voices available, show prompt message
        st.warning(
            tr(
            "No voices available for the selected TTS server. "
            "Please select a different TTS server."
            )
        )
        voice_name = ""
        config.ui["voice_name"] = ""

    # TTS API settings based on selected service
    selected_tts_server_from_session = st.session_state.get("selected_tts_server", 0)
    tts_server_value = tts_servers[selected_tts_server_from_session][0]
    
    # Define voice_from_session here, before it's used
    voice_from_session = st.session_state.get("selected_voice", "")
    
    if tts_server_value == "azure-tts-v2" or (
        voice_from_session and voice.is_azure_v2_voice(voice_from_session)
    ):
        saved_azure_speech_region = config.azure.get("speech_region", "")
        saved_azure_speech_key = config.azure.get("speech_key", "")

        azure_speech_region = st.text_input(
        tr("Azure Speech Region"), value=saved_azure_speech_region
        )
        azure_speech_key = st.text_input(
        tr("Azure Speech API Key"), 
            value=saved_azure_speech_key,
        type="password"
        )

        config.azure["speech_region"] = azure_speech_region
        config.azure["speech_key"] = azure_speech_key

    if tts_server_value == "siliconflow" or (
        voice_from_session and voice.is_siliconflow_voice(voice_from_session)
    ):
        saved_siliconflow_api_key = config.siliconflow.get("api_key", "")
        siliconflow_api_key = st.text_input(
            tr("SiliconFlow API Key"),
            value=saved_siliconflow_api_key,
            type="password",
            key="siliconflow_api_key_input",
        )
        config.siliconflow["api_key"] = siliconflow_api_key
        
    if tts_server_value == "google-gemini" or (
        voice_from_session and voice.is_google_gemini_voice(voice_from_session)
    ):
        # Check if google_gemini config exists, initialize if not
        if not hasattr(config, "google_gemini"):
            config.google_gemini = {}
            
        saved_gemini_api_key = config.google_gemini.get("api_key", "")
        gemini_api_key = st.text_input(
            tr("Google Gemini API Key"),
            value=saved_gemini_api_key,
            type="password",
            key="gemini_api_key_input",
        )
        config.google_gemini["api_key"] = gemini_api_key
        
        # Model selection
        gemini_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
        saved_gemini_model = config.google_gemini.get("model_name", "gemini-2.5-flash")
        gemini_model = st.selectbox(
            tr("Gemini Model"),
            options=gemini_models,
            index=gemini_models.index(saved_gemini_model) if saved_gemini_model in gemini_models else 0,
            key="gemini_model_selection"
        )
        config.google_gemini["model_name"] = gemini_model

    # Log settings
    st.subheader("⚙️ " + tr("Log Settings"))
    hide_log = st.checkbox(
        tr("Hide Log"), value=config.ui.get("hide_log", False)
    )
    config.ui["hide_log"] = hide_log

    hide_config = st.checkbox(
        tr("Hide Basic Settings"), value=config.app.get("hide_config", False)
    )
    config.app["hide_config"] = hide_config

# MAIN CONTENT AREA
# Initialize parameters
params = VideoParams(video_subject="")

# Simplified Main UI - Two-Click Philosophy
st.markdown("<h1 class='main-title'>UnQTube2 Video Creator</h1>", unsafe_allow_html=True)

# Create a nice card-like container for the main UI
main_container = st.container()
with main_container:
    # 1. Subject Input - the primary input field
    params.video_subject = st.text_input(
        "1. " + tr("Enter Your Video Subject"),
        value=st.session_state["video_subject"],
        key="video_subject_input",
        placeholder=tr("Example: Benefits of meditation")
    ).strip()

    # Save subject to session state for persistence
    st.session_state["video_subject"] = params.video_subject

    # 2. Duration input - the second main control (now using number_input instead of slider)
    target_duration = st.number_input(
        "2. " + tr("Set Target Duration (seconds)"),
        min_value=1,
        value=60,
        step=1
    )
    params.target_duration = target_duration if target_duration > 0 else None

    # 3. Generate button - large and prominent
    st.markdown("<div class='generate-btn'>", unsafe_allow_html=True)
    start_button = st.button(
        "🎬 " + tr("Generate Video"), 
        use_container_width=True,
        type="primary",
        key="main_generate_button"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Progress information area for displaying generation progress
log_container = st.empty()

# Advanced Customization Expander - closed by default
with st.expander("🔧 " + tr("Advanced Customization (Optional)"), expanded=False):
    # Create tabs for organizing the advanced settings
    adv_tabs = st.tabs(["📝 " + tr("Script"), "🔍 " + tr("Settings"), "🎥 " + tr("Video")])
    
    # Tab 1: Script Options
    with adv_tabs[0]:
        # Script language selection
        support_locales = [
            "zh-CN", "zh-HK", "zh-TW", "de-DE", "en-US", "fr-FR", "vi-VN", "th-TH",
            # Added Indian languages
            "hi-IN", "te-IN", "ta-IN", "kn-IN", "bn-IN", "mr-IN", "gu-IN", "ml-IN", "pa-IN",
        ]
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            video_languages = [
                (tr("Auto Detect"), ""),
            ]
            # Add language display names for Indian languages
            language_display_names = {
                "hi-IN": "Hindi (हिन्दी)",
                "te-IN": "Telugu (తెలుగు)",
                "ta-IN": "Tamil (தமிழ்)",
                "kn-IN": "Kannada (ಕನ್ನಡ)",
                "bn-IN": "Bengali (বাংলা)",
                "mr-IN": "Marathi (मराठी)",
                "gu-IN": "Gujarati (ગુજરાતી)",
                "ml-IN": "Malayalam (മലയാളം)",
                "pa-IN": "Punjabi (ਪੰਜਾਬੀ)"
            }
            
            for locale in support_locales:
                # Use special display names for Indian languages
                if locale in language_display_names:
                    display_name = language_display_names[locale]
                else:
                    display_name = locale
                video_languages.append((display_name, locale))
            
            selected_language_index = 0
            saved_language = config.ui.get("video_language", "")
            for i, (_, lang_code) in enumerate(video_languages):
                if lang_code == saved_language:
                    selected_language_index = i
                    break
            
            selected_language_option = st.selectbox(
                tr("Script Language"),
                options=range(len(video_languages)),
                format_func=lambda x: video_languages[x][0],
                index=selected_language_index,
                key="selected_language"
            )
            selected_language = video_languages[selected_language_option][1]
            config.ui["video_language"] = selected_language
            params.video_language = selected_language
            
            # Add hybrid language mode checkbox
            hybrid_mode = False
            if selected_language and not selected_language.startswith("en"):
                hybrid_mode = st.checkbox(
                    tr("Hybrid Language Mode"),
                    value=False,
                    help=tr("Generate voice-over in selected language but subtitles in English")
                )
                
                if hybrid_mode:
                    st.info(tr("Hybrid mode enabled: Voice-over will be in the selected language, subtitles will be in English."))
        
        with col2:
            # Custom Voice-over Script - renamed to make it clear this is an override
            voice_over_script = st.text_area(
                tr("Custom Voice-over Script"),
                value=st.session_state.get("voice_over_script", ""),
                height=150,
                placeholder=tr("Enter your custom voice-over script here, or leave blank to auto-generate..."),
                help=tr("This is an optional custom script that will be used for the voice-over. Leave blank to auto-generate from the video subject."),
            )
            # Update both the old and new session state keys for backward compatibility
            st.session_state["video_script"] = voice_over_script
            st.session_state["voice_over_script"] = voice_over_script
            params.voice_over_script = voice_over_script
            
            # Custom Subtitle script input - only show in hybrid mode
            if hybrid_mode:
                subtitle_script = st.text_area(
                    tr("Custom Subtitle Script (English)"),
                    value=st.session_state.get("subtitle_script", ""),
                    height=150,
                    placeholder=tr("Enter English subtitle script here, or leave blank to auto-generate..."),
                    help=tr("This custom script will be used for subtitles. Leave blank to auto-generate English subtitles."),
                )
                st.session_state["subtitle_script"] = subtitle_script
                params.subtitle_script = subtitle_script
            else:
                # In non-hybrid mode, subtitle script is the same as voice-over script
                params.subtitle_script = ""  # Will use voice_over_script by default in the backend
        
        # Keywords section
        st.subheader(tr("Video Keywords"))
        col1, col2 = st.columns([3, 1])
        
        with col1:
            params.video_terms = st.text_area(
                tr("Custom Keywords"),
                value=st.session_state["video_terms"],
                placeholder=tr("Keywords help find relevant video materials")
            )
        
        with col2:
            if st.button(
                "🔍 " + tr("Generate Keywords"), 
                key="auto_generate_terms",
                use_container_width=True
            ):
                if not params.voice_over_script and not params.video_subject:
                    st.error(tr("Please Enter the Video Subject or Script"))
                else:
                    with st.spinner(tr("Generating Video Keywords...")):
                        terms = llm.generate_terms(params.video_subject, params.voice_over_script)
                        if "Error: " in terms:
                            st.error(tr(terms))
                        else:
                            st.session_state["video_terms"] = ", ".join(terms)
                            st.success(tr("Keywords generated successfully!"))

    # Tab 2: Settings
    with adv_tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Video settings
            st.subheader(tr("Image Settings"))
            
            video_sources = [
                (tr("Pexels"), "pexels"),
                (tr("Pixabay"), "pixabay"),
                (tr("Local file"), "local"),
                (tr("TikTok"), "douyin"),
                (tr("Bilibili"), "bilibili"),
                (tr("Xiaohongshu"), "xiaohongshu"),
            ]

            saved_video_source_name = config.app.get("video_source", "pexels")
            saved_video_source_index = [v[1] for v in video_sources].index(
                saved_video_source_name
            )

            selected_index = st.selectbox(
                tr("Image Source"),
                options=range(len(video_sources)),
                format_func=lambda x: video_sources[x][0],
                index=saved_video_source_index,
            )
            params.video_source = video_sources[selected_index][1]
            config.app["video_source"] = params.video_source

            if params.video_source == "local":
                uploaded_files = st.file_uploader(
                    tr("Upload Local Files"),
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                )

            video_aspect_ratios = [
                (tr("Portrait") + " (9:16)", VideoAspect.portrait.value),
                (tr("Landscape") + " (16:9)", VideoAspect.landscape.value),
            ]
            selected_index = st.selectbox(
                tr("Video Ratio"),
                options=range(len(video_aspect_ratios)),
                format_func=lambda x: video_aspect_ratios[x][0],
            )
            params.video_aspect = VideoAspect(video_aspect_ratios[selected_index][1])

            # Add video resolution options
            video_resolutions = [
                (tr("HD") + " (720p)", VideoResolution.hd_720p.value),
                (tr("Full HD") + " (1080p)", VideoResolution.full_hd.value),
                (tr("Ultra HD") + " (4K)", VideoResolution.ultra_hd.value),
            ]
            selected_index = st.selectbox(
                tr("Video Resolution"),
                options=range(len(video_resolutions)),
                format_func=lambda x: video_resolutions[x][0],
            )
            params.video_resolution = VideoResolution(video_resolutions[selected_index][1])

            # Remove video concat modes and transition modes as they're not relevant for image-based videos
            
            # Keep only the number of videos to generate
            params.video_count = st.select_slider(
                tr("Videos to Generate"),
                options=[1, 2, 3, 4, 5],
                value=1
            )

        with col2:
            # Background music settings
            st.subheader(tr("Audio Settings"))

            # Basic BGM setting
            bgm_options = [
                (tr("No Background Music"), ""),
                (tr("Random Background Music"), "random"),
                (tr("Custom Background Music"), "custom"),
            ]
            selected_index = st.selectbox(
                tr("Background Music"),
                index=1,
                options=range(len(bgm_options)),
                format_func=lambda x: bgm_options[x][0],
            )
            # Get the selected background music type
            params.bgm_type = bgm_options[selected_index][1]

            # Advanced BGM settings - using subheader instead of expander
            st.subheader(tr("Background Music (BGM) Settings"))
            
            # Show or hide components based on the selection
            if params.bgm_type == "custom":
                custom_bgm_file = st.text_input(
                    tr("Custom Background Music File"), key="custom_bgm_file_input"
                )
                if custom_bgm_file and os.path.exists(custom_bgm_file):
                    params.bgm_file = custom_bgm_file
                    st.success(f"✅ {tr('Custom music selected')}: **{custom_bgm_file}**")

            params.bgm_volume = st.select_slider(
                tr("Background Music Volume"),
                options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                value=0.2
            )

            # Subtitle settings
            st.subheader(tr("Subtitle Settings"))
            params.subtitle_enabled = st.checkbox(tr("Enable Subtitles"), value=True)
            
            if params.subtitle_enabled:
                # Basic subtitle settings
                font_names = get_all_fonts()
                
                # Map language codes to recommended fonts
                language_font_recommendations = {
                    "hi-IN": "NotoSansDevanagari-Regular.ttf",  # Hindi
                    "te-IN": "NotoSansTelugu-Regular.ttf",      # Telugu
                    "ta-IN": "NotoSansTamil-Regular.ttf",       # Tamil
                    "kn-IN": "NotoSansKannada-Regular.ttf",     # Kannada
                    "bn-IN": "NotoSansBengali-Regular.ttf",     # Bengali
                    "mr-IN": "NotoSansDevanagari-Regular.ttf",  # Marathi
                    "gu-IN": "NotoSansGujarati-Regular.ttf",    # Gujarati
                    "ml-IN": "NotoSansMalayalam-Regular.ttf",   # Malayalam
                    "pa-IN": "NotoSansGurmukhi-Regular.ttf",    # Punjabi
                }
                
                # Get current language selected
                current_language = params.video_language
                
                # Determine the recommended font based on language
                recommended_font = None
                if current_language in language_font_recommendations:
                    recommended_font = language_font_recommendations[current_language]
                
                # Get the saved font name from config or use the recommended font for the selected language
                saved_font_name = config.ui.get("font_name", "MicrosoftYaHeiBold.ttc")
                
                # If we have a recommended font for this language and it's in our font list, use it
                if recommended_font and recommended_font in font_names:
                    saved_font_name = recommended_font
                    
                saved_font_name_index = 0
                if saved_font_name in font_names:
                    saved_font_name_index = font_names.index(saved_font_name)
                    
                # Create font options with recommendations for Indian languages
                font_options = font_names.copy()
                font_format_func = lambda x: x
                
                # If we have an Indian language selected, modify the display function to mark recommended fonts
                if current_language in language_font_recommendations:
                    font_format_func = lambda x: f"{x} ✓ (Recommended)" if x == language_font_recommendations[current_language] else x
                
                params.font_name = st.selectbox(
                    tr("Font"),
                    options=font_options,
                    index=saved_font_name_index,
                    format_func=font_format_func
                )
                config.ui["font_name"] = params.font_name
                
                # If an Indian language is selected but not using the recommended font, show a hint
                if current_language in language_font_recommendations and params.font_name != language_font_recommendations[current_language]:
                    st.info(f"Tip: For {current_language}, we recommend using {language_font_recommendations[current_language]} for best results.")

                # Advanced subtitle settings - using subheader instead of expander
                st.subheader(tr("Advanced Subtitle Settings"))
                
                subtitle_positions = [
                    (tr("Top"), "top"),
                    (tr("Center"), "center"),
                    (tr("Bottom"), "bottom"),
                    (tr("Custom"), "custom"),
                ]
                selected_index = st.selectbox(
                    tr("Position"),
                    index=2,
                    options=range(len(subtitle_positions)),
                    format_func=lambda x: subtitle_positions[x][0],
                )
                params.subtitle_position = subtitle_positions[selected_index][1]

                if params.subtitle_position == "custom":
                    custom_position = st.slider(
                        tr("Position (% from top)"), 
                        min_value=0,
                        max_value=100,
                        value=70,
                        step=5
                    )
                    params.custom_position = float(custom_position)

                font_cols = st.columns(2)
                with font_cols[0]:
                    saved_text_fore_color = config.ui.get("text_fore_color", "#FFFFFF")
                    params.text_fore_color = st.color_picker(
                        tr("Font Color"), saved_text_fore_color
                    )
                    config.ui["text_fore_color"] = params.text_fore_color

                with font_cols[1]:
                    saved_font_size = config.ui.get("font_size", 60)
                    params.font_size = st.slider(tr("Font Size"), 30, 100, saved_font_size)
                    config.ui["font_size"] = params.font_size

                stroke_cols = st.columns(2)
                with stroke_cols[0]:
                    params.stroke_color = st.color_picker(tr("Stroke Color"), "#000000")
                with stroke_cols[1]:
                    params.stroke_width = st.slider(tr("Stroke Width"), 0.0, 10.0, 1.5)

    # Tab 3: Video Generation (placed in advanced expander but simplified)
    with adv_tabs[2]:
        st.subheader(tr("Additional Video Settings"))
        
        # Check required fields and show warnings if needed
        warning_shown = False
        
        if params.video_source == "pexels" and not config.app.get("pexels_api_keys", ""):
            st.warning("⚠️ " + tr("Pexels API Key is required for Pexels video source"))
            warning_shown = True
            
        if params.video_source == "pixabay" and not config.app.get("pixabay_api_keys", ""):
            st.warning("⚠️ " + tr("Pixabay API Key is required for Pixabay video source"))
            warning_shown = True

if start_button:
    config.save_config()
    task_id = str(uuid4())
        
    # Double check requirements - simplified
    if not params.video_subject:
        st.error(tr("Please Enter a Video Subject"))
        scroll_to_bottom()
        st.stop()

    # Implement the "Smart" Logic for script generation
    # Get the value from the advanced settings area if provided, otherwise use auto-generation
    advanced_script = st.session_state.get("voice_over_script", "").strip()
    
    # If advanced script is empty, we'll use the auto-generation flow
    if not advanced_script:
        # The params.voice_over_script will be empty, which signals the backend
        # to auto-generate the script from params.video_subject
        logger.info(tr("Using automatic script generation from subject"))
    else:
        # Use the provided custom script
        params.voice_over_script = advanced_script
        logger.info(tr("Using custom script provided in advanced settings"))

    # Further requirements checks
    if params.video_source not in ["pexels", "pixabay", "local"]:
        st.error(tr("Please Select a Valid Video Source"))
        scroll_to_bottom()
        st.stop()

    if params.video_source == "pexels" and not config.app.get("pexels_api_keys", ""):
        st.error(tr("Please Enter the Pexels API Key in the sidebar"))
        scroll_to_bottom()
        st.stop()

    if params.video_source == "pixabay" and not config.app.get("pixabay_api_keys", ""):
        st.error(tr("Please Enter the Pixabay API Key in the sidebar"))
        scroll_to_bottom()
        st.stop()
        
    # Smart Voice Auto-Correction
    # Validate voice selection - Auto-select a default voice if none is selected
    voice_from_session = st.session_state.get("selected_voice", "")
    if not voice_from_session and filtered_voices:
        # Auto-select a voice based on the selected language
        for v in filtered_voices:
            language_code = params.video_language if params.video_language else "en-US"
            if language_code.split('-')[0].lower() in v.lower():
                voice_from_session = v
                st.session_state["selected_voice"] = v
                st.info(f"Auto-selected voice: {v}")
                break
        
        # If no matching voice, use the first available one
        if not voice_from_session:
            voice_from_session = filtered_voices[0]
            st.session_state["selected_voice"] = voice_from_session
            st.info(f"Auto-selected default voice: {voice_from_session}")
    
    # Set the voice name from session state for video generation
    params.voice_name = voice_from_session
    
    # Handle uploaded files
    uploaded_files = []  # Define this if it wasn't defined in the settings tab
    if 'uploaded_files' in locals() and uploaded_files:
        local_videos_dir = utils.storage_dir("local_videos", create=True)
        for file in uploaded_files:
            file_path = os.path.join(local_videos_dir, f"{file.file_id}_{file.name}")
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                m = MaterialInfo()
                m.provider = "local"
                m.url = file_path
                if not params.video_materials:
                    params.video_materials = []
                params.video_materials.append(m)

    # Log handling
    log_records = []

    def log_received(msg):
        if config.ui["hide_log"]:
            return
        with log_container:
            log_records.append(msg)
            st.code("\n".join(log_records))

    logger.add(log_received)

    # Show progress with detailed status updates
    with st.status(tr("Initializing video generation..."), expanded=True) as status:
        st.toast(tr("Generating Video"))
        logger.info(tr("Start Generating Video"))
        logger.info(utils.to_json(params))
        
        # Create a progress bar for the video generation process
        progress_bar = st.progress(0)
        
        # Update status for script generation
        status.update(label=tr("Generating video script..."))
        progress_bar.progress(0.1)
        
        # Update status for material search
        status.update(label=tr("Searching for video clips..."))
        progress_bar.progress(0.3)
        
        # Update status for speech generation
        status.update(label=tr("Generating speech audio..."))
        progress_bar.progress(0.4)
        
        # Process the video
        result = tm.start(task_id=task_id, params=params)
        
        # Update status for rendering
        status.update(label=tr("Rendering final video..."))
        progress_bar.progress(0.9)
        
        if not result or "videos" not in result:
            status.update(label=tr("Video generation failed!"), state="error")
            st.error(tr("Video Generation Failed"))
            logger.error(tr("Video Generation Failed"))
            scroll_to_bottom()
            st.stop()

        # Complete status when successful
        progress_bar.progress(1.0)
        status.update(label=tr("Completed!"), state="complete")
        video_files = result.get("videos", [])
            
    st.success(tr("Video Generation Completed"))
        
    try:
        if video_files:
            st.subheader(tr("Generated Videos"))
            cols = st.columns(min(len(video_files), 3))
            for i, url in enumerate(video_files):
                with cols[i % len(cols)]:
                    st.video(url)
                    # Optional: Add download button
                    with open(url, "rb") as file:
                        st.download_button(
                            label=tr("Download Video"),
                            data=file,
                            file_name=f"unqtube2_video_{i+1}.mp4",
                            mime="video/mp4",
                        )
    except Exception as e:
        st.error(f"Error displaying videos: {str(e)}")

    # Open task folder and show completion message
    open_task_folder(task_id)
    logger.info(tr("Video Generation Completed"))

# Save configuration on exit
config.save_config()
