import os
import platform
import sys
from uuid import uuid4
from datetime import datetime

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
h1 {
    padding-top: 0 !important;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 1.15rem;
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

        # Filter voices based on selected TTS server
        for v in all_voices:
            if selected_tts_server == "azure-tts-v2":
                # V2 version voice name contains "v2"
                if "V2" in v:
                    filtered_voices.append(v)
            else:
                # V1 version voice name does not contain "v2"
                if "V2" not in v:
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
            )
            voice_name = selected_friendly_name
            config.ui["voice_name"] = voice_name
            
        with col2:
            preview_button = st.button("🔊 " + tr("Preview"), use_container_width=True)
            
            # Handle voice preview functionality
            if preview_button and voice_name:
                with st.spinner(f"Generating preview for {friendly_names.get(voice_name, voice_name)}..."):
                    try:
                        # Define sample text for preview
                        sample_text = tr("Hello, this is a preview of my voice for the UnQTube2 project.")
                        
                        # Create a temporary file for the audio
                        preview_file = utils.storage_dir("temp", create=True)
                        preview_file = os.path.join(preview_file, f"voice_preview_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3")
                        
                        # Generate preview audio using the same TTS function used in video generation
                        sub_maker = voice.tts(
                            text=sample_text,
                            voice_name=voice_name,
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
    if selected_tts_server == "azure-tts-v2" or (
        voice_name and voice.is_azure_v2_voice(voice_name)
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

    if selected_tts_server == "siliconflow" or (
        voice_name and voice.is_siliconflow_voice(voice_name)
    ):
        saved_siliconflow_api_key = config.siliconflow.get("api_key", "")
        siliconflow_api_key = st.text_input(
            tr("SiliconFlow API Key"),
            value=saved_siliconflow_api_key,
            type="password",
            key="siliconflow_api_key_input",
        )
        config.siliconflow["api_key"] = siliconflow_api_key
        
    if selected_tts_server == "google-gemini" or (
        voice_name and voice.is_google_gemini_voice(voice_name)
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
# Create tabs for the main workflow
tabs = st.tabs(["📝 " + tr("Script"), "🔍 " + tr("Settings"), "🎥 " + tr("Video")])

# Tab 1: Script
with tabs[0]:
    # Video Subject Input
    st.subheader(tr("Video Subject"))
    params = VideoParams(video_subject="")
    params.video_subject = st.text_input(
        tr("Enter a topic or keyword for your video"),
        value=st.session_state["video_subject"],
        key="video_subject_input",
        placeholder=tr("Example: Benefits of meditation")
    ).strip()

    # Script language selection
    support_locales = [
        "zh-CN", "zh-HK", "zh-TW", "de-DE", "en-US", "fr-FR", "vi-VN", "th-TH",
    ]
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        video_languages = [
            (tr("Auto Detect"), ""),
        ]
        for code in support_locales:
            video_languages.append((code, code))

        selected_index = st.selectbox(
            tr("Script Language"),
            index=0,
            options=range(len(video_languages)),
            format_func=lambda x: video_languages[x][0],
        )
        params.video_language = video_languages[selected_index][1]
    
    with col2:
        if st.button(
            "✨ " + tr("Generate Video Script and Keywords"), 
            key="auto_generate_script",
            use_container_width=True
        ):
            with st.spinner(tr("Generating Video Script and Keywords...")):
                script = llm.generate_script(
                    video_subject=params.video_subject, language=params.video_language
                )
                terms = llm.generate_terms(params.video_subject, script)
                if "Error: " in script:
                    st.error(tr(script))
                elif "Error: " in terms:
                    st.error(tr(terms))
                else:
                    st.session_state["video_script"] = script
                    st.session_state["video_terms"] = ", ".join(terms)
                    st.success(tr("Script and keywords generated successfully!"))
    
    # Display script in a larger area
    st.subheader(tr("Video Script"))
    params.video_script = st.text_area(
        "",
        value=st.session_state["video_script"],
        height=250,
        placeholder=tr("Your script will appear here. You can also write your own script.")
    )
    
    # Keywords section
    st.subheader(tr("Video Keywords"))
    col1, col2 = st.columns([3, 1])
    
    with col1:
        params.video_terms = st.text_area(
            "",
            value=st.session_state["video_terms"],
            placeholder=tr("Keywords help find relevant video materials")
        )
    
    with col2:
        if st.button(
            "🔍 " + tr("Generate Keywords"), 
            key="auto_generate_terms",
            use_container_width=True
        ):
            if not params.video_script:
                st.error(tr("Please Enter the Video Script"))
            else:
                with st.spinner(tr("Generating Video Keywords...")):
                    terms = llm.generate_terms(params.video_subject, params.video_script)
                    if "Error: " in terms:
                        st.error(tr(terms))
                    else:
                        st.session_state["video_terms"] = ", ".join(terms)
                        st.success(tr("Keywords generated successfully!"))

# Tab 2: Settings
with tabs[1]:
    col1, col2 = st.columns(2)
    
    with col1:
        # Video settings
        st.subheader(tr("Video Settings"))
        
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
            tr("Video Source"),
            options=range(len(video_sources)),
            format_func=lambda x: video_sources[x][0],
            index=saved_video_source_index,
        )
        params.video_source = video_sources[selected_index][1]
        config.app["video_source"] = params.video_source

        if params.video_source == "local":
            uploaded_files = st.file_uploader(
                tr("Upload Local Files"),
                type=["mp4", "mov", "avi", "flv", "mkv", "jpg", "jpeg", "png"],
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

        video_concat_modes = [
            (tr("Sequential"), "sequential"),
            (tr("Random"), "random"),
        ]
        selected_index = st.selectbox(
            tr("Video Concat Mode"),
            options=range(len(video_concat_modes)),
            format_func=lambda x: video_concat_modes[x][0],
            index=1,
        )
        params.video_concat_mode = VideoConcatMode(
            video_concat_modes[selected_index][1]
        )

        # Video transition mode
        video_transition_modes = [
            (tr("None"), VideoTransitionMode.none.value),
            (tr("Shuffle"), VideoTransitionMode.shuffle.value),
            (tr("FadeIn"), VideoTransitionMode.fade_in.value),
            (tr("FadeOut"), VideoTransitionMode.fade_out.value),
            (tr("SlideIn"), VideoTransitionMode.slide_in.value),
            (tr("SlideOut"), VideoTransitionMode.slide_out.value),
        ]
        selected_index = st.selectbox(
            tr("Video Transition Mode"),
            options=range(len(video_transition_modes)),
            format_func=lambda x: video_transition_modes[x][0],
            index=0,
        )
        params.video_transition_mode = VideoTransitionMode(
            video_transition_modes[selected_index][1]
        )

        cols = st.columns(2)
        with cols[0]:
            params.video_clip_duration = st.select_slider(
                tr("Clip Duration (seconds)"), 
                options=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                value=3
            )
        with cols[1]:
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

        # Advanced BGM settings in expander
        with st.expander(tr("Background Music (BGM) Settings"), expanded=False):
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
            saved_font_name = config.ui.get("font_name", "MicrosoftYaHeiBold.ttc")
            saved_font_name_index = 0
            if saved_font_name in font_names:
                saved_font_name_index = font_names.index(saved_font_name)
            params.font_name = st.selectbox(
                tr("Font"), font_names, index=saved_font_name_index
            )
            config.ui["font_name"] = params.font_name
            
            # Advanced subtitle settings in expander
            with st.expander(tr("Advanced Subtitle Settings"), expanded=False):
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

# Tab 3: Video Generation
with tabs[2]:
    st.subheader(tr("Generate Your Video"))
    
    # Check required fields and show warnings if needed
    warning_shown = False
    
    if not params.video_subject and not params.video_script:
        st.warning("⚠️ " + tr("You need to provide either a video subject or a script"))
        warning_shown = True
    
    if params.video_source == "pexels" and not config.app.get("pexels_api_keys", ""):
        st.warning("⚠️ " + tr("Pexels API Key is required for Pexels video source"))
        warning_shown = True
        
    if params.video_source == "pixabay" and not config.app.get("pixabay_api_keys", ""):
        st.warning("⚠️ " + tr("Pixabay API Key is required for Pixabay video source"))
        warning_shown = True
    
    # Generate button with proper spacing
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_button = st.button(
            "🎬 " + tr("Generate Video"), 
            use_container_width=True, 
            type="primary",
            disabled=warning_shown
        )

    # Progress information area
    log_container = st.empty()

if start_button:
    config.save_config()
    task_id = str(uuid4())
        
    # Double check requirements
    if not params.video_subject and not params.video_script:
        st.error(tr("Video Script and Subject Cannot Both Be Empty"))
        scroll_to_bottom()
        st.stop()

    if params.video_source not in ["pexels", "pixabay", "local"]:
        st.error(tr("Please Select a Valid Video Source"))
        scroll_to_bottom()
        st.stop()

    if params.video_source == "pexels" and not config.app.get("pexels_api_keys", ""):
        st.error(tr("Please Enter the Pexels API Key"))
        scroll_to_bottom()
        st.stop()

    if params.video_source == "pixabay" and not config.app.get("pixabay_api_keys", ""):
        st.error(tr("Please Enter the Pixabay API Key"))
        scroll_to_bottom()
        st.stop()
        
    # Validate voice selection - Fix for "Invalid voice" crash
    if not voice_name and (selected_tts_server.startswith("azure") or selected_tts_server == "siliconflow"):
        st.warning(tr("Please select a voice before generating the video."))
        scroll_to_bottom()
        st.stop()
        
    # Additional check to ensure a voice is selected for any TTS provider
    if not voice_name:
        st.warning(tr("Please select a voice before generating the video."))
        scroll_to_bottom()
        st.stop()

    # Handle uploaded files
    uploaded_files = []  # Define this if it wasn't defined in the settings tab
    if uploaded_files:
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
        
        # Update status for script generation
        status.update(label=tr("Generating video script..."))
        
        # Update status for material search
        status.update(label=tr("Searching for video clips..."))
        
        # Update status for speech generation
        status.update(label=tr("Generating speech audio..."))
        
        # Process the video
        result = tm.start(task_id=task_id, params=params)
        
        # Update status for rendering
        status.update(label=tr("Rendering final video..."))
        
        if not result or "videos" not in result:
            status.update(label=tr("Video generation failed!"), state="error")
            st.error(tr("Video Generation Failed"))
            logger.error(tr("Video Generation Failed"))
            scroll_to_bottom()
            st.stop()

        # Complete status when successful
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
