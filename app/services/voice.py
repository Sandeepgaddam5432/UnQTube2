import asyncio
import os
import re
import base64
import json
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, Union
from pathlib import Path
from xml.sax.saxutils import unescape

import edge_tts
import requests
from edge_tts import SubMaker, submaker
from edge_tts.submaker import mktimestamp
from loguru import logger
from moviepy.video.tools import subtitles
from pydantic import BaseModel

from app.config import config
from app.utils import utils


# Gemini TTS implementation
class GeminiTTSConfig(BaseModel):
    api_key: str
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/models"
    model: str = "gemini-1.5-flash"
    voice_name: str = "Aoede"
    max_retries: int = 3
    timeout: int = 30


class GeminiTTSService:
    def __init__(self, config: GeminiTTSConfig):
        self.config = config
        self.session = None
    
    async def __aenter__(self):
        self.session = requests.Session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()

    def _build_request_payload(self, text: str, voice_config: Optional[str] = None) -> Dict[str, Any]:
        """Build the JSON payload for Gemini TTS API"""
        voice_name = voice_config or self.config.voice_name
        
        return {
            "contents": [{
                "parts": [{"text": f"Generate speech for the following text: {text}"}]
            }],
            "generationConfig": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": voice_name
                        }
                    }
                }
            }
        }

    async def generate_speech_async(
        self, 
        text: str, 
        voice_config: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate speech using Gemini TTS API (async)
        
        Args:
            text: Text to convert to speech
            voice_config: Voice configuration name
            output_path: Path to save the audio file
            
        Returns:
            Path to the generated audio file
        """
        if not self.session:
            raise RuntimeError("GeminiTTSService must be used as async context manager")
        
        url = f"{self.config.base_url}/{self.config.model}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key
        }
        
        payload = self._build_request_payload(text, voice_config)
        
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"Gemini TTS attempt {attempt + 1}/{self.config.max_retries}")
                
                response = self.session.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    response_data = response.json()
                    return await self._process_audio_response(response_data, output_path)
                else:
                    error_text = response.text
                    logger.error(f"Gemini TTS API error {response.status_code}: {error_text}")
                    
                    if response.status_code == 400:
                        raise ValueError(f"Invalid request: {error_text}")
                    elif response.status_code == 401:
                        raise ValueError("Invalid API key")
                    elif response.status_code == 429:
                        # Rate limit, wait and retry
                        await asyncio.sleep(2 ** attempt)
                        continue
                        
            except requests.RequestException as e:
                logger.error(f"Network error on attempt {attempt + 1}: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(1)
        
        raise RuntimeError(f"Failed to generate speech after {self.config.max_retries} attempts")

    async def _process_audio_response(self, response_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Process the API response and save audio data"""
        try:
            # Extract audio data from response
            candidates = response_data.get("candidates", [])
            if not candidates:
                raise ValueError("No audio candidates in response")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            audio_data = None
            for part in parts:
                if "inline_data" in part:
                    audio_data = part["inline_data"]["data"]
                    break
            
            if not audio_data:
                raise ValueError("No audio data found in response")
            
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Save to file
            if output_path:
                audio_path = Path(output_path)
            else:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio_path = Path(temp_file.name)
                temp_file.close()
            
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            
            logger.info(f"Audio saved to: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            logger.error(f"Error processing audio response: {e}")
            raise

    def generate_speech_sync(
        self, 
        text: str, 
        voice_config: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Synchronous wrapper for speech generation
        """
        return asyncio.run(self.generate_speech_async(text, voice_config, output_path))


# Integration with existing voice service
async def gemini_tts_v1(
    text: str,
    voice_name: str = "Aoede",
    api_key: str = None,
    output_path: str = None,
    **kwargs
) -> str:
    """
    Main function to integrate with existing voice service
    """
    if not api_key:
        api_key = config.app.get("gemini_api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
    
    config_obj = GeminiTTSConfig(
        api_key=api_key,
        voice_name=voice_name
    )
    
    async with GeminiTTSService(config_obj) as tts_service:
        return await tts_service.generate_speech_async(text, voice_name, output_path)


# Voice preview functionality
async def generate_voice_preview(
    voice_name: str,
    api_key: str = None,
    sample_text: str = "Hello, this is a voice preview sample."
) -> str:
    """
    Generate a short voice preview for UI
    """
    if not api_key:
        api_key = config.app.get("gemini_api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")
            
    config_obj = GeminiTTSConfig(
        api_key=api_key,
        voice_name=voice_name
    )
    
    async with GeminiTTSService(config_obj) as tts_service:
        return await tts_service.generate_speech_async(sample_text, voice_name)


def get_siliconflow_voices() -> list[str]:
    """
    Get the list of Silicon Flow voices

    Returns:
        Voice list, format is ["siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex", ...]
    """
    # Silicon Flow voice list and corresponding gender (for display)
    voices_with_gender = [
        ("FunAudioLLM/CosyVoice2-0.5B", "alex", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "anna", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "bella", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "benjamin", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "charles", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "claire", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "david", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "diana", "Female"),
    ]

    # Add siliconflow: prefix and format as display name
    return [
        f"siliconflow:{model}:{voice}-{gender}"
        for model, voice, gender in voices_with_gender
    ]


def get_all_azure_voices(filter_locals=None) -> list[str]:
    azure_voices_str = """
Name: af-ZA-AdriNeural
Gender: Female

Name: af-ZA-WillemNeural
Gender: Male

Name: am-ET-AmehaNeural
Gender: Male

Name: am-ET-MekdesNeural
Gender: Female

Name: ar-AE-FatimaNeural
Gender: Female

Name: ar-AE-HamdanNeural
Gender: Male

Name: ar-BH-AliNeural
Gender: Male

Name: ar-BH-LailaNeural
Gender: Female

Name: ar-DZ-AminaNeural
Gender: Female

Name: ar-DZ-IsmaelNeural
Gender: Male

Name: ar-EG-SalmaNeural
Gender: Female

Name: ar-EG-ShakirNeural
Gender: Male

Name: ar-IQ-BasselNeural
Gender: Male

Name: ar-IQ-RanaNeural
Gender: Female

Name: ar-JO-SanaNeural
Gender: Female

Name: ar-JO-TaimNeural
Gender: Male

Name: ar-KW-FahedNeural
Gender: Male

Name: ar-KW-NouraNeural
Gender: Female

Name: ar-LB-LaylaNeural
Gender: Female

Name: ar-LB-RamiNeural
Gender: Male

Name: ar-LY-ImanNeural
Gender: Female

Name: ar-LY-OmarNeural
Gender: Male

Name: ar-MA-JamalNeural
Gender: Male

Name: ar-MA-MounaNeural
Gender: Female

Name: ar-OM-AbdullahNeural
Gender: Male

Name: ar-OM-AyshaNeural
Gender: Female

Name: ar-QA-AmalNeural
Gender: Female

Name: ar-QA-MoazNeural
Gender: Male

Name: ar-SA-HamedNeural
Gender: Male

Name: ar-SA-ZariyahNeural
Gender: Female

Name: ar-SY-AmanyNeural
Gender: Female

Name: ar-SY-LaithNeural
Gender: Male

Name: ar-TN-HediNeural
Gender: Male

Name: ar-TN-ReemNeural
Gender: Female

Name: ar-YE-MaryamNeural
Gender: Female

Name: ar-YE-SalehNeural
Gender: Male

Name: az-AZ-BabekNeural
Gender: Male

Name: az-AZ-BanuNeural
Gender: Female

Name: bg-BG-BorislavNeural
Gender: Male

Name: bg-BG-KalinaNeural
Gender: Female

Name: bn-BD-NabanitaNeural
Gender: Female

Name: bn-BD-PradeepNeural
Gender: Male

Name: bn-IN-BashkarNeural
Gender: Male

Name: bn-IN-TanishaaNeural
Gender: Female

Name: bs-BA-GoranNeural
Gender: Male

Name: bs-BA-VesnaNeural
Gender: Female

Name: ca-ES-EnricNeural
Gender: Male

Name: ca-ES-JoanaNeural
Gender: Female

Name: cs-CZ-AntoninNeural
Gender: Male

Name: cs-CZ-VlastaNeural
Gender: Female

Name: cy-GB-AledNeural
Gender: Male

Name: cy-GB-NiaNeural
Gender: Female

Name: da-DK-ChristelNeural
Gender: Female

Name: da-DK-JeppeNeural
Gender: Male

Name: de-AT-IngridNeural
Gender: Female

Name: de-AT-JonasNeural
Gender: Male

Name: de-CH-JanNeural
Gender: Male

Name: de-CH-LeniNeural
Gender: Female

Name: de-DE-AmalaNeural
Gender: Female

Name: de-DE-ConradNeural
Gender: Male

Name: de-DE-FlorianMultilingualNeural
Gender: Male

Name: de-DE-KatjaNeural
Gender: Female

Name: de-DE-KillianNeural
Gender: Male

Name: de-DE-SeraphinaMultilingualNeural
Gender: Female

Name: el-GR-AthinaNeural
Gender: Female

Name: el-GR-NestorasNeural
Gender: Male

Name: en-AU-NatashaNeural
Gender: Female

Name: en-AU-WilliamNeural
Gender: Male

Name: en-CA-ClaraNeural
Gender: Female

Name: en-CA-LiamNeural
Gender: Male

Name: en-GB-LibbyNeural
Gender: Female

Name: en-GB-MaisieNeural
Gender: Female

Name: en-GB-RyanNeural
Gender: Male

Name: en-GB-SoniaNeural
Gender: Female

Name: en-GB-ThomasNeural
Gender: Male

Name: en-HK-SamNeural
Gender: Male

Name: en-HK-YanNeural
Gender: Female

Name: en-IE-ConnorNeural
Gender: Male

Name: en-IE-EmilyNeural
Gender: Female

Name: en-IN-NeerjaExpressiveNeural
Gender: Female

Name: en-IN-NeerjaNeural
Gender: Female

Name: en-IN-PrabhatNeural
Gender: Male

Name: en-KE-AsiliaNeural
Gender: Female

Name: en-KE-ChilembaNeural
Gender: Male

Name: en-NG-AbeoNeural
Gender: Male

Name: en-NG-EzinneNeural
Gender: Female

Name: en-NZ-MitchellNeural
Gender: Male

Name: en-NZ-MollyNeural
Gender: Female

Name: en-PH-JamesNeural
Gender: Male

Name: en-PH-RosaNeural
Gender: Female

Name: en-SG-LunaNeural
Gender: Female

Name: en-SG-WayneNeural
Gender: Male

Name: en-TZ-ElimuNeural
Gender: Male

Name: en-TZ-ImaniNeural
Gender: Female

Name: en-US-AnaNeural
Gender: Female

Name: en-US-AndrewMultilingualNeural
Gender: Male

Name: en-US-AndrewNeural
Gender: Male

Name: en-US-AriaNeural
Gender: Female

Name: en-US-AvaMultilingualNeural
Gender: Female

Name: en-US-AvaNeural
Gender: Female

Name: en-US-BrianMultilingualNeural
Gender: Male

Name: en-US-BrianNeural
Gender: Male

Name: en-US-ChristopherNeural
Gender: Male

Name: en-US-EmmaMultilingualNeural
Gender: Female

Name: en-US-EmmaNeural
Gender: Female

Name: en-US-EricNeural
Gender: Male

Name: en-US-GuyNeural
Gender: Male

Name: en-US-JennyNeural
Gender: Female

Name: en-US-MichelleNeural
Gender: Female

Name: en-US-RogerNeural
Gender: Male

Name: en-US-SteffanNeural
Gender: Male

Name: en-ZA-LeahNeural
Gender: Female

Name: en-ZA-LukeNeural
Gender: Male

Name: es-AR-ElenaNeural
Gender: Female

Name: es-AR-TomasNeural
Gender: Male

Name: es-BO-MarceloNeural
Gender: Male

Name: es-BO-SofiaNeural
Gender: Female

Name: es-CL-CatalinaNeural
Gender: Female

Name: es-CL-LorenzoNeural
Gender: Male

Name: es-CO-GonzaloNeural
Gender: Male

Name: es-CO-SalomeNeural
Gender: Female

Name: es-CR-JuanNeural
Gender: Male

Name: es-CR-MariaNeural
Gender: Female

Name: es-CU-BelkysNeural
Gender: Female

Name: es-CU-ManuelNeural
Gender: Male

Name: es-DO-EmilioNeural
Gender: Male

Name: es-DO-RamonaNeural
Gender: Female

Name: es-EC-AndreaNeural
Gender: Female

Name: es-EC-LuisNeural
Gender: Male

Name: es-ES-AlvaroNeural
Gender: Male

Name: es-ES-ElviraNeural
Gender: Female

Name: es-ES-XimenaNeural
Gender: Female

Name: es-GQ-JavierNeural
Gender: Male

Name: es-GQ-TeresaNeural
Gender: Female

Name: es-GT-AndresNeural
Gender: Male

Name: es-GT-MartaNeural
Gender: Female

Name: es-HN-CarlosNeural
Gender: Male

Name: es-HN-KarlaNeural
Gender: Female

Name: es-MX-DaliaNeural
Gender: Female

Name: es-MX-JorgeNeural
Gender: Male

Name: es-NI-FedericoNeural
Gender: Male

Name: es-NI-YolandaNeural
Gender: Female

Name: es-PA-MargaritaNeural
Gender: Female

Name: es-PA-RobertoNeural
Gender: Male

Name: es-PE-AlexNeural
Gender: Male

Name: es-PE-CamilaNeural
Gender: Female

Name: es-PR-KarinaNeural
Gender: Female

Name: es-PR-VictorNeural
Gender: Male

Name: es-PY-MarioNeural
Gender: Male

Name: es-PY-TaniaNeural
Gender: Female

Name: es-SV-LorenaNeural
Gender: Female

Name: es-SV-RodrigoNeural
Gender: Male

Name: es-US-AlonsoNeural
Gender: Male

Name: es-US-PalomaNeural
Gender: Female

Name: es-UY-MateoNeural
Gender: Male

Name: es-UY-ValentinaNeural
Gender: Female

Name: es-VE-PaolaNeural
Gender: Female

Name: es-VE-SebastianNeural
Gender: Male

Name: et-EE-AnuNeural
Gender: Female

Name: et-EE-KertNeural
Gender: Male

Name: fa-IR-DilaraNeural
Gender: Female

Name: fa-IR-FaridNeural
Gender: Male

Name: fi-FI-HarriNeural
Gender: Male

Name: fi-FI-NooraNeural
Gender: Female

Name: fil-PH-AngeloNeural
Gender: Male

Name: fil-PH-BlessicaNeural
Gender: Female

Name: fr-BE-CharlineNeural
Gender: Female

Name: fr-BE-GerardNeural
Gender: Male

Name: fr-CA-AntoineNeural
Gender: Male

Name: fr-CA-JeanNeural
Gender: Male

Name: fr-CA-SylvieNeural
Gender: Female

Name: fr-CA-ThierryNeural
Gender: Male

Name: fr-CH-ArianeNeural
Gender: Female

Name: fr-CH-FabriceNeural
Gender: Male

Name: fr-FR-DeniseNeural
Gender: Female

Name: fr-FR-EloiseNeural
Gender: Female

Name: fr-FR-HenriNeural
Gender: Male

Name: fr-FR-RemyMultilingualNeural
Gender: Male

Name: fr-FR-VivienneMultilingualNeural
Gender: Female

Name: ga-IE-ColmNeural
Gender: Male

Name: ga-IE-OrlaNeural
Gender: Female

Name: gl-ES-RoiNeural
Gender: Male

Name: gl-ES-SabelaNeural
Gender: Female

Name: gu-IN-DhwaniNeural
Gender: Female

Name: gu-IN-NiranjanNeural
Gender: Male

Name: he-IL-AvriNeural
Gender: Male

Name: he-IL-HilaNeural
Gender: Female

Name: hi-IN-MadhurNeural
Gender: Male

Name: hi-IN-SwaraNeural
Gender: Female

Name: hr-HR-GabrijelaNeural
Gender: Female

Name: hr-HR-SreckoNeural
Gender: Male

Name: hu-HU-NoemiNeural
Gender: Female

Name: hu-HU-TamasNeural
Gender: Male

Name: id-ID-ArdiNeural
Gender: Male

Name: id-ID-GadisNeural
Gender: Female

Name: is-IS-GudrunNeural
Gender: Female

Name: is-IS-GunnarNeural
Gender: Male

Name: it-IT-DiegoNeural
Gender: Male

Name: it-IT-ElsaNeural
Gender: Female

Name: it-IT-GiuseppeMultilingualNeural
Gender: Male

Name: it-IT-IsabellaNeural
Gender: Female

Name: iu-Cans-CA-SiqiniqNeural
Gender: Female

Name: iu-Cans-CA-TaqqiqNeural
Gender: Male

Name: iu-Latn-CA-SiqiniqNeural
Gender: Female

Name: iu-Latn-CA-TaqqiqNeural
Gender: Male

Name: ja-JP-KeitaNeural
Gender: Male

Name: ja-JP-NanamiNeural
Gender: Female

Name: jv-ID-DimasNeural
Gender: Male

Name: jv-ID-SitiNeural
Gender: Female

Name: ka-GE-EkaNeural
Gender: Female

Name: ka-GE-GiorgiNeural
Gender: Male

Name: kk-KZ-AigulNeural
Gender: Female

Name: kk-KZ-DauletNeural
Gender: Male

Name: km-KH-PisethNeural
Gender: Male

Name: km-KH-SreymomNeural
Gender: Female

Name: kn-IN-GaganNeural
Gender: Male

Name: kn-IN-SapnaNeural
Gender: Female

Name: ko-KR-HyunsuMultilingualNeural
Gender: Male

Name: ko-KR-InJoonNeural
Gender: Male

Name: ko-KR-SunHiNeural
Gender: Female

Name: lo-LA-ChanthavongNeural
Gender: Male

Name: lo-LA-KeomanyNeural
Gender: Female

Name: lt-LT-LeonasNeural
Gender: Male

Name: lt-LT-OnaNeural
Gender: Female

Name: lv-LV-EveritaNeural
Gender: Female

Name: lv-LV-NilsNeural
Gender: Male

Name: mk-MK-AleksandarNeural
Gender: Male

Name: mk-MK-MarijaNeural
Gender: Female

Name: ml-IN-MidhunNeural
Gender: Male

Name: ml-IN-SobhanaNeural
Gender: Female

Name: mn-MN-BataaNeural
Gender: Male

Name: mn-MN-YesuiNeural
Gender: Female

Name: mr-IN-AarohiNeural
Gender: Female

Name: mr-IN-ManoharNeural
Gender: Male

Name: ms-MY-OsmanNeural
Gender: Male

Name: ms-MY-YasminNeural
Gender: Female

Name: mt-MT-GraceNeural
Gender: Female

Name: mt-MT-JosephNeural
Gender: Male

Name: my-MM-NilarNeural
Gender: Female

Name: my-MM-ThihaNeural
Gender: Male

Name: nb-NO-FinnNeural
Gender: Male

Name: nb-NO-PernilleNeural
Gender: Female

Name: ne-NP-HemkalaNeural
Gender: Female

Name: ne-NP-SagarNeural
Gender: Male

Name: nl-BE-ArnaudNeural
Gender: Male

Name: nl-BE-DenaNeural
Gender: Female

Name: nl-NL-ColetteNeural
Gender: Female

Name: nl-NL-FennaNeural
Gender: Female

Name: nl-NL-MaartenNeural
Gender: Male

Name: pl-PL-MarekNeural
Gender: Male

Name: pl-PL-ZofiaNeural
Gender: Female

Name: ps-AF-GulNawazNeural
Gender: Male

Name: ps-AF-LatifaNeural
Gender: Female

Name: pt-BR-AntonioNeural
Gender: Male

Name: pt-BR-FranciscaNeural
Gender: Female

Name: pt-BR-ThalitaMultilingualNeural
Gender: Female

Name: pt-PT-DuarteNeural
Gender: Male

Name: pt-PT-RaquelNeural
Gender: Female

Name: ro-RO-AlinaNeural
Gender: Female

Name: ro-RO-EmilNeural
Gender: Male

Name: ru-RU-DmitryNeural
Gender: Male

Name: ru-RU-SvetlanaNeural
Gender: Female

Name: si-LK-SameeraNeural
Gender: Male

Name: si-LK-ThiliniNeural
Gender: Female

Name: sk-SK-LukasNeural
Gender: Male

Name: sk-SK-ViktoriaNeural
Gender: Female

Name: sl-SI-PetraNeural
Gender: Female

Name: sl-SI-RokNeural
Gender: Male

Name: so-SO-MuuseNeural
Gender: Male

Name: so-SO-UbaxNeural
Gender: Female

Name: sq-AL-AnilaNeural
Gender: Female

Name: sq-AL-IlirNeural
Gender: Male

Name: sr-RS-NicholasNeural
Gender: Male

Name: sr-RS-SophieNeural
Gender: Female

Name: su-ID-JajangNeural
Gender: Male

Name: su-ID-TutiNeural
Gender: Female

Name: sv-SE-MattiasNeural
Gender: Male

Name: sv-SE-SofieNeural
Gender: Female

Name: sw-KE-RafikiNeural
Gender: Male

Name: sw-KE-ZuriNeural
Gender: Female

Name: sw-TZ-DaudiNeural
Gender: Male

Name: sw-TZ-RehemaNeural
Gender: Female

Name: ta-IN-PallaviNeural
Gender: Female

Name: ta-IN-ValluvarNeural
Gender: Male

Name: ta-LK-KumarNeural
Gender: Male

Name: ta-LK-SaranyaNeural
Gender: Female

Name: ta-MY-KaniNeural
Gender: Female

Name: ta-MY-SuryaNeural
Gender: Male

Name: ta-SG-AnbuNeural
Gender: Male

Name: ta-SG-VenbaNeural
Gender: Female

Name: te-IN-MohanNeural
Gender: Male

Name: te-IN-ShrutiNeural
Gender: Female

Name: th-TH-NiwatNeural
Gender: Male

Name: th-TH-PremwadeeNeural
Gender: Female

Name: tr-TR-AhmetNeural
Gender: Male

Name: tr-TR-EmelNeural
Gender: Female

Name: uk-UA-OstapNeural
Gender: Male

Name: uk-UA-PolinaNeural
Gender: Female

Name: ur-IN-GulNeural
Gender: Female

Name: ur-IN-SalmanNeural
Gender: Male

Name: ur-PK-AsadNeural
Gender: Male

Name: ur-PK-UzmaNeural
Gender: Female

Name: uz-UZ-MadinaNeural
Gender: Female

Name: uz-UZ-SardorNeural
Gender: Male

Name: vi-VN-HoaiMyNeural
Gender: Female

Name: vi-VN-NamMinhNeural
Gender: Male

Name: zh-CN-XiaoxiaoNeural
Gender: Female

Name: zh-CN-XiaoyiNeural
Gender: Female

Name: zh-CN-YunjianNeural
Gender: Male

Name: zh-CN-YunxiNeural
Gender: Male

Name: zh-CN-YunxiaNeural
Gender: Male

Name: zh-CN-YunyangNeural
Gender: Male

Name: zh-CN-liaoning-XiaobeiNeural
Gender: Female

Name: zh-CN-shaanxi-XiaoniNeural
Gender: Female

Name: zh-HK-HiuGaaiNeural
Gender: Female

Name: zh-HK-HiuMaanNeural
Gender: Female

Name: zh-HK-WanLungNeural
Gender: Male

Name: zh-TW-HsiaoChenNeural
Gender: Female

Name: zh-TW-HsiaoYuNeural
Gender: Female

Name: zh-TW-YunJheNeural
Gender: Male

Name: zu-ZA-ThandoNeural
Gender: Female

Name: zu-ZA-ThembaNeural
Gender: Male


Name: en-US-AvaMultilingualNeural-V2
Gender: Female

Name: en-US-AndrewMultilingualNeural-V2
Gender: Male

Name: en-US-EmmaMultilingualNeural-V2
Gender: Female

Name: en-US-BrianMultilingualNeural-V2
Gender: Male

Name: de-DE-FlorianMultilingualNeural-V2
Gender: Male

Name: de-DE-SeraphinaMultilingualNeural-V2
Gender: Female

Name: fr-FR-RemyMultilingualNeural-V2
Gender: Male

Name: fr-FR-VivienneMultilingualNeural-V2
Gender: Female

Name: zh-CN-XiaoxiaoMultilingualNeural-V2
Gender: Female
    """.strip()
    voices = []
    # Define regular expression pattern to match Name and Gender lines
    pattern = re.compile(r"Name:\s*(.+)\s*Gender:\s*(.+)\s*", re.MULTILINE)
    # Use regular expression to find all matches
    matches = pattern.findall(azure_voices_str)

    for name, gender in matches:
        # Apply filter conditions
        if filter_locals and any(
            name.lower().startswith(fl.lower()) for fl in filter_locals
        ):
            voices.append(f"{name}-{gender}")
        elif not filter_locals:
            voices.append(f"{name}-{gender}")

    voices.sort()
    return voices


def parse_voice_name(name: str):
    # zh-CN-XiaoyiNeural-Female
    # zh-CN-YunxiNeural-Male
    # zh-CN-XiaoxiaoMultilingualNeural-V2-Female
    name = name.replace("-Female", "").replace("-Male", "").strip()
    return name


def is_azure_v2_voice(voice_name: str):
    voice_name = parse_voice_name(voice_name)
    if voice_name.endswith("-V2"):
        return voice_name.replace("-V2", "").strip()
    return ""


def is_siliconflow_voice(voice_name: str):
    """Check if it's a Silicon Flow voice"""
    return voice_name.startswith("siliconflow:")


def google_gemini_tts(
    text: str, 
    voice_name: str, 
    voice_rate: float = 1.0, 
    voice_file: str = None,
    voice_volume: float = 1.0
) -> Union[SubMaker, None]:
    """
    Generate speech using Google Gemini API
    
    Args:
        text: The text to convert to speech
        voice_name: The voice name, e.g. "natural", "casual", "expressive"
        voice_rate: Speech rate (not directly supported by Gemini API)
        voice_file: Path to the output audio file
        voice_volume: Volume adjustment (not directly supported by Gemini API)
        
    Returns:
        SubMaker object or None on failure
    """
    try:
        api_key = config.google_gemini.get("api_key", "")
        model = config.google_gemini.get("model_name", "gemini-2.5-flash")
        
        if not api_key:
            logger.error("Google Gemini API key not configured")
            return None
            
        if not voice_file:
            voice_file = utils.storage_dir("temp", create=True)
            voice_file = os.path.join(voice_file, f"gemini_tts_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3")
            
        # If voice_name is not provided, use "natural" as default
        if not voice_name:
            voice_name = "natural"
            
        # Base URL for Gemini API
        base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        model_endpoint = f"{model}:generateContent"
        
        # Create API URL with key
        api_url = f"{base_url}/{model_endpoint}?key={api_key}"
        
        # Prepare the payload
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"Generate speech for the following text: {text}"
                }]
            }],
            "generationConfig": {
                "response_modalities": ["AUDIO"],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": voice_name
                        }
                    }
                }
            }
        }
        
        logger.info(f"Calling Google Gemini TTS API with model: {model}, voice: {voice_name}")
        
        # Make the API call
        response = requests.post(
            api_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        if not response.ok:
            logger.error(f"Google Gemini TTS API error: {response.status_code}, {response.text}")
            return None
            
        # Parse the response
        response_data = response.json()
        
        # Extract the audio data
        try:
            audio_data = response_data["candidates"][0]["content"]["parts"][0]["audio_data"]
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract audio data from Gemini response: {e}")
            logger.debug(f"Response structure: {json.dumps(response_data, indent=2)}")
            return None
            
        # Decode the base64 audio data and write to file
        with open(voice_file, "wb") as f:
            f.write(base64.b64decode(audio_data))
            
        logger.info(f"Google Gemini TTS completed, output file: {voice_file}")
        
        # Create a basic SubMaker object for compatibility
        # Note: Gemini doesn't provide word timing information like Azure
        sub_maker = SubMaker()
        # Add a single entry covering the entire audio
        # We'd need to estimate duration, for now just use a placeholder
        # In a real implementation, we could analyze the audio file to get duration
        estimated_duration = len(text) * 100  # Rough estimation of duration in ms
        sub_maker.create_sub((0, estimated_duration), text)
        
        return sub_maker
        
    except Exception as e:
        logger.error(f"Google Gemini TTS error: {str(e)}")
        return None


def get_google_gemini_voices() -> list[str]:
    """
    Get the list of Google Gemini voices
    
    Returns:
        Voice list with format "google-gemini:voice_name-gender"
    """
    # Google Gemini voice list
    voices_with_gender = [
        ("Aoede", "Female"),
        ("Arachne", "Female"),
        ("Calliope", "Female"),
        ("Clio", "Female"),
        ("Erato", "Female"),
        ("Euterpe", "Female"),
        ("Melete", "Female"),
        ("Polyhymnia", "Female"),
        ("Terpsichore", "Female"),
        ("Thalia", "Female"),
        ("Urania", "Female"),
        ("Apollo", "Male"),
        ("Helios", "Male"),
        ("Hyperion", "Male"),
        ("Oceanus", "Male"),
        ("Prometheus", "Male"),
    ]
    
    # Format as display name
    return [
        f"google-gemini:{voice}-{gender}"
        for voice, gender in voices_with_gender
    ]


def is_google_gemini_voice(voice_name: str):
    return voice_name and voice_name.startswith("google-gemini:")


def google_gemini_tts(
    text: str, 
    voice_name: str, 
    voice_rate: float = 1.0, 
    voice_file: str = None,
    voice_volume: float = 1.0
) -> Union[SubMaker, None]:
    """
    Generate speech using Google Gemini TTS
    
    Args:
        text: Text to convert to speech
        voice_name: Voice name (format: "google-gemini:voice_name-gender")
        voice_rate: Voice rate (not used by Gemini, but kept for API compatibility)
        voice_file: Path to save the audio file
        voice_volume: Voice volume (not used by Gemini, but kept for API compatibility)
        
    Returns:
        SubMaker object for subtitle generation
    """
    try:
        # Extract the actual voice name from the format "google-gemini:voice_name-gender"
        voice_parts = voice_name.split(":")
        if len(voice_parts) > 1:
            voice_name = voice_parts[1].split("-")[0]  # Get just the voice name part
        
        logger.info(f"Using Google Gemini TTS with voice: {voice_name}")
        
        # Create a SubMaker instance for subtitle generation
        sub_maker = SubMaker()
        
        # Get the audio file path
        audio_path = asyncio.run(gemini_tts_v1(
            text=text,
            voice_name=voice_name,
            output_path=voice_file
        ))
        
        # Add a dummy entry to the SubMaker for compatibility
        # This is needed because the subtitle generation code expects a SubMaker with offset data
        sub_maker.add(0, len(text), text)
        
        return sub_maker
        
    except Exception as e:
        logger.error(f"Error in Google Gemini TTS: {e}")
        return None


def tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Generate speech from text using different TTS providers
    
    Args:
        text: Text to convert to speech
        voice_name: Voice name
        voice_rate: Voice rate
        voice_file: Path to save the audio file
        voice_volume: Voice volume
        
    Returns:
        SubMaker object for subtitle generation
    """
    # Format text for TTS
    text = _format_text(text)
    
    # Check which TTS provider to use
    if is_siliconflow_voice(voice_name):
        # Silicon Flow TTS
        model, voice = voice_name.split(":")[1].split("/")
        voice = voice.split(":")[1]
        return siliconflow_tts(
            text=text,
            model=model,
            voice=voice,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
        )
    elif is_google_gemini_voice(voice_name):
        # Google Gemini TTS
        return google_gemini_tts(
            text=text,
            voice_name=voice_name,
            voice_rate=voice_rate,
            voice_file=voice_file,
            voice_volume=voice_volume,
        )
    elif is_azure_v2_voice(voice_name):
        # Azure TTS v2
        return azure_tts_v2(text=text, voice_name=voice_name, voice_file=voice_file)
    else:
        # Default to Azure TTS v1
        return azure_tts_v1(
            text=text, voice_name=voice_name, voice_rate=voice_rate, voice_file=voice_file
        )


def convert_rate_to_percent(rate: float) -> str:
    if rate == 1.0:
        return "+0%"
    percent = round((rate - 1.0) * 100)
    if percent > 0:
        return f"+{percent}%"
    else:
        return f"{percent}%"


def azure_tts_v1(
    text: str, voice_name: str, voice_rate: float, voice_file: str
) -> Union[SubMaker, None]:
    voice_name = parse_voice_name(voice_name)
    text = text.strip()
    rate_str = convert_rate_to_percent(voice_rate)
    for i in range(3):
        try:
            logger.info(f"start, voice name: {voice_name}, try: {i + 1}")

            async def _do() -> SubMaker:
                communicate = edge_tts.Communicate(text, voice_name, rate=rate_str)
                sub_maker = edge_tts.SubMaker()
                with open(voice_file, "wb") as file:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            file.write(chunk["data"])
                        elif chunk["type"] == "WordBoundary":
                            sub_maker.create_sub(
                                (chunk["offset"], chunk["duration"]), chunk["text"]
                            )
                return sub_maker

            sub_maker = asyncio.run(_do())
            if not sub_maker or not sub_maker.subs:
                logger.warning("failed, sub_maker is None or sub_maker.subs is None")
                continue

            logger.info(f"completed, output file: {voice_file}")
            return sub_maker
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
    return None


def siliconflow_tts(
    text: str,
    model: str,
    voice: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Use Silicon Flow API to generate speech

    Args:
        text: Text to be converted to speech
        model: Model name, e.g. "FunAudioLLM/CosyVoice2-0.5B"
        voice: Voice name, e.g. "FunAudioLLM/CosyVoice2-0.5B:alex"
        voice_rate: Speech speed, range[0.25, 4.0]
        voice_file: Path to output audio file
        voice_volume: Speech volume, range[0.6, 5.0], need to convert to Silicon Flow gain range [-10, 10]

    Returns:
        SubMaker object or None
    """
    text = text.strip()
    api_key = config.siliconflow.get("api_key", "")

    if not api_key:
        logger.error("SiliconFlow API key is not set")
        return None

    # Convert voice_volume to Silicon Flow gain range
    # Default voice_volume is 1.0, corresponding to gain=0
    gain = voice_volume - 1.0
    # Ensure gain is within [-10, 10] range
    gain = max(-10, min(10, gain))

    url = "https://api.siliconflow.cn/v1/audio/speech"

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "sample_rate": 32000,
        "stream": False,
        "speed": voice_rate,
        "gain": gain,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for i in range(3):  # Try 3 times
        try:
            logger.info(
                f"start siliconflow tts, model: {model}, voice: {voice}, try: {i + 1}"
            )

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                # Save audio file
                with open(voice_file, "wb") as f:
                    f.write(response.content)

                # Create an empty SubMaker object
                sub_maker = SubMaker()

                # Get the actual length of the audio file
                try:
                    # Try to get audio length using moviepy
                    from moviepy import AudioFileClip

                    audio_clip = AudioFileClip(voice_file)
                    audio_duration = audio_clip.duration
                    audio_clip.close()

                    # Convert audio length to 100 nanosecond units (compatible with edge_tts)
                    audio_duration_100ns = int(audio_duration * 10000000)

                    # Use text splitting to create more accurate subtitles
                    # Split text into sentences by punctuation
                    sentences = utils.split_string_by_punctuations(text)

                    if sentences:
                        # Calculate approximate duration for each sentence (allocated by character count proportion)
                        total_chars = sum(len(s) for s in sentences)
                        char_duration = (
                            audio_duration_100ns / total_chars if total_chars > 0 else 0
                        )

                        current_offset = 0
                        for sentence in sentences:
                            if not sentence.strip():
                                continue

                            # Calculate current sentence duration
                            sentence_chars = len(sentence)
                            sentence_duration = int(sentence_chars * char_duration)

                            # Add to SubMaker
                            sub_maker.subs.append(sentence)
                            sub_maker.offset.append(
                                (current_offset, current_offset + sentence_duration)
                            )

                            # Update offset
                            current_offset += sentence_duration
                    else:
                        # If splitting is not possible, use the entire text as one subtitle
                        sub_maker.subs = [text]
                        sub_maker.offset = [(0, audio_duration_100ns)]

                except Exception as e:
                    logger.warning(f"Failed to create accurate subtitles: {str(e)}")
                    # Fall back to simple subtitle
                    sub_maker.subs = [text]
                    # Use the actual length of the audio file, if not available, assume 10 seconds
                    sub_maker.offset = [
                        (
                            0,
                            audio_duration_100ns
                            if "audio_duration_100ns" in locals()
                            else 10000000,
                        )
                    ]

                logger.success(f"siliconflow tts succeeded: {voice_file}")
                print("s", sub_maker.subs, sub_maker.offset)
                return sub_maker
            else:
                logger.error(
                    f"siliconflow tts failed with status code {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"siliconflow tts failed: {str(e)}")

    return None


def azure_tts_v2(text: str, voice_name: str, voice_file: str) -> Union[SubMaker, None]:
    voice_name = is_azure_v2_voice(voice_name)
    if not voice_name:
        logger.error(f"invalid voice name: {voice_name}")
        raise ValueError(f"invalid voice name: {voice_name}")
    text = text.strip()

    def _format_duration_to_offset(duration) -> int:
        if isinstance(duration, str):
            time_obj = datetime.strptime(duration, "%H:%M:%S.%f")
            milliseconds = (
                (time_obj.hour * 3600000)
                + (time_obj.minute * 60000)
                + (time_obj.second * 1000)
                + (time_obj.microsecond // 1000)
            )
            return milliseconds * 10000

        if isinstance(duration, int):
            return duration

        return 0

    for i in range(3):
        try:
            logger.info(f"start, voice name: {voice_name}, try: {i + 1}")

            import azure.cognitiveservices.speech as speechsdk

            sub_maker = SubMaker()

            def speech_synthesizer_word_boundary_cb(evt: speechsdk.SessionEventArgs):
                # print('WordBoundary event:')
                # print('\tBoundaryType: {}'.format(evt.boundary_type))
                # print('\tAudioOffset: {}ms'.format((evt.audio_offset + 5000)))
                # print('\tDuration: {}'.format(evt.duration))
                # print('\tText: {}'.format(evt.text))
                # print('\tTextOffset: {}'.format(evt.text_offset))
                # print('\tWordLength: {}'.format(evt.word_length))

                duration = _format_duration_to_offset(str(evt.duration))
                offset = _format_duration_to_offset(evt.audio_offset)
                sub_maker.subs.append(evt.text)
                sub_maker.offset.append((offset, offset + duration))

            # Creates an instance of a speech config with specified subscription key and service region.
            speech_key = config.azure.get("speech_key", "")
            service_region = config.azure.get("speech_region", "")
            if not speech_key or not service_region:
                logger.error("Azure speech key or region is not set")
                return None

            audio_config = speechsdk.audio.AudioOutputConfig(
                filename=voice_file, use_default_speaker=True
            )
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, region=service_region
            )
            speech_config.speech_synthesis_voice_name = voice_name
            # speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
            #                            value='true')
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
                value="true",
            )

            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
            )
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                audio_config=audio_config, speech_config=speech_config
            )
            speech_synthesizer.synthesis_word_boundary.connect(
                speech_synthesizer_word_boundary_cb
            )

            result = speech_synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.success(f"azure v2 speech synthesis succeeded: {voice_file}")
                return sub_maker
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(
                    f"azure v2 speech synthesis canceled: {cancellation_details.reason}"
                )
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error(
                        f"azure v2 speech synthesis error: {cancellation_details.error_details}"
                    )
            logger.info(f"completed, output file: {voice_file}")
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
    return None


def _format_text(text: str) -> str:
    # text = text.replace("\n", " ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    text = text.strip()
    return text


def create_subtitle(sub_maker: submaker.SubMaker, text: str, subtitle_file: str):
    """
    Optimize subtitle file
    1. Split subtitle file into multiple lines by punctuation
    2. Match each line with text in subtitle file
    3. Generate new subtitle file
    """

    text = _format_text(text)

    def formatter(idx: int, start_time: float, end_time: float, sub_text: str) -> str:
        """
        1
        00:00:00,000 --> 00:00:02,360
        跑步是一项简单易行的运动
        """
        start_t = mktimestamp(start_time).replace(".", ",")
        end_t = mktimestamp(end_time).replace(".", ",")
        return f"{idx}\n{start_t} --> {end_t}\n{sub_text}\n"

    start_time = -1.0
    sub_items = []
    sub_index = 0

    script_lines = utils.split_string_by_punctuations(text)

    def match_line(_sub_line: str, _sub_index: int):
        if len(script_lines) <= _sub_index:
            return ""

        _line = script_lines[_sub_index]
        if _sub_line == _line:
            return script_lines[_sub_index].strip()

        _sub_line_ = re.sub(r"[^\w\s]", "", _sub_line)
        _line_ = re.sub(r"[^\w\s]", "", _line)
        if _sub_line_ == _line_:
            return _line_.strip()

        _sub_line_ = re.sub(r"\W+", "", _sub_line)
        _line_ = re.sub(r"\W+", "", _line)
        if _sub_line_ == _line_:
            return _line.strip()

        return ""

    sub_line = ""

    try:
        for _, (offset, sub) in enumerate(zip(sub_maker.offset, sub_maker.subs)):
            _start_time, end_time = offset
            if start_time < 0:
                start_time = _start_time

            sub = unescape(sub)
            sub_line += sub
            sub_text = match_line(sub_line, sub_index)
            if sub_text:
                sub_index += 1
                line = formatter(
                    idx=sub_index,
                    start_time=start_time,
                    end_time=end_time,
                    sub_text=sub_text,
                )
                sub_items.append(line)
                start_time = -1.0
                sub_line = ""

        if len(sub_items) == len(script_lines):
            with open(subtitle_file, "w", encoding="utf-8") as file:
                file.write("\n".join(sub_items) + "\n")
            try:
                sbs = subtitles.file_to_subtitles(subtitle_file, encoding="utf-8")
                duration = max([tb for ((ta, tb), txt) in sbs])
                logger.info(
                    f"completed, subtitle file created: {subtitle_file}, duration: {duration}"
                )
            except Exception as e:
                logger.error(f"failed, error: {str(e)}")
                os.remove(subtitle_file)
        else:
            logger.warning(
                f"failed, sub_items len: {len(sub_items)}, script_lines len: {len(script_lines)}"
            )

    except Exception as e:
        logger.error(f"failed, error: {str(e)}")


def get_audio_duration(sub_maker: submaker.SubMaker):
    """
    Get audio duration
    """
    if not sub_maker.offset:
        return 0.0
    return sub_maker.offset[-1][1] / 10000000


def get_compatible_voice_for_language(script_language: str, current_voice: str) -> str:
    """
    Get a compatible voice for the given script language
    
    Args:
        script_language: The language code of the script (e.g., 'te-IN', 'hi-IN')
        current_voice: The currently selected voice name
        
    Returns:
        A compatible voice name for the given language, or the current voice if it's compatible
    """
    # Extract the language prefix from the script language code
    language_prefix = script_language.split('-')[0].lower() if script_language and '-' in script_language else ''
    
    # Check if the current voice matches the script language
    if language_prefix and current_voice and script_language.lower() in current_voice.lower():
        # Current voice is already compatible
        return current_voice
    
    # Voice-to-language mapping
    voice_language_map = {
        "te": ["te-IN-ShrutiNeural-Female", "te-IN-MohanNeural-Male"],
        "hi": ["hi-IN-SwaraNeural-Female", "hi-IN-MadhurNeural-Male"],
        "en": ["en-US-AriaNeural-Female", "en-GB-SoniaNeural-Female", "en-US-GuyNeural-Male"],
        "ta": ["ta-IN-PallaviNeural-Female", "ta-IN-ValluvarNeural-Male"],
        "kn": ["kn-IN-SapnaNeural-Female", "kn-IN-GaganNeural-Male"],
        "bn": ["bn-IN-TanishaaNeural-Female", "bn-IN-BashkarNeural-Male"],
        "mr": ["mr-IN-AarohiNeural-Female", "mr-IN-ManoharNeural-Male"],
        "gu": ["gu-IN-DhwaniNeural-Female", "gu-IN-NiranjanNeural-Male"],
        "ml": ["ml-IN-SobhanaNeural-Female", "ml-IN-MidhunNeural-Male"],
        "pa": ["pa-IN-GurleenNeural-Female", "pa-IN-JaskaranNeural-Male"],
        "zh": ["zh-CN-XiaoxiaoNeural-Female", "zh-CN-YunxiNeural-Male"],
        "de": ["de-DE-KatjaNeural-Female", "de-DE-ConradNeural-Male"],
        "fr": ["fr-FR-DeniseNeural-Female", "fr-FR-HenriNeural-Male"],
        "vi": ["vi-VN-HoaiMyNeural-Female", "vi-VN-NamMinhNeural-Male"],
        "th": ["th-TH-AcharaNeural-Female", "th-TH-PremwadeeNeural-Female"],
    }
    
    # If language prefix is found in the mapping, return the first compatible voice
    if language_prefix in voice_language_map:
        compatible_voices = voice_language_map[language_prefix]
        return compatible_voices[0]  # Return the first (female) voice by default
    
    # If no compatible voice found, return the current voice or a default English voice
    if current_voice:
        return current_voice
    
    # Fallback to English
    return "en-US-AriaNeural-Female"


if __name__ == "__main__":
    voice_name = "zh-CN-XiaoxiaoMultilingualNeural-V2-Female"
    voice_name = parse_voice_name(voice_name)
    voice_name = is_azure_v2_voice(voice_name)
    print(voice_name)

    voices = get_all_azure_voices()
    print(len(voices))

    async def _do():
        temp_dir = utils.storage_dir("temp")

        voice_names = [
            "zh-CN-XiaoxiaoMultilingualNeural",
            # Female
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-XiaoyiNeural",
            # Male
            "zh-CN-YunyangNeural",
            "zh-CN-YunxiNeural",
        ]
        text = """
        静夜思是唐代诗人李白创作的一首五言古诗。这首诗描绘了诗人在寂静的夜晚，看到窗前的明月，不禁想起远方的家乡和亲人，表达了他对家乡和亲人的深深思念之情。全诗内容是："床前明月光，疑是地上霜。举头望明月，低头思故乡。"在这短短的四句诗中，诗人通过"明月"和"思故乡"的意象，巧妙地表达了离乡背井人的孤独与哀愁。首句"床前明月光"设景立意，通过明亮的月光引出诗人的遐想；"疑是地上霜"增添了夜晚的寒冷感，加深了诗人的孤寂之情；"举头望明月"和"低头思故乡"则是情感的升华，展现了诗人内心深处的乡愁和对家的渴望。这首诗简洁明快，情感真挚，是中国古典诗歌中非常著名的一首，也深受后人喜爱和推崇。
            """

        text = """
        What is the meaning of life? This question has puzzled philosophers, scientists, and thinkers of all kinds for centuries. Throughout history, various cultures and individuals have come up with their interpretations and beliefs around the purpose of life. Some say it's to seek happiness and self-fulfillment, while others believe it's about contributing to the welfare of others and making a positive impact in the world. Despite the myriad of perspectives, one thing remains clear: the meaning of life is a deeply personal concept that varies from one person to another. It's an existential inquiry that encourages us to reflect on our values, desires, and the essence of our existence.
        """

        text = """
               预计未来3天深圳冷空气活动频繁，未来两天持续阴天有小雨，出门带好雨具；
               10-11日持续阴天有小雨，日温差小，气温在13-17℃之间，体感阴凉；
               12日天气短暂好转，早晚清凉；
                   """

        text = "[Opening scene: A sunny day in a suburban neighborhood. A young boy named Alex, around 8 years old, is playing in his front yard with his loyal dog, Buddy.]\n\n[Camera zooms in on Alex as he throws a ball for Buddy to fetch. Buddy excitedly runs after it and brings it back to Alex.]\n\nAlex: Good boy, Buddy! You're the best dog ever!\n\n[Buddy barks happily and wags his tail.]\n\n[As Alex and Buddy continue playing, a series of potential dangers loom nearby, such as a stray dog approaching, a ball rolling towards the street, and a suspicious-looking stranger walking by.]\n\nAlex: Uh oh, Buddy, look out!\n\n[Buddy senses the danger and immediately springs into action. He barks loudly at the stray dog, scaring it away. Then, he rushes to retrieve the ball before it reaches the street and gently nudges it back towards Alex. Finally, he stands protectively between Alex and the stranger, growling softly to warn them away.]\n\nAlex: Wow, Buddy, you're like my superhero!\n\n[Just as Alex and Buddy are about to head inside, they hear a loud crash from a nearby construction site. They rush over to investigate and find a pile of rubble blocking the path of a kitten trapped underneath.]\n\nAlex: Oh no, Buddy, we have to help!\n\n[Buddy barks in agreement and together they work to carefully move the rubble aside, allowing the kitten to escape unharmed. The kitten gratefully nuzzles against Buddy, who responds with a friendly lick.]\n\nAlex: We did it, Buddy! We saved the day again!\n\n[As Alex and Buddy walk home together, the sun begins to set, casting a warm glow over the neighborhood.]\n\nAlex: Thanks for always being there to watch over me, Buddy. You're not just my dog, you're my best friend.\n\n[Buddy barks happily and nuzzles against Alex as they disappear into the sunset, ready to face whatever adventures tomorrow may bring.]\n\n[End scene.]"

        text = "大家好，我是乔哥，一个想帮你把信用卡全部还清的家伙！\n今天我们要聊的是信用卡的取现功能。\n你是不是也曾经因为一时的资金紧张，而拿着信用卡到ATM机取现？如果是，那你得好好看看这个视频了。\n现在都2024年了，我以为现在不会再有人用信用卡取现功能了。前几天一个粉丝发来一张图片，取现1万。\n信用卡取现有三个弊端。\n一，信用卡取现功能代价可不小。会先收取一个取现手续费，比如这个粉丝，取现1万，按2.5%收取手续费，收取了250元。\n二，信用卡正常消费有最长56天的免息期，但取现不享受免息期。从取现那一天开始，每天按照万5收取利息，这个粉丝用了11天，收取了55元利息。\n三，频繁的取现行为，银行会认为你资金紧张，会被标记为高风险用户，影响你的综合评分和额度。\n那么，如果你资金紧张了，该怎么办呢？\n乔哥给你支一招，用破思机摩擦信用卡，只需要少量的手续费，而且还可以享受最长56天的免息期。\n最后，如果你对玩卡感兴趣，可以找乔哥领取一本《卡神秘籍》，用卡过程中遇到任何疑惑，也欢迎找乔哥交流。\n别忘了，关注乔哥，回复用卡技巧，免费领取《2024用卡技巧》，让我们一起成为用卡高手！"

        text = """
        2023全年业绩速览
公司全年累计实现营业收入1476.94亿元，同比增长19.01%，归母净利润747.34亿元，同比增长19.16%。EPS达到59.49元。第四季度单季，营业收入444.25亿元，同比增长20.26%，环比增长31.86%；归母净利润218.58亿元，同比增长19.33%，环比增长29.37%。这一阶段
的业绩表现不仅突显了公司的增长动力和盈利能力，也反映出公司在竞争激烈的市场环境中保持了良好的发展势头。
2023年Q4业绩速览
第四季度，营业收入贡献主要增长点；销售费用高增致盈利能力承压；税金同比上升27%，扰动净利率表现。
业绩解读
利润方面，2023全年贵州茅台，>归母净利润增速为19%，其中营业收入正贡献18%，营业成本正贡献百分之一，管理费用正贡献百分之一点四。(注：归母净利润增速值=营业收入增速+各科目贡献，展示贡献/拖累的前四名科目，且要求贡献值/净利润增速>15%)
"""
        text = "静夜思是唐代诗人李白创作的一首五言古诗。这首诗描绘了诗人在寂静的夜晚，看到窗前的明月，不禁想起远方的家乡和亲人"

        text = _format_text(text)
        lines = utils.split_string_by_punctuations(text)
        print(lines)

        for voice_name in voice_names:
            voice_file = f"{temp_dir}/tts-{voice_name}.mp3"
            subtitle_file = f"{temp_dir}/tts.mp3.srt"
            sub_maker = azure_tts_v2(
                text=text, voice_name=voice_name, voice_file=voice_file
            )
            create_subtitle(sub_maker=sub_maker, text=text, subtitle_file=subtitle_file)
            audio_duration = get_audio_duration(sub_maker)
            print(f"voice: {voice_name}, audio duration: {audio_duration}s")

    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(_do())
    finally:
        loop.close()
