import base64
import json
import logging
import requests
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

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
                
                async with self.session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return await self._process_audio_response(response_data, output_path)
                    else:
                        error_text = await response.text()
                        logger.error(f"Gemini TTS API error {response.status}: {error_text}")
                        
                        if response.status == 400:
                            raise ValueError(f"Invalid request: {error_text}")
                        elif response.status == 401:
                            raise ValueError("Invalid API key")
                        elif response.status == 429:
                            # Rate limit, wait and retry
                            await asyncio.sleep(2 ** attempt)
                            continue
                            
            except aiohttp.ClientError as e:
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
        raise ValueError("Gemini API key is required")
    
    config = GeminiTTSConfig(
        api_key=api_key,
        voice_name=voice_name
    )
    
    async with GeminiTTSService(config) as tts_service:
        return await tts_service.generate_speech_async(text, voice_name, output_path)

# Voice preview functionality
async def generate_voice_preview(
    voice_name: str,
    api_key: str,
    sample_text: str = "Hello, this is a voice preview sample."
) -> str:
    """
    Generate a short voice preview for UI
    """
    config = GeminiTTSConfig(
        api_key=api_key,
        voice_name=voice_name
    )
    
    async with GeminiTTSService(config) as tts_service:
        return await tts_service.generate_speech_async(sample_text, voice_name)

# Language compatibility checker
def get_compatible_voice_for_language(script_language: str, current_voice: str) -> str:
    """
    Get a compatible voice for the given script language
    """
    # Voice-to-language mapping
    voice_language_map = {
        "te": ["te-IN-ShrutiNeural", "te-IN-MohanNeural"],
        "hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
        "en": ["en-US-AriaNeural", "en-GB-SoniaNeural"],
        "ta": ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"],
        # Add more mappings as needed
    }
    
    # If current voice is compatible, return it
    if script_language in voice_language_map:
        compatible_voices = voice_language_map[script_language]
        if current_voice in compatible_voices:
            return current_voice
        # Return first compatible voice
        return compatible_voices[0]
    
    # Fallback to English
    return "en-US-AriaNeural"