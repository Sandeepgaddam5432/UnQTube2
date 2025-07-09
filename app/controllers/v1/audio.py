import os
import tempfile

from fastapi import Request
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel
from typing import Optional

from app.controllers.v1.base import new_router
from app.services import voice
from app.utils import utils

# Create a router for audio-related endpoints
router = new_router()


class AudioPreviewRequest(BaseModel):
    text: str
    voice_name: str
    voice_rate: Optional[float] = 1.0


@router.post(
    "/audio/preview",
    summary="Generate a preview audio clip of the TTS voice",
    response_class=FileResponse,
)
def generate_audio_preview(request: Request, body: AudioPreviewRequest):
    """
    Generates a short audio preview using the specified TTS voice.
    
    Args:
        body: AudioPreviewRequest with text, voice_name, and optional voice_rate
        
    Returns:
        Audio file as a streaming response
    """
    # Limit preview text length for faster processing
    preview_text = body.text[:200] if len(body.text) > 200 else body.text
    if not preview_text:
        preview_text = "This is a preview of the selected voice. How does it sound to you?"
    
    # Create a temporary file for the audio preview
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        temp_audio_path = tmp_file.name
        logger.info(f"Generating audio preview for voice: {body.voice_name}")
        
        # Generate the audio using the voice service
        sub_maker = voice.tts(
            text=preview_text,
            voice_name=voice.parse_voice_name(body.voice_name),
            voice_rate=body.voice_rate,
            voice_file=temp_audio_path
        )
        
        if not sub_maker or not os.path.exists(temp_audio_path):
            logger.error(f"Failed to generate audio preview for voice: {body.voice_name}")
            # Return a default audio file or an error response
            error_audio_path = utils.public_dir("error.mp3")
            if os.path.exists(error_audio_path):
                return FileResponse(
                    error_audio_path,
                    media_type="audio/mpeg",
                    filename="error.mp3"
                )
            else:
                # If no error audio exists, return the failed temp file path
                # This will result in a 404 if the file doesn't exist
                return FileResponse(
                    temp_audio_path, 
                    media_type="audio/mpeg",
                    filename="preview.mp3"
                )
        
        # Return the audio file
        return FileResponse(
            temp_audio_path,
            media_type="audio/mpeg",
            filename="preview.mp3"
        ) 