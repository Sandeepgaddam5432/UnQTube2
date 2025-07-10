# app/services/bulletproof_assembler.py

import subprocess
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BulletproofVideoAssembler:
    """
    Ultra-reliable video assembly using simple FFmpeg commands instead of complex filter_complex
    """
    
    def __init__(self, temp_dir: str = "./temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    def assemble_video_reliable(self, video_clips: List[str], audio_path: str, 
                               subtitle_path: str, output_path: str) -> bool:
        """
        Assemble video using step-by-step approach instead of complex filter_complex
        This is more reliable and easier to debug
        """
        try:
            # Step 1: Concatenate video clips
            concat_video = self.temp_dir / "concat_video.mp4"
            if not self._concatenate_videos(video_clips, str(concat_video)):
                return False
            
            # Step 2: Add audio
            video_with_audio = self.temp_dir / "video_with_audio.mp4"
            if not self._add_audio(str(concat_video), audio_path, str(video_with_audio)):
                return False
            
            # Step 3: Add subtitles
            if not self._add_subtitles(str(video_with_audio), subtitle_path, output_path):
                return False
            
            # Cleanup
            self._cleanup_temp_files([concat_video, video_with_audio])
            
            return True
            
        except Exception as e:
            logger.error(f"Video assembly failed: {str(e)}")
            return False
    
    def _concatenate_videos(self, video_clips: List[str], output_path: str) -> bool:
        """Concatenate video clips using concat demuxer (fastest method)"""
        try:
            # Create concat file
            concat_file = self.temp_dir / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for clip in video_clips:
                    f.write(f"file '{os.path.abspath(clip)}'\n")
            
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',  # Copy streams without re-encoding
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"Video concatenation failed: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Video concatenation timed out")
            return False
        except Exception as e:
            logger.error(f"Video concatenation error: {str(e)}")
            return False
    
    def _add_audio(self, video_path: str, audio_path: str, output_path: str) -> bool:
        """Add audio to video"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Re-encode audio to AAC
                '-shortest',     # Match shortest stream
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            
            if result.returncode != 0:
                logger.error(f"Audio addition failed: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Audio addition timed out")
            return False
        except Exception as e:
            logger.error(f"Audio addition error: {str(e)}")
            return False
    
    def _add_subtitles(self, video_path: str, subtitle_path: str, output_path: str) -> bool:
        """Add subtitles to video"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', subtitle_path,
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'copy',  # Copy audio stream
                '-c:s', 'mov_text',  # Subtitle codec
                '-disposition:s:0', 'default',  # Make subtitles default
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            
            if result.returncode != 0:
                logger.error(f"Subtitle addition failed: {result.stderr}")
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Subtitle addition timed out")
            return False
        except Exception as e:
            logger.error(f"Subtitle addition error: {str(e)}")
            return False
    
    def _cleanup_temp_files(self, files: List[Path]):
        """Clean up temporary files"""
        for file in files:
            try:
                if file.exists():
                    file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup {file}: {str(e)}")
    
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """Get video information for debugging"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format', '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.error(f"FFprobe failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Video info extraction failed: {str(e)}")
            return None

# Usage in your video.py
def combine_videos_ultra_fast(video_clips: List[str], audio_path: str, 
                             subtitle_path: str, output_path: str) -> bool:
    """
    Replacement for your failing combine_videos_ultra_fast function
    """
    assembler = BulletproofVideoAssembler()
    return assembler.assemble_video_reliable(video_clips, audio_path, subtitle_path, output_path) 