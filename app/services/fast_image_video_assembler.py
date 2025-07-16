# Image-to-Video Assembly module with Ken Burns effect
# Creates smooth video transitions from still images with precise duration control

import os
import random
import subprocess
import tempfile
import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from loguru import logger

class ImageVideoAssembler:
    """Creates videos from still images with Ken Burns effect"""
    
    def __init__(self):
        self.logger = logger
    
    def get_image_info(self, image_path: str) -> Dict:
        """Get image info using ffprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', image_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            self.logger.error(f"Failed to get image info for {image_path}: {e}")
        
        return {}
    
    def create_ken_burns_effect(self, 
                               image_path: str, 
                               output_path: str, 
                               duration: float,
                               effect_type: str = 'random') -> bool:
        """
        Create a video clip with Ken Burns effect from a still image
        effect_type can be: 'random', 'zoom_in', 'zoom_out', 'pan_left', 'pan_right', etc.
        """
        # Get image dimensions
        img_info = self.get_image_info(image_path)
        if not img_info or 'streams' not in img_info or not img_info['streams']:
            self.logger.error(f"Failed to get image info for {image_path}")
            return False
        
        try:
            img_width = int(img_info['streams'][0]['width'])
            img_height = int(img_info['streams'][0]['height'])
        except (KeyError, IndexError) as e:
            self.logger.error(f"Invalid image info for {image_path}: {e}")
            return False
        
        # Define Ken Burns effect parameters
        if effect_type == 'random':
            effect_type = random.choice(['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 
                                         'pan_up', 'pan_down', 'diagonal_pan'])
        
        # Set zoom/pan parameters based on effect type
        zoom_start, zoom_end = 1.0, 1.0
        x_start, y_start = 0, 0
        x_end, y_end = 0, 0
        
        if effect_type == 'zoom_in':
            zoom_start, zoom_end = 1.0, 1.3
        elif effect_type == 'zoom_out':
            zoom_start, zoom_end = 1.3, 1.0
        elif effect_type == 'pan_left':
            x_start, x_end = 0.1 * img_width, -0.1 * img_width
        elif effect_type == 'pan_right':
            x_start, x_end = -0.1 * img_width, 0.1 * img_width
        elif effect_type == 'pan_up':
            y_start, y_end = 0.1 * img_height, -0.1 * img_height
        elif effect_type == 'pan_down':
            y_start, y_end = -0.1 * img_height, 0.1 * img_height
        elif effect_type == 'diagonal_pan':
            x_start, x_end = 0.1 * img_width, -0.1 * img_width
            y_start, y_end = 0.1 * img_height, -0.1 * img_height
            zoom_start, zoom_end = 1.0, 1.2
        
        # Build the zoompan filter with added crop filter to ensure even dimensions
        filter_complex = (
            f"zoompan=z='min(zoom+{(zoom_end-zoom_start)/duration}*on,{zoom_end})':"
            f"x='iw/2-(iw/zoom/2)+{x_start}+({x_end-x_start})*on/{duration}':"
            f"y='ih/2-(ih/zoom/2)+{y_start}+({y_end-y_start})*on/{duration}':"
            f"d={int(duration*25)}:s={img_width}x{img_height}:fps=25,"
            f"crop=trunc(iw/2)*2:trunc(ih/2)*2"  # Ensure dimensions are divisible by 2
        )
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-filter_complex', filter_complex,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-tune', 'stillimage',
            '-crf', '23',
            '-t', str(duration),
            '-pix_fmt', 'yuv420p',  # Ensure compatibility across players
            output_path
        ]
        
        try:
            self.logger.debug(f"Running FFmpeg command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg error creating Ken Burns effect: {e.stderr.decode() if e.stderr else str(e)}")
            return False
    
    def create_video_from_images(self,
                                image_paths: List[str],
                                output_path: str,
                                audio_path: Optional[str] = None,
                                target_duration: float = 60.0,
                                progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """
        Create a video from a list of images with Ken Burns effect
        
        Parameters:
        - image_paths: List of image file paths
        - output_path: Path for the final video
        - audio_path: Optional path to an audio file to use as soundtrack
        - target_duration: Target duration for the final video
        - progress_callback: Optional callback function for progress updates
        """
        if not image_paths:
            self.logger.error("No images provided")
            return False
            
        if progress_callback:
            progress_callback("Preparing images for video assembly...", 0.1)
        
        # Create temporary directory for clips
        temp_dir = tempfile.mkdtemp()
        
        # Calculate duration per image based on target duration and number of images
        num_images = len(image_paths)
        duration_per_image = target_duration / num_images
        self.logger.info(f"Creating video with {num_images} images, {duration_per_image:.2f} seconds per image")
        
        # Create video clips with Ken Burns effect for each image
        temp_clips = []
        effect_types = ['zoom_in', 'zoom_out', 'pan_left', 'pan_right', 'diagonal_pan', 'pan_up', 'pan_down']
        
        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress = 0.1 + (0.7 * i / num_images)
                progress_callback(f"Creating video effect for image {i+1}/{num_images}...", progress)
                
            # Select a random effect for each image, ensuring variety
            effect_type = effect_types[i % len(effect_types)]
            
            temp_clip_path = os.path.join(temp_dir, f"clip_{i:03d}.mp4")
            success = self.create_ken_burns_effect(
                image_path=image_path,
                output_path=temp_clip_path,
                duration=duration_per_image,
                effect_type=effect_type
            )
            
            if success:
                temp_clips.append(temp_clip_path)
            else:
                self.logger.warning(f"Failed to create clip for {image_path}, skipping")
        
        if not temp_clips:
            self.logger.error("Failed to create any video clips")
            return False
        
        # Create file list for concatenation
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, 'w') as f:
            for clip in temp_clips:
                f.write(f"file '{clip}'\n")
        
        if progress_callback:
            progress_callback("Assembling final video...", 0.8)
        
        # Build FFmpeg command to concatenate all clips with correctly specified codecs
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-c:v', 'libx264'  # Correctly specify video codec
        ]
        
        # If audio is provided, add it to the command with correct codec specification
        if audio_path and os.path.exists(audio_path):
            cmd.extend([
                '-i', audio_path,
                '-c:a', 'aac',  # Correctly specify audio codec
                '-shortest'  # Ensures video length matches audio length
            ])
        
        # Add output file
        cmd.append(output_path)
        
        try:
            self.logger.debug(f"Running FFmpeg concatenation command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True)
            
            if progress_callback:
                progress_callback("Video assembly completed", 1.0)
            
            # Clean up temporary files
            for clip in temp_clips:
                try:
                    os.remove(clip)
                except:
                    pass
            try:
                os.remove(concat_file)
                os.rmdir(temp_dir)
            except:
                pass
            
            return os.path.exists(output_path)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg error during video assembly: {e.stderr.decode() if e.stderr else str(e)}")
            return False

# Factory function to create and return an instance
def create_image_video_assembler():
    return ImageVideoAssembler() 