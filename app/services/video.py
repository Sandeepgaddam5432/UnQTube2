import glob
import itertools
import os
import random
import gc
import shutil
import subprocess
import concurrent.futures
import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
    vfx,
    concatenate_videoclips,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import ImageFont
import math
import re

from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
    VideoResolution,
)
from app.services.utils import video_effects
from app.utils import utils
from .fast_video_assembler import UltraFastVideoAssembler, VideoGenerationOrchestrator, VideoSegment
from .breakthrough_optimizer import BreakthroughOptimizer

# Add module-level constants for frequently used imports to avoid NameErrors
# This provides a fallback mechanism if imports somehow get lost during execution
_IMPORTED_MODULES = {
    'os': os,
    'shutil': shutil,
    'subprocess': subprocess,
    'gc': gc,
    'json': json,
    'random': random,
    'glob': glob
}

# Helper function for resource monitoring
def _log_system_resources(stage=""):
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        logger.info(f"Resource Check [{stage}]: Memory: {memory.percent:.2f}%, CPU: {cpu:.2f}%")
    except ImportError:
        pass # psutil might not be installed, and that's okay

# Defensive function to ensure a module is available
def ensure_module(module_name):
    """Ensure a module is available, return the module or None if not available."""
    if module_name in _IMPORTED_MODULES:
        return _IMPORTED_MODULES[module_name]
    try:
        if module_name == 'os':
            import os as module
        elif module_name == 'shutil':
            import shutil as module
        elif module_name == 'subprocess':
            import subprocess as module
        elif module_name == 'gc':
            import gc as module
        elif module_name == 'json':
            import json as module
        elif module_name == 'random':
            import random as module
        elif module_name == 'glob':
            import glob as module
        else:
            return None
        _IMPORTED_MODULES[module_name] = module
        return module
    except ImportError:
        logger.error(f"Failed to import {module_name}")
        return None

class SubClippedVideoClip:
    def __init__(self, file_path, start_time=None, end_time=None, width=None, height=None, duration=None):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
video_codec = "libx264"
fps = 30

def close_clip(clip):
    if clip is None:
        return
        
    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()
            
        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio
            
        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask
            
        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)
            
        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []
            
    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")
    
    del clip
    gc.collect()

def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]
        
    for file in files:
        try:
            os.remove(file)
        except:
            pass

def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file and os.path.exists(bgm_file):
        return bgm_file

    if bgm_type == "random":
        suffix = "*.mp3"
        song_dir = utils.song_dir()
        files = glob.glob(os.path.join(song_dir, suffix))
        return random.choice(files)

    return ""


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    video_resolution: VideoResolution = VideoResolution.hd_720p,
    max_clip_duration: int = 5,
    threads: int = 2,
    progress_callback = None,
) -> str:
    """
    Combine multiple video clips into a single video with audio
    
    Args:
        combined_video_path: Path to save the combined video
        video_paths: List of video paths to combine
        audio_file: Path to the audio file to add to the video
        video_aspect: Aspect ratio of the video
        video_concat_mode: Mode for concatenating videos
        video_transition_mode: Mode for transitioning between videos
        video_resolution: Resolution of the output video
        max_clip_duration: Maximum duration of each clip
        threads: Number of threads to use for video processing
        progress_callback: Callback function for progress updates
        
    Returns:
        Path to the combined video file
    """
    # Validate inputs
    if not video_paths:
        logger.critical("No video paths provided for combining")
        return combined_video_path
        
    if not os.path.exists(audio_file):
        logger.critical(f"Audio file does not exist: {audio_file}")
        return combined_video_path
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(combined_video_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize variables for cleanup
    audio_clip = None
    preprocessed_clips = []
    video_clip_objects = []
    temp_files_to_delete = []
    final_video = None
    final_video_without_audio = None
    
    try:
        # Load audio file and get duration
        audio_clip = AudioFileClip(audio_file)
        audio_duration = audio_clip.duration
        audio_clip.close()
        audio_clip = None
        
        logger.info(f"Audio duration: {audio_duration} seconds")
        logger.info(f"Maximum clip duration: {max_clip_duration} seconds")
        
        # === START OF HYPER-SPEED WORKER OPTIMIZATION ===
        # Dynamically calculate the optimal number of workers based on CPU cores.
        # We use a multiplier because video processing can be I/O-heavy,
        # so more threads can keep the CPU saturated.
        cpu_cores = os.cpu_count() or 2  # Default to 2 cores if undetectable
        optimal_workers = cpu_cores * 2
        logger.info(f"Dynamically setting max_workers for preprocessing to {optimal_workers} based on {cpu_cores} CPU cores.")
        # === END OF HYPER-SPEED WORKER OPTIMIZATION ===
        
        # Step 1: Preprocess all video clips in parallel using FFmpeg
        logger.info("Starting parallel clip preprocessing with FFmpeg")
        if progress_callback:
            progress_callback(0.1)  # 10% progress for starting preprocessing
        
        preprocessed_clips = preprocess_clips_bulletproof(
            video_paths=video_paths,
            video_aspect=video_aspect,
            video_resolution=video_resolution,
            video_transition_mode=video_transition_mode,
            max_clip_duration=max_clip_duration,
            output_dir=output_dir,
            max_workers=optimal_workers
        )
        
        if not preprocessed_clips:
            logger.critical("No clips were successfully preprocessed! Cannot continue with video generation.")
            return combined_video_path
        
        logger.info(f"Successfully preprocessed {len(preprocessed_clips)} clips")
        if progress_callback:
            progress_callback(0.5)  # 50% progress after preprocessing
        
        # If using random concat mode, shuffle the clips
        if video_concat_mode.value == VideoConcatMode.random.value:
            random.shuffle(preprocessed_clips)
        
        # Filter clips to match audio duration
        total_duration = 0
        final_clips = []
        
        for clip_info in preprocessed_clips:
            if total_duration >= audio_duration:
                break
            final_clips.append(clip_info)
            total_duration += clip_info.duration
        
        if total_duration < audio_duration:
            logger.warning(f"Video duration ({total_duration:.2f}s) is shorter than audio duration ({audio_duration:.2f}s). Adding more clips.")
            # Loop clips to fill the duration
            additional_clips = []
            for clip_info in itertools.cycle(preprocessed_clips):
                if total_duration >= audio_duration:
                    break
                additional_clips.append(clip_info)
                total_duration += clip_info.duration
            final_clips.extend(additional_clips)
        
        if not final_clips:
            logger.critical("No final clips selected for video! Cannot continue.")
            # Store all preprocessed clip paths for later deletion
            temp_files_to_delete.extend([clip.file_path for clip in preprocessed_clips])
            return combined_video_path
            
        logger.info(f"Selected {len(final_clips)} clips for final video (total duration: {total_duration:.2f}s)")
        
        # Step 2: Create the final video using a single concatenation for efficiency
        logger.info("Creating final video from preprocessed clips")
        
        # Store all clip file paths for later deletion
        temp_files_to_delete.extend([clip.file_path for clip in final_clips])
        
        # Load clips in memory one by one
        for i, clip_info in enumerate(final_clips):
            logger.info(f"Loading clip {i+1}/{len(final_clips)}")
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(0.5 + (i / len(final_clips)) * 0.4)  # 50-90% for clip loading
                
            try:
                # Verify the file exists before attempting to load it
                if not os.path.exists(clip_info.file_path) or os.path.getsize(clip_info.file_path) == 0:
                    logger.warning(f"Skipping missing or empty clip file: {clip_info.file_path}")
                    continue
                    
                # Create a clip object with minimal memory footprint
                with VideoFileClip(clip_info.file_path) as clip:
                    # Create subclip that references the original file but doesn't load all frames
                    subclip = clip.without_audio()
                    video_clip_objects.append(subclip)
            except Exception as e:
                logger.error(f"Failed to load clip {i+1}: {str(e)}")
        
        if not video_clip_objects:
            logger.critical("No video clips could be loaded! Cannot create final video.")
            return combined_video_path
        
        # Step 3: Concatenate all clips in a single operation
        logger.info(f"Concatenating {len(video_clip_objects)} clips")
        
        final_video_without_audio = concatenate_videoclips(
            video_clip_objects, 
            method="compose"  # Use compose method for faster concatenation
        )
        
        # Add audio to the video
        audio_clip = AudioFileClip(audio_file)
        final_video = final_video_without_audio.with_audio(audio_clip)
        
        # Write the final video
        if progress_callback:
            progress_callback(0.9)  # 90% after concatenation
            
        # Use hardware acceleration for final render
        gpu_params = utils.get_gpu_acceleration_params()
        logger.info(f"Writing final video to {combined_video_path}")
        final_video.write_videofile(
            combined_video_path,
            temp_audiofile_path=output_dir,
            threads=threads,
            logger=None,
            fps=fps,
            audio_codec=audio_codec,
            codec=video_codec,
            **gpu_params
        )
        
        logger.success("Video combining completed successfully")
        return combined_video_path
        
    except Exception as e:
        logger.critical(f"Critical error creating final video: {str(e)}")
        return combined_video_path
    finally:
        # Clean up all resources, regardless of success or failure
        
        # Close all MoviePy objects
        if video_clip_objects:
            for clip in video_clip_objects:
                close_clip(clip)
                
        if final_video:
            close_clip(final_video)
            
        if final_video_without_audio:
            close_clip(final_video_without_audio)
            
        if audio_clip:
            close_clip(audio_clip)
        
        # Delete all temporary files
        if temp_files_to_delete:
            logger.info(f"Cleaning up {len(temp_files_to_delete)} temporary files")
            delete_files(temp_files_to_delete)
            
        # Force garbage collection
        gc.collect()
        
        if progress_callback:
            progress_callback(1.0)  # 100% when done


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # Create ImageFont
    font = ImageFont.truetype(font, fontsize)

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    processed = True

    _wrapped_lines_ = []
    words = text.split(" ")
    _txt_ = ""
    for word in words:
        _before = _txt_
        _txt_ += f"{word} "
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            if _txt_.strip() == word.strip():
                processed = False
                break
            _wrapped_lines_.append(_before)
            _txt_ = f"{word} "
    _wrapped_lines_.append(_txt_)
    if processed:
        _wrapped_lines_ = [line.strip() for line in _wrapped_lines_]
        result = "\n".join(_wrapped_lines_).strip()
        height = len(_wrapped_lines_) * height
        return result, height

    _wrapped_lines_ = []
    chars = list(text)
    _txt_ = ""
    for word in chars:
        _txt_ += word
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            _wrapped_lines_.append(_txt_)
            _txt_ = ""
    _wrapped_lines_.append(_txt_)
    result = "\n".join(_wrapped_lines_).strip()
    height = len(_wrapped_lines_) * height
    return result, height


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
):
    """
    Generate a video with audio and subtitles
    
    Args:
        video_path: Path to the input video file
        audio_path: Path to the audio file
        subtitle_path: Path to the subtitle file
        output_file: Path to save the output video
        params: Video parameters
        
    Returns:
        None
    """
    # Validate inputs
    if not os.path.exists(video_path):
        logger.critical(f"Video file does not exist: {video_path}")
        return
        
    if not os.path.exists(audio_path):
        logger.critical(f"Audio file does not exist: {audio_path}")
        return
    
    # Ensure required modules are available
    shutil_mod = ensure_module('shutil')
    if not shutil_mod:
        logger.error("shutil module not available, cannot continue")
        return
    
    # Initialize variables for cleanup
    video_clip = None
    audio_clip = None
    bgm_clip = None
    video_with_audio = None
    temp_files_to_delete = []
    
    try:
        aspect = VideoAspect(params.video_aspect)
        video_width, video_height = aspect.to_resolution()

        logger.info(f"Generating final video: {video_width} x {video_height}")
        logger.info(f"  ① video: {video_path}")
        logger.info(f"  ② audio: {audio_path}")
        logger.info(f"  ③ subtitle: {subtitle_path}")
        logger.info(f"  ④ output: {output_file}")

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create intermediate video with audio but no subtitles
        video_with_audio = f"{output_dir}/video_with_audio_temp.mp4"
        temp_files_to_delete.append(video_with_audio)
        
        # Process video with audio first
        video_clip = VideoFileClip(video_path).without_audio()
        audio_clip = AudioFileClip(audio_path).with_effects(
            [afx.MultiplyVolume(params.voice_volume)]
        )

        # Add background music if specified
        bgm_file = get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)
        if bgm_file and os.path.exists(bgm_file):
            try:
                bgm_clip = AudioFileClip(bgm_file).with_effects(
                    [
                        afx.MultiplyVolume(params.bgm_volume),
                        afx.AudioFadeOut(3),
                        afx.AudioLoop(duration=video_clip.duration),
                    ]
                )
                audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
            except Exception as e:
                logger.error(f"Failed to add bgm: {str(e)}")
                # Continue without BGM

        # Add audio to video
        video_clip = video_clip.with_audio(audio_clip)
        
        # Write intermediate video with audio
        gpu_params = utils.get_gpu_acceleration_params()
        logger.info(f"Writing intermediate video with audio to {video_with_audio}")
        video_clip.write_videofile(
            video_with_audio,
            audio_codec=audio_codec,
            temp_audiofile_path=output_dir,
            threads=params.n_threads or 2,
            logger=None,
            fps=fps,
            **gpu_params
        )
        
        # Clean up MoviePy objects early to free memory
        if video_clip:
            close_clip(video_clip)
            video_clip = None
        
        if audio_clip:
            close_clip(audio_clip)
            audio_clip = None
            
        if bgm_clip:
            close_clip(bgm_clip)
            bgm_clip = None
        
        # If no subtitles or subtitle rendering disabled, just rename the file and return
        if not subtitle_path or not os.path.exists(subtitle_path) or not params.subtitle_enabled:
            logger.info("No subtitles to add, using video with audio as final output")
            try:
                shutil_mod.move(video_with_audio, output_file)
                temp_files_to_delete.remove(video_with_audio)  # Removed by move operation
            except Exception as e:
                logger.error(f"Failed to move file: {str(e)}")
                # Try copy as fallback
                try:
                    shutil_mod.copy(video_with_audio, output_file)
                except Exception as copy_error:
                    logger.critical(f"Failed to copy file: {str(copy_error)}")
            return
        
        # Check for FFmpeg availability before attempting subtitle processing
        try:
            ffmpeg_path = shutil_mod.which('ffmpeg')
            if not ffmpeg_path:
                logger.error("FFmpeg not found in PATH. Cannot add subtitles with FFmpeg.")
                logger.warning("Using intermediate video without subtitles as final output")
                try:
                    shutil_mod.move(video_with_audio, output_file)
                    temp_files_to_delete.remove(video_with_audio)  # Removed by move operation
                except Exception as e:
                    logger.error(f"Failed to move file: {str(e)}")
                    # Try copy as fallback
                    try:
                        shutil_mod.copy(video_with_audio, output_file)
                    except Exception as copy_error:
                        logger.critical(f"Failed to copy file: {str(copy_error)}")
                return
        except Exception as e:
            logger.error(f"Error checking for FFmpeg: {str(e)}")
            try:
                shutil_mod.copy(video_with_audio, output_file)
            except Exception as copy_error:
                logger.critical(f"Failed to copy file: {str(copy_error)}")
            return
        
        logger.info(f"Using FFmpeg at: {ffmpeg_path} for subtitle processing")
        
        # Process subtitles with FFmpeg for much better performance
        logger.info("Adding subtitles using FFmpeg for optimal performance")
        
        # Prepare font path - Use appropriate font based on language
        if not params.font_name or params.font_name == "STHeitiMedium.ttc":
            # If the font isn't explicitly set or is using the default Chinese font,
            # automatically select an appropriate font for the language
            if params.video_language:
                # Use our utility function to get the appropriate font for this language
                params.font_name = utils.get_font_for_language(params.video_language, "STHeitiMedium.ttc")
                logger.info(f"Auto-selected font for {params.video_language}: {params.font_name}")
            else:
                params.font_name = "STHeitiMedium.ttc"
                
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if not os.path.exists(font_path):
            logger.warning(f"Font file not found: {font_path}, falling back to default")
            font_path = os.path.join(utils.font_dir(), "STHeitiMedium.ttc")
            if not os.path.exists(font_path):
                logger.error("Default font not found either, using intermediate video without subtitles")
                try:
                    shutil_mod.move(video_with_audio, output_file)
                    temp_files_to_delete.remove(video_with_audio)  # Removed by move operation
                except Exception as e:
                    logger.error(f"Failed to move file: {str(e)}")
                    # Try copy as fallback
                    try:
                        shutil_mod.copy(video_with_audio, output_file)
                    except Exception as copy_error:
                        logger.critical(f"Failed to copy file: {str(copy_error)}")
                return
                
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")
        logger.info(f"Using font: {font_path}")
        
        # Calculate subtitle position based on params
        position = "center" 
        if params.subtitle_position == "bottom":
            position = "10"  # 10% from bottom
        elif params.subtitle_position == "top":
            position = "90"  # 90% from bottom (10% from top)
        elif params.subtitle_position == "custom":
            # Convert percentage to FFmpeg compatible value (0-100)
            position = str(100 - params.custom_position)  # Invert because FFmpeg counts from bottom
        
        # Prepare subtitle style string
        # FontSize is in points, convert from pixels using a rough approximation
        font_size_pts = int(params.font_size * 0.75)  
        
        # Convert hex colors to FFmpeg format if needed
        text_color = params.text_fore_color.lstrip('#')
        if len(text_color) == 6:
            text_color = f"&H{text_color}&"
        
        outline_color = params.stroke_color.lstrip('#')
        if len(outline_color) == 6:
            outline_color = f"&H{outline_color}&"
        
        # Create the style string
        subtitle_style = (
            f"FontName={os.path.basename(font_path)},"
            f"FontSize={font_size_pts},"
            f"PrimaryColour={text_color},"
            f"OutlineColour={outline_color},"
            f"Outline={params.stroke_width},"
            f"Alignment=2,"  # Center align text
            f"BorderStyle=1"  # Outline border
        )
        
        # Use hardware acceleration for the final render
        gpu_params = utils.get_gpu_acceleration_params()
        hw_encoder = gpu_params.get("ffmpeg_params", ["-c:v", "libx264"])
        
        # Build and execute FFmpeg command
        try:
            ffmpeg_cmd = [
                ffmpeg_path, "-y",
                "-i", video_with_audio,
                "-vf", f"subtitles='{subtitle_path}':force_style='{subtitle_style}'",
                "-c:v", hw_encoder[1],  # Video codec from GPU params
                "-preset", "fast",
                "-c:a", "copy",  # Copy audio stream unchanged
                output_file
            ]
            
            logger.info("Running FFmpeg command to burn subtitles")
            subprocess.run(ffmpeg_cmd, check=True)
            logger.success("Subtitles successfully added with FFmpeg")
            
        except Exception as e:
            logger.error(f"Failed to add subtitles with FFmpeg: {str(e)}")
            logger.warning("Falling back to intermediate video without subtitles")
            try:
                shutil_mod.copy(video_with_audio, output_file)
            except Exception as copy_error:
                logger.critical(f"Failed to copy intermediate video: {str(copy_error)}")
    
    except Exception as e:
        logger.critical(f"Critical error generating video: {str(e)}")
        # If we have a valid intermediate video, use it as the output
        if video_with_audio and os.path.exists(video_with_audio) and os.path.getsize(video_with_audio) > 0:
            try:
                logger.warning("Using intermediate video as fallback output due to error")
                if shutil_mod:
                    shutil_mod.copy(video_with_audio, output_file)
                else:
                    # Fallback if shutil is not available
                    with open(video_with_audio, 'rb') as src, open(output_file, 'wb') as dst:
                        dst.write(src.read())
            except Exception as copy_error:
                logger.error(f"Failed to copy intermediate video: {str(copy_error)}")
    
    finally:
        # Clean up MoviePy objects
        if video_clip:
            close_clip(video_clip)
            
        if audio_clip:
            close_clip(audio_clip)
            
        if bgm_clip:
            close_clip(bgm_clip)
        
        # Clean up temporary files
        for temp_file in temp_files_to_delete:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {str(e)}")
        
        # Force garbage collection
        gc.collect()


def preprocess_video(materials: List[MaterialInfo], clip_duration=4):
    for material in materials:
        if not material.url:
            continue

        ext = utils.parse_extension(material.url)
        try:
            clip = VideoFileClip(material.url)
        except Exception:
            clip = ImageClip(material.url)

        width = clip.size[0]
        height = clip.size[1]
        if width < 480 or height < 480:
            logger.warning(f"low resolution material: {width}x{height}, minimum 480x480 required")
            continue

        if ext in const.FILE_TYPE_IMAGES:
            logger.info(f"processing image: {material.url}")
            # Create an image clip and set its duration to 3 seconds
            clip = (
                ImageClip(material.url)
                .with_duration(clip_duration)
                .with_position("center")
            )
            # Apply a zoom effect using the resize method.
            # A lambda function is used to make the zoom effect dynamic over time.
            # The zoom effect starts from the original size and gradually scales up to 120%.
            # t represents the current time, and clip.duration is the total duration of the clip (3 seconds).
            # Note: 1 represents 100% size, so 1.2 represents 120% size.
            zoom_clip = clip.resized(
                lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
            )

            # Optionally, create a composite video clip containing the zoomed clip.
            # This is useful when you want to add other elements to the video.
            final_clip = CompositeVideoClip([zoom_clip])

            # Output the video to a file.
            video_file = f"{material.url}.mp4"
            gpu_params = utils.get_gpu_acceleration_params()
            final_clip.write_videofile(video_file, fps=30, logger=None, **gpu_params)
            close_clip(clip)
            material.url = video_file
            logger.success(f"image processed: {video_file}")
    return materials

def combine_videos_ultra_fast(
    combined_video_path,
    video_paths,
    audio_file,
    video_aspect=VideoAspect.portrait.value,
    video_concat_mode=VideoConcatMode.random.value,
    video_transition_mode=VideoTransitionMode.none.value,
    video_resolution=VideoResolution.hd_720p.value,
    max_clip_duration=5,
    threads=2,
    progress_callback=None,
    target_duration=None,
):
    """
    Combine multiple videos into one using FFmpeg directly for much better performance
    
    Args:
        combined_video_path: Path to save the combined video
        video_paths: List of video paths to combine
        audio_file: Path to the audio file
        video_aspect: Aspect ratio of the video
        video_concat_mode: How to concatenate videos (random or sequential)
        video_transition_mode: Transition effect between videos
        video_resolution: Resolution of the output video
        max_clip_duration: Maximum duration of each clip
        threads: Number of threads to use
        progress_callback: Callback function to report progress
        target_duration: Target duration of the final video in seconds
        
    Returns:
        Path to the combined video
    """
    if not video_paths:
        logger.error("No video paths provided")
        return None

    # Get audio duration
    audio_duration = get_audio_duration(audio_file)
    if audio_duration <= 0:
        logger.error(f"Invalid audio duration: {audio_duration}")
        return None

    # Determine target duration (use audio duration if target_duration is not specified)
    final_duration = target_duration if target_duration else audio_duration
    logger.info(f"Target video duration: {final_duration} seconds")
    
    # Calculate how many clips we need based on target duration
    total_clips_needed = math.ceil(final_duration / max_clip_duration)
    logger.info(f"Need approximately {total_clips_needed} clips for target duration")
    
    # Prepare video paths based on concat mode
    if video_concat_mode == VideoConcatMode.random.value:
        # Shuffle the video paths
        random.shuffle(video_paths)
        
        # If we don't have enough videos, repeat them
        if len(video_paths) < total_clips_needed:
            original_paths = video_paths.copy()
            while len(video_paths) < total_clips_needed:
                random.shuffle(original_paths)
                video_paths.extend(original_paths)
    
    # Limit to the number of clips we need
    video_paths = video_paths[:total_clips_needed]
    
    # Calculate clip duration to match target duration
    adjusted_clip_duration = final_duration / len(video_paths)
    logger.info(f"Adjusted clip duration: {adjusted_clip_duration:.2f} seconds")

    # Create a temporary directory for intermediate files
    temp_dir = os.path.dirname(combined_video_path)
    os.makedirs(temp_dir, exist_ok=True)

    # Get video dimensions based on aspect ratio and resolution
    width, height = VideoAspect(video_aspect).to_resolution()
    resolution_multiplier = VideoResolution(video_resolution).to_multiplier()
    width = int(width * resolution_multiplier)
    height = int(height * resolution_multiplier)

    # Prepare FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg", "-y",  # Overwrite output files without asking
        "-threads", str(threads),  # Use multiple threads
    ]

    # Add input files
    input_files = []
    for i, video_path in enumerate(video_paths):
        ffmpeg_cmd.extend(["-i", video_path])
        input_files.append(f"[{i}:v]")

    # Add audio input
    ffmpeg_cmd.extend(["-i", audio_file])
    audio_input_index = len(video_paths)

    # Build complex filter for scaling, trimming, and concatenating videos
    filter_complex = []
    
    # Scale and trim each video
    for i in range(len(video_paths)):
        # Scale to target dimensions maintaining aspect ratio
        filter_complex.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={adjusted_clip_duration}[v{i}];"
        )
    
    # Concatenate videos
    concat_inputs = "".join([f"[v{i}]" for i in range(len(video_paths))])
    filter_complex.append(f"{concat_inputs}concat=n={len(video_paths)}:v=1:a=0[outv]")
    
    # Add filter complex to command
    ffmpeg_cmd.extend(["-filter_complex", "".join(filter_complex)])
    
    # Map outputs
    ffmpeg_cmd.extend([
        "-map", "[outv]",  # Video output
        "-map", f"{audio_input_index}:a",  # Audio output
    ])
    
    # Set output options
    ffmpeg_cmd.extend([
        "-c:v", "libx264",  # Video codec
        "-preset", "medium",  # Encoding speed/quality tradeoff
        "-crf", "23",  # Quality level (lower is better)
        "-c:a", "aac",  # Audio codec
        "-b:a", "192k",  # Audio bitrate
    ])
    
    # Set target duration if specified
    if target_duration:
        ffmpeg_cmd.extend(["-t", str(target_duration)])
    
    # Output file
    ffmpeg_cmd.append(combined_video_path)
    
    # Execute FFmpeg command
    logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        
        # Process output to track progress
        for line in process.stderr:
            if "time=" in line and progress_callback:
                # Extract time information
                time_match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
                if time_match:
                    hours, minutes, seconds = map(float, time_match.groups())
                    current_time = hours * 3600 + minutes * 60 + seconds
                    progress = min(100, (current_time / final_duration) * 100)
                    progress_callback(progress)
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg command failed with return code {process.returncode}")
            return None
        
        return combined_video_path
        
    except Exception as e:
        logger.error(f"Error running FFmpeg command: {e}")
        return None

def get_audio_duration(audio_file):
    """
    Get the duration of an audio file using ffprobe
    
    Args:
        audio_file: Path to the audio file
        
    Returns:
        Duration in seconds, or -1 if an error occurs
    """
    if not os.path.exists(audio_file):
        logger.error(f"Audio file does not exist: {audio_file}")
        return -1
        
    try:
        cmd = [
            'ffprobe', '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'csv=p=0', 
            audio_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
        else:
            logger.error(f"ffprobe failed: {result.stderr}")
            return -1
    except Exception as e:
        logger.error(f"Error getting audio duration: {e}")
        return -1