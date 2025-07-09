import glob
import itertools
import os
import random
import gc
import shutil
import subprocess
import concurrent.futures
from typing import List, Dict, Tuple
from pathlib import Path
from loguru import logger
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
    audio_clip = AudioFileClip(audio_file)
    audio_duration = audio_clip.duration
    audio_clip.close()
    logger.info(f"Audio duration: {audio_duration} seconds")
    logger.info(f"Maximum clip duration: {max_clip_duration} seconds")
    output_dir = os.path.dirname(combined_video_path)
    
    # Determine optimal number of workers based on CPU count
    max_workers = min(os.cpu_count() or 4, 8)  # Limit to 8 workers max
    
    # Step 1: Preprocess all video clips in parallel using FFmpeg
    logger.info("Starting parallel clip preprocessing with FFmpeg")
    if progress_callback:
        progress_callback(0.1)  # 10% progress for starting preprocessing
    
    preprocessed_clips = preprocess_clips_in_parallel(
        video_paths=video_paths,
        video_aspect=video_aspect,
        video_resolution=video_resolution,
        video_transition_mode=video_transition_mode,
        max_clip_duration=max_clip_duration,
        output_dir=output_dir,
        max_workers=max_workers
    )
    
    if not preprocessed_clips:
        logger.error("No clips were successfully preprocessed!")
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
    
    logger.info(f"Selected {len(final_clips)} clips for final video (total duration: {total_duration:.2f}s)")
    
    # Step 2: Create the final video using a single concatenation for efficiency
    logger.info("Creating final video from preprocessed clips")
    
    # Load clips in memory one by one
    video_clip_objects = []
    temp_files_to_delete = []
    
    try:
        # Store all clip file paths for later deletion
        clip_files = [clip.file_path for clip in final_clips]
        temp_files_to_delete.extend(clip_files)
        
        # Process clips with optimal memory usage
        for i, clip_info in enumerate(final_clips):
            logger.info(f"Loading clip {i+1}/{len(final_clips)}")
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback(0.5 + (i / len(final_clips)) * 0.4)  # 50-90% for clip loading
                
            try:
                # Create a clip object with minimal memory footprint
                with VideoFileClip(clip_info.file_path) as clip:
                    # Create subclip that references the original file but doesn't load all frames
                    subclip = clip.without_audio()
                    video_clip_objects.append(subclip)
            except Exception as e:
                logger.error(f"Failed to load clip {i+1}: {str(e)}")
        
        if not video_clip_objects:
            logger.error("No video clips could be loaded!")
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
        
        # Clean up
        for clip in video_clip_objects:
            close_clip(clip)
        close_clip(final_video)
        close_clip(final_video_without_audio)
        close_clip(audio_clip)
        
    except Exception as e:
        logger.error(f"Error creating final video: {str(e)}")
    finally:
        # Delete all temporary files
        delete_files(temp_files_to_delete)
        gc.collect()
    
    if progress_callback:
        progress_callback(1.0)  # 100% when done
    
    logger.success("Video combining completed successfully")
    return combined_video_path


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
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"Generating final video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # Write into the same directory as the output file
    output_dir = os.path.dirname(output_file)
    
    # Create intermediate video with audio but no subtitles
    video_with_audio = f"{output_dir}/video_with_audio_temp.mp4"
    
    # Process video with audio first
    video_clip = VideoFileClip(video_path).without_audio()
    audio_clip = AudioFileClip(audio_path).with_effects(
        [afx.MultiplyVolume(params.voice_volume)]
    )

    # Add background music if specified
    bgm_file = get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)
    if bgm_file:
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
    
    # Clean up MoviePy objects
    close_clip(video_clip)
    close_clip(audio_clip)
    
    # If no subtitles or subtitle rendering disabled, just rename the file and return
    if not subtitle_path or not os.path.exists(subtitle_path) or not params.subtitle_enabled:
        logger.info("No subtitles to add, using video with audio as final output")
        shutil.move(video_with_audio, output_file)
        return
    
    # Process subtitles with FFmpeg for much better performance
    logger.info("Adding subtitles using FFmpeg for optimal performance")
    
    # Prepare font path
    if not params.font_name:
        params.font_name = "STHeitiMedium.ttc"
    font_path = os.path.join(utils.font_dir(), params.font_name)
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
            "ffmpeg", "-y",
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
        shutil.copy(video_with_audio, output_file)
    
    # Clean up temporary files
    try:
        os.remove(video_with_audio)
    except:
        pass


def _preprocess_clip_with_ffmpeg(
    clip_idx: int,
    video_path: str, 
    start_time: float,
    end_time: float,
    output_dir: str,
    target_width: int, 
    target_height: int,
    transition_mode: VideoTransitionMode = None,
) -> str:
    """
    Preprocess a single video clip using direct FFmpeg commands for maximum performance
    
    Args:
        clip_idx: Index of the clip for output filename
        video_path: Source video file
        start_time: Start time for the subclip
        end_time: End time for the subclip
        output_dir: Directory to write the processed clip to
        target_width: Target width for resizing
        target_height: Target height for resizing
        transition_mode: Video transition mode
    
    Returns:
        Path to the preprocessed video file
    """
    try:
        clip_file = f"{output_dir}/ffproc-clip-{clip_idx}.mp4"
        
        # Get source video info using FFprobe
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        
        # Extract source dimensions, defaults to target if ffprobe fails
        source_width, source_height = target_width, target_height
        try:
            import json
            info = json.loads(result.stdout)
            if 'streams' in info and len(info['streams']) > 0:
                stream = info['streams'][0]
                source_width = int(stream.get('width', target_width))
                source_height = int(stream.get('height', target_height))
        except Exception as e:
            logger.warning(f"Failed to get video dimensions, using defaults: {str(e)}")
        
        # Calculate the proper padding for letterboxing/pillarboxing
        source_ratio = source_width / source_height
        target_ratio = target_width / target_height
        
        # Apply FFmpeg filters for scaling and padding
        if abs(source_ratio - target_ratio) < 0.01:
            # Same aspect ratio, just scale
            scale_filter = f"scale={target_width}:{target_height}"
        else:
            # Different aspect ratio, scale and pad
            scale_filter = f"scale='if(gt(dar,{target_width}/{target_height}),min({target_width},iw),-1):if(gt(dar,{target_width}/{target_height}),-1,min({target_height},ih))',pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black"

        # Hardware acceleration parameters
        gpu_params = utils.get_gpu_acceleration_params()
        hw_encoder = gpu_params.get("ffmpeg_params", ["-c:v", "libx264"])
        
        # Base command for video extraction and processing
        ffmpeg_cmd = [
            "ffmpeg", "-y",  # Overwrite output file if it exists
            "-ss", f"{start_time:.3f}",  # Start time
            "-i", video_path,  # Input file
            "-t", f"{end_time - start_time:.3f}",  # Duration
            "-vf", scale_filter,  # Video filters
            "-c:v", hw_encoder[1],  # Video codec (from hw_encoder)
            "-preset", "ultrafast",  # For maximum speed
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",  # Ensure compatibility
            "-an",  # No audio
            clip_file
        ]
        
        # Run FFmpeg
        logger.info(f"Processing clip {clip_idx} with FFmpeg: {os.path.basename(video_path)} ({start_time:.2f}s-{end_time:.2f}s)")
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Verify output file exists and has valid size
        if os.path.exists(clip_file) and os.path.getsize(clip_file) > 0:
            return clip_file
            
    except Exception as e:
        logger.error(f"FFmpeg preprocessing failed for clip {clip_idx}: {str(e)}")
    
    return ""

def preprocess_clips_in_parallel(
    video_paths: List[str],
    video_aspect: VideoAspect,
    video_resolution: VideoResolution,
    video_transition_mode: VideoTransitionMode,
    max_clip_duration: int,
    output_dir: str,
    max_workers: int = 4,
) -> List[SubClippedVideoClip]:
    """
    Preprocess videos using FFmpeg in parallel for much faster processing
    
    Args:
        video_paths: Paths to source video files
        video_aspect: Target aspect ratio
        video_resolution: Target resolution
        video_transition_mode: Video transition mode
        max_clip_duration: Maximum duration for each clip
        output_dir: Directory to save processed clips
        max_workers: Number of parallel workers
        
    Returns:
        List of SubClippedVideoClip objects with paths to preprocessed videos
    """
    # Calculate target resolution
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.portrait:  # 9:16
        base_width, base_height = 720, 1280
    elif aspect == VideoAspect.square:  # 1:1
        base_width, base_height = 720, 720
    else:  # 16:9
        base_width, base_height = 1280, 720
    
    # Apply resolution multiplier
    resolution_multiplier = VideoResolution(video_resolution).to_multiplier()
    target_width = int(base_width * resolution_multiplier)
    target_height = int(base_height * resolution_multiplier)
    
    logger.info(f"Preprocessing {len(video_paths)} videos with FFmpeg at {target_width}x{target_height}")
    
    # Prepare work items for parallel processing
    work_items = []
    clip_idx = 0
    
    for video_path in video_paths:
        try:
            # Get video duration using FFprobe
            ffprobe_cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                video_path
            ]
            result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            clip_duration = float(result.stdout.strip())
            
            start_time = 0
            while start_time < clip_duration:
                end_time = min(start_time + max_clip_duration, clip_duration)
                if end_time - start_time >= 1.0:  # Only include clips of at least 1 second
                    work_items.append({
                        'clip_idx': clip_idx,
                        'video_path': video_path,
                        'start_time': start_time,
                        'end_time': end_time,
                        'output_dir': output_dir,
                        'target_width': target_width,
                        'target_height': target_height,
                        'transition_mode': video_transition_mode
                    })
                    clip_idx += 1
                start_time = end_time
        except Exception as e:
            logger.error(f"Failed to get duration for {video_path}: {str(e)}")
    
    # Process clips in parallel
    logger.info(f"Starting parallel preprocessing of {len(work_items)} video segments with {max_workers} workers")
    processed_clips = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all preprocessing tasks
        future_to_item = {
            executor.submit(
                _preprocess_clip_with_ffmpeg,
                item['clip_idx'],
                item['video_path'], 
                item['start_time'], 
                item['end_time'],
                item['output_dir'], 
                item['target_width'], 
                item['target_height'],
                item['transition_mode']
            ): item for item in work_items
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                processed_clip_path = future.result()
                if processed_clip_path:
                    duration = item['end_time'] - item['start_time']
                    clip_obj = SubClippedVideoClip(
                        file_path=processed_clip_path, 
                        duration=duration,
                        width=item['target_width'], 
                        height=item['target_height']
                    )
                    processed_clips.append(clip_obj)
                    logger.info(f"Successfully preprocessed clip {item['clip_idx']}")
                else:
                    logger.warning(f"Failed to preprocess clip {item['clip_idx']}")
            except Exception as e:
                logger.error(f"Error processing future for clip {item['clip_idx']}: {str(e)}")
    
    logger.success(f"Completed preprocessing {len(processed_clips)}/{len(work_items)} clips")
    return processed_clips


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