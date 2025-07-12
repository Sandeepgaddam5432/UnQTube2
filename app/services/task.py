import math
import os.path
import re
from os import path

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import llm, material, subtitle, video, voice
from app.services import state as sm
from app.utils import utils
from app.services.bulletproof_assembler import BulletproofVideoAssembler
from app.services.progress_estimator import ProgressEstimator, StepEstimate


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    voice_over_script = params.voice_over_script.strip()
    subtitle_script = params.subtitle_script.strip()
    
    if not voice_over_script:
        # Generate script using LLM
        script_result = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
        )
        
        if isinstance(script_result, dict):
            voice_over_script = script_result.get("voice_over_script", "")
            subtitle_script = script_result.get("subtitle_script", "")
        else:
            # Backward compatibility with older LLM service
            voice_over_script = script_result
            subtitle_script = script_result
    else:
        logger.debug(f"Voice-over script: \n{voice_over_script}")
        # If subtitle script is not provided but voice-over script is, use voice-over for both
        if not subtitle_script:
            subtitle_script = voice_over_script
            logger.debug("Using voice-over script for subtitles as well")
        else:
            logger.debug(f"Subtitle script: \n{subtitle_script}")

    if not voice_over_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video script.")
        return None, None

    return voice_over_script, subtitle_script


def generate_terms(task_id, params, voice_over_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        video_terms = llm.generate_terms(
            video_subject=params.video_subject, video_script=voice_over_script, amount=5
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video terms.")
        return None

    return video_terms


def save_script_data(task_id, voice_over_script, subtitle_script, video_terms, params):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    script_data = {
        "voice_over_script": voice_over_script,
        "subtitle_script": subtitle_script,
        "search_terms": video_terms,
        "params": params,
    }

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(utils.to_json(script_data))


def generate_audio(task_id, params, voice_over_script):
    logger.info("\n\n## generating audio")
    audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
    
    # Smart voice auto-correction: check if voice is compatible with language
    voice_name = voice.parse_voice_name(params.voice_name)
    if params.video_language and voice_name:
        compatible_voice = voice.get_compatible_voice_for_language(params.video_language, voice_name)
        if compatible_voice != voice_name:
            logger.warning(f"Voice {voice_name} is not compatible with language {params.video_language}")
            logger.info(f"Auto-correcting to compatible voice: {compatible_voice}")
            voice_name = compatible_voice
    else:
        voice_name = voice.parse_voice_name(params.voice_name)
    
    sub_maker = voice.tts(
        text=voice_over_script,
        voice_name=voice_name,
        voice_rate=params.voice_rate,
        voice_file=audio_file,
    )
    if sub_maker is None:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error(
            """failed to generate audio:
1. check if the language of the voice matches the language of the video script.
2. check if the network is available. If you are in China, it is recommended to use a VPN and enable the global traffic mode.
        """.strip()
        )
        return None, None, None

    audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
    return audio_file, audio_duration, sub_maker


def generate_subtitle(task_id, params, subtitle_script, sub_maker, audio_file):
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    subtitle_fallback = False
    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=subtitle_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            subtitle_fallback = True
            logger.warning("subtitle file not found, fallback to whisper")

    if subtitle_provider == "whisper" or subtitle_fallback:
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=subtitle_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def get_video_materials(task_id, params, video_terms, audio_duration):
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials found, please check the materials and try again."
            )
            return None
        return [material_info.url for material_info in materials]
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=video_terms,
            source=params.video_source,
            video_aspect=params.video_aspect,
            video_contact_mode=params.video_concat_mode,
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
        )
        if not downloaded_videos:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
            )
            return None
        return downloaded_videos


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path
):
    final_video_paths = []
    combined_video_paths = []
    video_concat_mode = (
        params.video_concat_mode if params.video_count == 1 else VideoConcatMode.random
    )
    video_transition_mode = params.video_transition_mode
    video_resolution = params.video_resolution

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video with ULTIMATE architecture: {index} => {combined_video_path}")
        
        # Define progress callback function for video combining
        def progress_callback(progress_value):
            # Calculate overall progress (50% to 75% of total task)
            overall_progress = _progress + (progress_value * 25 / params.video_count)
            sm.state.update_task(task_id, progress=int(overall_progress))
        
        # Use the new ultra-fast video assembly function
        video.combine_videos_ultra_fast(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            video_resolution=video_resolution,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            progress_callback=progress_callback,
            target_duration=params.target_duration,  # Pass target_duration to video assembly
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=int(_progress))

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=int(_progress))

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths


def start(task_id, params: VideoParams, stop_at: str = "video"):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    # 1. Generate script
    voice_over_script, subtitle_script = generate_script(task_id, params)
    if not voice_over_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=voice_over_script
        )
        return {"script": voice_over_script}

    # 2. Generate terms
    video_terms = ""
    if params.video_source != "local":
        video_terms = generate_terms(task_id, params, voice_over_script)
        if not video_terms:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return

    save_script_data(task_id, voice_over_script, subtitle_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": voice_over_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id, params, voice_over_script
    )
    if not audio_file:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, subtitle_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id, params, video_terms, audio_duration
    )
    if not downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 6. Generate final videos using the new bulletproof assembler
    final_video_paths = []
    combined_video_paths = []
    
    # Initialize the new assembler
    temp_dir = path.join(utils.task_dir(task_id), "temp")
    assembler = BulletproofVideoAssembler(temp_dir=temp_dir)
    
    # Define progress steps for the estimator
    steps = [
        StepEstimate("video_concatenation", 20.0, 0.4),
        StepEstimate("audio_addition", 15.0, 0.3),
        StepEstimate("subtitle_addition", 15.0, 0.3)
    ]
    
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video with BULLETPROOF architecture: {index} => {combined_video_path}")
        
        # Create progress estimator
        estimator = ProgressEstimator(steps)
        
        # Define progress callback function for the estimator
        def progress_callback(progress_data):
            # Calculate overall progress (50% to 75% of total task)
            progress_percentage = progress_data["progress_percentage"]
            overall_progress = 50 + (progress_percentage * 0.25 / params.video_count)
            message = f"{progress_data['current_step']} - {progress_data['eta_formatted']} remaining"
            logger.info(f"Progress: {message} ({progress_percentage:.1f}%)")
            sm.state.update_task(task_id, progress=int(overall_progress))
        
        # Set the callback
        estimator.set_progress_callback(progress_callback)
        estimator.start_pipeline()
        
        # Start video concatenation
        estimator.start_step("video_concatenation")
        # Use the bulletproof assembler for reliable video generation
        success = assembler.assemble_video_reliable(
            video_clips=downloaded_videos,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_path=combined_video_path
        )
        estimator.complete_step("video_concatenation")
        
        if not success:
            logger.error(f"Failed to generate video {index}")
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return
            
        final_video_path = combined_video_path  # In this approach, the combined video is the final video
        
        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)
    
    if not final_video_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": voice_over_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )
    return kwargs


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
