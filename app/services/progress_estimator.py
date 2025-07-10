# app/services/progress_estimator.py

import time
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from threading import Thread
import json

@dataclass
class StepEstimate:
    name: str
    estimated_duration: float  # seconds
    weight: float  # 0.0 to 1.0
    
class ProgressEstimator:
    """
    Provides real-time progress updates and ETA calculations
    """
    
    def __init__(self, steps: List[StepEstimate]):
        self.steps = steps
        self.current_step_index = 0
        self.start_time = None
        self.step_start_time = None
        self.progress_callback: Optional[Callable] = None
        self.total_estimated_duration = sum(step.estimated_duration for step in steps)
        
    def set_progress_callback(self, callback: Callable[[Dict], None]):
        """Set callback function for progress updates"""
        self.progress_callback = callback
        
    def start_pipeline(self):
        """Start the progress tracking"""
        self.start_time = time.time()
        self.step_start_time = time.time()
        self._update_progress()
        
    def start_step(self, step_name: str):
        """Mark the beginning of a new step"""
        # Find step index
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                self.current_step_index = i
                break
        
        self.step_start_time = time.time()
        self._update_progress()
        
    def complete_step(self, step_name: str):
        """Mark completion of a step"""
        self.current_step_index += 1
        self._update_progress()
        
    def _update_progress(self):
        """Calculate and broadcast progress update"""
        if not self.start_time:
            return
            
        elapsed = time.time() - self.start_time
        
        # Calculate progress percentage
        completed_weight = sum(step.weight for step in self.steps[:self.current_step_index])
        
        # Add partial progress for current step
        if self.current_step_index < len(self.steps):
            current_step = self.steps[self.current_step_index]
            step_elapsed = time.time() - (self.step_start_time or time.time())
            step_progress = min(step_elapsed / current_step.estimated_duration, 1.0)
            completed_weight += current_step.weight * step_progress
        
        progress_percentage = min(completed_weight * 100, 100)
        
        # Calculate ETA
        if progress_percentage > 0:
            estimated_total_time = elapsed / (progress_percentage / 100)
            eta_seconds = estimated_total_time - elapsed
        else:
            eta_seconds = self.total_estimated_duration
        
        # Format ETA
        eta_formatted = self._format_duration(max(0, eta_seconds))
        
        # Current step info
        current_step_name = ""
        if self.current_step_index < len(self.steps):
            current_step_name = self.steps[self.current_step_index].name
        
        progress_data = {
            "progress_percentage": round(progress_percentage, 1),
            "eta_seconds": round(eta_seconds, 1),
            "eta_formatted": eta_formatted,
            "current_step": current_step_name,
            "elapsed_seconds": round(elapsed, 1),
            "elapsed_formatted": self._format_duration(elapsed)
        }
        
        if self.progress_callback:
            self.progress_callback(progress_data)
            
    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string"""
        if seconds < 60:
            return f"{int(seconds)}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            return f"{minutes}m {remaining_seconds}s"

# Example usage (commented out for production)
"""
# Usage in your task.py
class VideoGenerationTask:
    def __init__(self):
        # Define your pipeline steps with realistic estimates
        self.steps = [
            StepEstimate("subtitle_generation", 10.0, 0.25),
            StepEstimate("video_concatenation", 18.0, 0.35),
            StepEstimate("audio_addition", 12.0, 0.25),
            StepEstimate("subtitle_addition", 10.0, 0.15)
        ]
        
        self.progress_estimator = ProgressEstimator(self.steps)
        
    def set_progress_callback(self, callback):
        # Set callback for frontend updates
        self.progress_estimator.set_progress_callback(callback)
        
    def generate_video(self, video_clips, audio_path, subtitle_text, output_path):
        # Main video generation with progress tracking
        
        self.progress_estimator.start_pipeline()
        
        try:
            # Step 1: Generate subtitles
            self.progress_estimator.start_step("subtitle_generation")
            subtitle_path = self._generate_subtitles_fast(subtitle_text)
            self.progress_estimator.complete_step("subtitle_generation")
            
            # Step 2: Concatenate videos
            self.progress_estimator.start_step("video_concatenation")
            concat_video = self._concatenate_videos(video_clips)
            self.progress_estimator.complete_step("video_concatenation")
            
            # Step 3: Add audio
            self.progress_estimator.start_step("audio_addition")
            video_with_audio = self._add_audio(concat_video, audio_path)
            self.progress_estimator.complete_step("audio_addition")
            
            # Step 4: Add subtitles
            self.progress_estimator.start_step("subtitle_addition")
            final_video = self._add_subtitles(video_with_audio, subtitle_path, output_path)
            self.progress_estimator.complete_step("subtitle_addition")
            
            return True
            
        except Exception as e:
            print(f"Video generation failed: {str(e)}")
            return False

# Frontend WebSocket handler example
def websocket_progress_handler(progress_data):
    # Send progress updates to frontend via WebSocket
    message = {
        "type": "progress_update",
        "data": progress_data
    }
    # Send to your WebSocket connection
    # websocket.send(json.dumps(message))
""" 