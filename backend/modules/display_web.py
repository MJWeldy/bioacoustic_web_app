"""
Web-compatible display module for bioacoustics active learning.
Replaces the Jupyter notebook widgets with web API compatible functions.
"""
import os
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display
import random
import time
import soundfile as sf
import io
import base64
from typing import Dict, List, Tuple, Optional, Union

from modules import config as cfg
from modules import utilities as u

class WebAnnotationInterface:
    """
    Web-compatible annotation interface for audio clips.
    Provides functionality equivalent to the original Jupyter widget interface.
    """
    
    def __init__(self, audio_db, color_mode="viridis", review_mode="random"):
        """
        Initialize the web annotation interface.
        
        Args:
            audio_db: An instance of Audio_DB class containing the clips to annotate
            color_mode: Spectrogram color scheme - "viridis", "gray_r", etc.
            review_mode: Mode for selecting clips - "random" or "top_down"
        """
        self.audio_db = audio_db
        self.color_mode = color_mode
        self.review_mode = review_mode
        
        # State variables
        self.current_index = None
        self.filtered_df = None
        self.sorted_position = -1
        self.viewed_clips = []
        self.current_position = -1
        
        # Top-down mode specific variables
        self.top_down_position = -1
        
        # Validate review mode
        if self.review_mode not in ["random", "top_down"]:
            print(f"Invalid review mode: {self.review_mode}. Using 'random' instead.")
            self.review_mode = "random"

    def create_mel_spectrogram(self, audio_path: str, clip_start: float, clip_end: float) -> str:
        """
        Create mel spectrogram with buffer and markers, return as base64 encoded image.
        
        Args:
            audio_path: Path to audio file
            clip_start: Start time of clip in seconds
            clip_end: End time of clip in seconds
            
        Returns:
            Base64 encoded PNG image data
        """
        # Add buffer of up to 1 second on each side
        buffer_samples = 32000
        buffer_s = buffer_samples / cfg.TARGET_SR

        # Load the full audio file info
        f = sf.SoundFile(audio_path)
        file_duration = f.frames / f.samplerate

        # Calculate buffered indices
        buffered_start = max(0, clip_start - buffer_s)
        buffered_end = min(file_duration, clip_end + buffer_s)
        
        # Extract the buffered audio segment
        y_buffered = u.load_audio(audio_path, (buffered_start, buffered_end, f.samplerate))

        # Create figure
        plt.figure(figsize=(15, 6))

        # Generate mel spectrogram
        nyquist = cfg.MODEL_SR // 2
        fmax = min(cfg.MAX_FREQ, nyquist)

        S = librosa.feature.melspectrogram(
            y=y_buffered, 
            sr=cfg.MODEL_SR,
            n_mels=256,
            fmax=fmax,
            hop_length=128
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Display mel spectrogram
        img = librosa.display.specshow(
            S_dB, 
            sr=cfg.MODEL_SR,
            x_axis='time', 
            y_axis='mel', 
            fmax=fmax, 
            x_coords=np.linspace(buffered_start, buffered_end, S.shape[1]),
            cmap=self.color_mode
        )

        # Add colorbar
        plt.colorbar(img, format='%+2.0f dB')
        plt.xlabel("Time (seconds)")

        # Add vertical lines for clip boundaries
        plt.axvline(x=clip_start, color='r', linestyle='-', linewidth=2, alpha=0.7)
        plt.axvline(x=clip_end, color='r', linestyle='-', linewidth=2, alpha=0.7)

        # Add text labels
        plt.text(clip_start, 0, f"{clip_start:.1f}s", color='r', fontweight='bold', 
                verticalalignment='bottom', horizontalalignment='center')
        plt.text(clip_end, 0, f"{clip_end:.1f}s", color='r', fontweight='bold', 
                verticalalignment='bottom', horizontalalignment='center')

        # Set title
        plt.title(f'Clip: {clip_start:.1f}s - {clip_end:.1f}s (Duration: {clip_end-clip_start:.1f}s)')
        plt.tight_layout(pad=0.5)

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return f"data:image/png;base64,{image_data}"

    def get_filtered_clips(self, score_min: float = 0.0, score_max: float = 1.0, 
                          annotation_filter: Optional[List[int]] = None) -> pl.DataFrame:
        """
        Get filtered clips based on score range and annotation status.
        
        Args:
            score_min: Minimum score threshold
            score_max: Maximum score threshold
            annotation_filter: List of annotation values to include (None for all)
            
        Returns:
            Filtered polars DataFrame
        """
        df = self.audio_db.df
        
        # Apply score filter
        filtered_df = df.filter(
            (pl.col("score") >= score_min) & 
            (pl.col("score") <= score_max)
        )
        
        # Apply annotation filter if specified
        if annotation_filter is not None:
            filtered_df = filtered_df.filter(pl.col("annotation").is_in(annotation_filter))
        else:
            # Default to unreviewed clips
            filtered_df = filtered_df.filter(pl.col("annotation") == 4)
        
        # Sort if in top_down mode
        if self.review_mode == "top_down":
            filtered_df = filtered_df.sort("score", descending=True)
        
        self.filtered_df = filtered_df
        return filtered_df

    def get_next_clip(self) -> Optional[Dict]:
        """
        Get the next clip for annotation based on review mode.
        
        Returns:
            Dictionary with clip information or None if no clips available
        """
        if self.filtered_df is None or len(self.filtered_df) == 0:
            return None
        
        if self.review_mode == "random":
            random_idx = random.randint(0, len(self.filtered_df) - 1)
            clip_row = self.filtered_df.row(random_idx)
            self.current_index = random_idx
        elif self.review_mode == "top_down":
            if self.top_down_position < len(self.filtered_df) - 1:
                self.top_down_position += 1
                clip_row = self.filtered_df.row(self.top_down_position)
                self.current_index = self.top_down_position
            else:
                return None
        
        return self._clip_row_to_dict(clip_row)

    def _clip_row_to_dict(self, clip_row: tuple) -> Dict:
        """Convert clip row tuple to dictionary"""
        return {
            "file_name": clip_row[0],
            "file_path": clip_row[1],
            "duration_sec": clip_row[2],
            "clip_start": clip_row[3],
            "clip_end": clip_row[4],
            "sampling_rate": clip_row[5],
            "score": clip_row[6],
            "annotation": clip_row[7],
            "predictions": clip_row[8],
            "annotation_status": clip_row[9],
            "label_strength": clip_row[10],
            "created_at": clip_row[11],
            "clip_id": f"{clip_row[1]}|{clip_row[3]}|{clip_row[4]}"
        }



    def update_annotation(self, clip_id: str, annotation_value: int) -> bool:
        """
        Update annotation for a specific clip.
        
        Args:
            clip_id: Clip identifier in format "file_path|clip_start|clip_end"
            annotation_value: Annotation value (0: not present, 1: present, 3: uncertain)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Parse clip_id
            parts = clip_id.split("|")
            if len(parts) != 3:
                return False
            
            file_path, clip_start_str, clip_end_str = parts
            clip_start = float(clip_start_str)
            clip_end = float(clip_end_str)
            
            # Update annotation in database
            mask = (
                (self.audio_db.df["file_path"] == file_path) &
                (self.audio_db.df["clip_start"] == clip_start) &
                (self.audio_db.df["clip_end"] == clip_end)
            )
            
            update_col = pl.when(mask).then(annotation_value).otherwise(pl.col("annotation"))
            self.audio_db.df = self.audio_db.df.with_columns(update_col.alias("annotation"))
            
            # Update CDE progress if in CDE review mode
            if self.review_mode == "cde_review" and annotation_value in [0, 1]:
                bin_name = os.path.basename(os.path.dirname(file_path))
                if bin_name in self.bin_progress:
                    self.bin_progress[bin_name] += 1
            
            return True
        except Exception:
            return False

    def get_annotation_stats(self) -> Dict:
        """Get annotation statistics"""
        total = len(self.audio_db.df)
        not_reviewed = len(self.audio_db.df.filter(pl.col("annotation") == 4))
        reviewed = total - not_reviewed
        
        stats = {
            "total_clips": total,
            "reviewed": reviewed,
            "not_reviewed": not_reviewed,
            "review_percentage": (reviewed / total * 100) if total > 0 else 0
        }
        
        if reviewed > 0:
            not_present = len(self.audio_db.df.filter(pl.col("annotation") == 0))
            present = len(self.audio_db.df.filter(pl.col("annotation") == 1))
            uncertain = len(self.audio_db.df.filter(pl.col("annotation") == 3))
            
            stats.update({
                "not_present": not_present,
                "present": present,
                "uncertain": uncertain,
                "not_present_percentage": (not_present / reviewed * 100),
                "present_percentage": (present / reviewed * 100),
                "uncertain_percentage": (uncertain / reviewed * 100)
            })
        
        # CDE review progress
        if self.review_mode == "cde_review" and self.cde_bins:
            cde_progress = []
            for bin_name in sorted(self.cde_bins.keys()):
                definitive = self.bin_progress[bin_name]
                target = min(self.bin_target, self.bin_counts[bin_name])
                complete = definitive >= target
                
                uncertain_clips = self.audio_db.df.filter(
                    (pl.col("file_path").str.contains(f"/{bin_name}/")) & 
                    (pl.col("annotation") == 3)
                )
                uncertain_count = len(uncertain_clips)
                
                cde_progress.append({
                    "bin_name": bin_name,
                    "definitive_annotations": definitive,
                    "target": target,
                    "uncertain_annotations": uncertain_count,
                    "complete": complete,
                    "percentage": (definitive / target * 100) if target > 0 else 0
                })
            
            stats["cde_progress"] = cde_progress
        
        return stats

    def export_audio_clip(self, file_path: str, clip_start: float, clip_end: float) -> bytes:
        """
        Extract audio clip and return as bytes.
        
        Args:
            file_path: Path to audio file
            clip_start: Start time in seconds
            clip_end: End time in seconds
            
        Returns:
            Audio data as bytes (WAV format)
        """
        try:
            # Load audio file
            audio = u.load_audio(file_path, None)
            start_idx = int(clip_start * cfg.TARGET_SR)
            end_idx = int(clip_end * cfg.TARGET_SR)
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, len(audio) - 1))
            end_idx = max(start_idx + 1, min(end_idx, len(audio)))
            
            clip_audio = audio[start_idx:end_idx]
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, clip_audio, cfg.TARGET_SR, format='WAV')
            buffer.seek(0)
            
            return buffer.getvalue()
        except Exception:
            return b""

# Utility functions for web interface
def create_spectrogram_base64(audio_path: str, clip_start: float, clip_end: float, 
                             color_mode: str = "viridis") -> str:
    """
    Create a standalone spectrogram as base64 encoded image.
    
    Args:
        audio_path: Path to audio file
        clip_start: Start time of clip
        clip_end: End time of clip
        color_mode: Color scheme for spectrogram
        
    Returns:
        Base64 encoded PNG image
    """
    interface = WebAnnotationInterface(None, color_mode=color_mode)
    return interface.create_mel_spectrogram(audio_path, clip_start, clip_end)

def get_audio_clip_bytes(file_path: str, clip_start: float, clip_end: float) -> bytes:
    """
    Get audio clip as WAV bytes.
    
    Args:
        file_path: Path to audio file
        clip_start: Start time in seconds
        clip_end: End time in seconds
        
    Returns:
        Audio data as bytes
    """
    interface = WebAnnotationInterface(None)
    return interface.export_audio_clip(file_path, clip_start, clip_end)