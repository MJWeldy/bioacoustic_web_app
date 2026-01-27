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
from matplotlib.ticker import ScalarFormatter
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

def format_time_hms(seconds: float) -> str:
    """
    Format time in seconds as HH:MM:SS.ss

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes}:{secs:05.2f}"

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
        
        # Top 10+score quantiles mode specific variables
        self.quantile_round_clips = []
        self.quantile_round_metadata = []  # Store category info for each clip in round
        self.quantile_position = -1
        self.quantile_round_complete = False
        
        # Validate review mode
        if self.review_mode not in ["random", "top_down", "top_10+score_quantiles", "review_annotated"]:
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

        # Convert stereo to mono if necessary
        if len(y_buffered.shape) > 1:
            y_buffered = np.mean(y_buffered, axis=1)

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
        plt.xlabel("Time (minutes:seconds)")

        # Format x-axis to mm:ss
        from matplotlib.ticker import FuncFormatter
        def format_m_s(x, pos):
            minutes = int(x // 60)
            seconds = int(x % 60)
            return f"{minutes:02d}:{seconds:02d}"
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FuncFormatter(format_m_s))

        # Add vertical lines for clip boundaries
        plt.axvline(x=clip_start, color='r', linestyle='-', linewidth=2, alpha=0.7)
        plt.axvline(x=clip_end, color='r', linestyle='-', linewidth=2, alpha=0.7)

        # Add text labels with HMS format
        plt.text(clip_start, 0, format_time_hms(clip_start), color='r', fontweight='bold',
                verticalalignment='bottom', horizontalalignment='center')
        plt.text(clip_end, 0, format_time_hms(clip_end), color='r', fontweight='bold',
                verticalalignment='bottom', horizontalalignment='center')

        # Set title with HMS format
        duration = clip_end - clip_start
        plt.title(f'Clip: {format_time_hms(clip_start)} - {format_time_hms(clip_end)} (Duration: {duration:.2f}s)')
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
        # Use the backwards compatible df property that has score and annotation columns
        df = self.audio_db.df
        
        # Apply score filter based on confidence predictions
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
        
        # Reset quantile mode state when filters change
        if self.review_mode == "top_10+score_quantiles":
            self.quantile_round_clips = []
            self.quantile_round_metadata = []
            self.quantile_position = -1
            self.quantile_round_complete = False
        
        self.filtered_df = filtered_df
        return filtered_df

    def get_annotated_clips_for_review(self, class_name: str = None) -> pl.DataFrame:
        """
        Get clips that have been annotated, for review mode.

        Args:
            class_name: Optional class name to filter by

        Returns:
            DataFrame of annotated clips with their annotation information
        """
        # Get annotated clips from the database
        annotated_df = self.audio_db.get_annotated_clips_only(class_name)

        # Sort in sequential order for review
        if self.review_mode == "top_down" or self.review_mode == "random":
            # Sort by file, then by clip start time for logical ordering
            annotated_df = annotated_df.sort(["file_path", "clip_start"])

        self.filtered_df = annotated_df
        # Reset position for sequential review
        self.top_down_position = -1

        return annotated_df

    def _generate_quantile_round_clips(self) -> Tuple[List[int], List[Dict]]:
        """
        Generate indices for the top 10+score quantiles review mode.
        Returns up to 50 clips: top 10 highest scoring + 10 clips from each of 4 score quantiles.

        Returns:
            Tuple of (list of clip indices, list of metadata dicts with category info)
        """
        if self.filtered_df is None or len(self.filtered_df) == 0:
            return [], []

        # Get the current class index from the audio database
        class_index = getattr(self.audio_db, '_current_class_index', 0)

        # Get scores and sort indices by score (descending)
        scores_with_indices = []
        for i in range(len(self.filtered_df)):
            # Get confidence prediction scores - assuming they exist in the dataframe
            row = self.filtered_df.row(i)
            clip_dict = dict(zip(self.filtered_df.columns, row))

            # Extract score for the current target class only (not max across all classes)
            confidence_predictions = clip_dict.get('confidence_predictions', [])
            if confidence_predictions and isinstance(confidence_predictions, list):
                # Use the score for the current target class
                if class_index < len(confidence_predictions):
                    target_class_score = confidence_predictions[class_index]
                else:
                    target_class_score = 0.0
            elif confidence_predictions:
                # Single value (shouldn't happen in multiclass, but handle it)
                target_class_score = confidence_predictions
            else:
                target_class_score = 0.0

            scores_with_indices.append((i, target_class_score))

        # Sort by score descending
        scores_with_indices.sort(key=lambda x: x[1], reverse=True)

        round_indices = []
        round_metadata = []

        # 1. Top 10 highest scoring clips
        top_10_indices = [idx for idx, score in scores_with_indices[:10]]
        round_indices.extend(top_10_indices)
        for idx in top_10_indices:
            round_metadata.append({
                "category": "top_10",
                "category_label": "Top 10 Highest Scores"
            })
        print(f"DEBUG Quantiles: Added top 10 clips with scores: {[score for _, score in scores_with_indices[:10]]}")

        # 2. Score quantile ranges: 0-0.5, 0.5-0.75, 0.75-0.875, 0.875-1.0
        score_ranges = [
            (0.0, 0.5, "quantile_0.0_0.5", "Quantile: 0.0-0.5"),
            (0.5, 0.75, "quantile_0.5_0.75", "Quantile: 0.5-0.75"),
            (0.75, 0.875, "quantile_0.75_0.875", "Quantile: 0.75-0.875"),
            (0.875, 1.0, "quantile_0.875_1.0", "Quantile: 0.875-1.0")
        ]

        for min_score, max_score, category, category_label in score_ranges:
            # Find clips in this score range (excluding top 10 already selected)
            range_indices = []
            for idx, score in scores_with_indices:
                if idx not in top_10_indices and min_score <= score <= max_score:
                    range_indices.append(idx)

            # Randomly select up to 10 clips from this range
            if range_indices:
                selected_count = min(10, len(range_indices))
                selected_indices = random.sample(range_indices, selected_count)
                round_indices.extend(selected_indices)
                for idx in selected_indices:
                    round_metadata.append({
                        "category": category,
                        "category_label": category_label
                    })
                scores_for_range = [scores_with_indices[i][1] for i in range(len(scores_with_indices)) if scores_with_indices[i][0] in selected_indices]
                print(f"DEBUG Quantiles: Added {selected_count} clips from range {min_score}-{max_score} with scores: {scores_for_range}")
            else:
                print(f"DEBUG Quantiles: No clips found in range {min_score}-{max_score}")

        print(f"DEBUG Quantiles: Generated round with {len(round_indices)} total clips")
        return round_indices, round_metadata

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
        elif self.review_mode == "top_down" or self.review_mode == "review_annotated":
            if self.top_down_position < len(self.filtered_df) - 1:
                self.top_down_position += 1
                clip_row = self.filtered_df.row(self.top_down_position)
                self.current_index = self.top_down_position
            else:
                return None
        elif self.review_mode == "top_10+score_quantiles":
            # Check if we need to generate a new round
            if not self.quantile_round_clips or self.quantile_position >= len(self.quantile_round_clips) - 1:
                # Generate new round of clips
                self.quantile_round_clips, self.quantile_round_metadata = self._generate_quantile_round_clips()
                self.quantile_position = -1
                self.quantile_round_complete = False
                print(f"DEBUG Quantiles: Generated new round with {len(self.quantile_round_clips)} clips")

                if not self.quantile_round_clips:
                    return None

            # Get next clip from current round
            if self.quantile_position < len(self.quantile_round_clips) - 1:
                self.quantile_position += 1
                clip_index = self.quantile_round_clips[self.quantile_position]
                clip_row = self.filtered_df.row(clip_index)
                self.current_index = clip_index

                # Mark round as complete when we reach the last clip
                if self.quantile_position == len(self.quantile_round_clips) - 1:
                    self.quantile_round_complete = True
                    print(f"DEBUG Quantiles: Completed round ({self.quantile_position + 1}/{len(self.quantile_round_clips)} clips)")
            else:
                return None

        # Convert row to dict using proper column names
        clip_dict = dict(zip(self.filtered_df.columns, clip_row))
        result = self._convert_clip_dict(clip_dict)

        # Add round metadata for quantile mode
        if self.review_mode == "top_10+score_quantiles" and self.quantile_position >= 0:
            result["round_position"] = self.quantile_position + 1
            result["round_total"] = len(self.quantile_round_clips)
            result["round_category"] = self.quantile_round_metadata[self.quantile_position]["category"]
            result["round_category_label"] = self.quantile_round_metadata[self.quantile_position]["category_label"]

        return result


    def _convert_clip_dict(self, clip_dict: Dict) -> Dict:
        """Convert clip dictionary with proper type conversion"""
        def convert_value(value):
            """Convert numpy/polars types to native Python types"""
            if value is None:
                return None
            elif hasattr(value, 'item'):  # numpy scalar
                return float(value.item()) if hasattr(value.item(), '__float__') else value.item()
            elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'item'):  # numpy array elements
                return [v.item() if hasattr(v, 'item') else v for v in value]
            elif hasattr(value, '__float__'):  # Try to convert to float if possible
                return float(value)
            else:
                return value
        
        # Convert all values and use the proper clip_id
        result = {}
        for key, value in clip_dict.items():
            result[key] = convert_value(value)
        
        print(f"DEBUG: Final clip dict clip_id: {result.get('clip_id')}")
        print(f"DEBUG: Final clip dict clip_end: {result.get('clip_end')}, type: {type(result.get('clip_end'))}")
        return result



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