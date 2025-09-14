import polars as pl
import numpy as np
from datetime import datetime
#from sklearn.metrics.pairwise import euclidean_distances
#import json
#import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

from modules import config as cfg

class Audio_DB:
  def __init__(self, embedding_dim: int = 1280, num_classes: int = 1):
      """
      Initialize the audio prediction and embedding database with three-table structure:
      - files: File metadata (filepath, duration, sample_rate, etc.)
      - clips: Clip segments (file_id FK, start_time, end_time, annotation_status, confidence)
      - annotations: Human annotations (clip_id FK, class_name, label)
      
      Args:
          embedding_dim: Dimension size of the embeddings
          num_classes: Number of classes for multiclass prediction
      """
      self.score_min = 0.0
      self.score_max = 1.0
      self.embedding_dim = embedding_dim
      self.num_classes = num_classes
      
      # Files table - stores file metadata
      self.files_df = pl.DataFrame(
        schema={
          'file_id': pl.Utf8,  # Primary key: unique identifier for each file
          'file_name': pl.Utf8,  # Name of the audio file (without path)
          'file_path': pl.Utf8,  # Full path to the audio file
          'duration_sec': pl.Float32,  # Total duration of the audio file
          'sampling_rate': pl.Int32,  # Audio sampling rate
          'created_at': pl.Datetime  # When file was added to database
        }
      )
      
      # Clips table - stores clip segments with analysis state
      self.clips_df = pl.DataFrame(
        schema={
          'clip_id': pl.Utf8,  # Primary key: unique identifier for each clip
          'file_id': pl.Utf8,  # Foreign key to files table
          'clip_start': pl.Float32,  # Start time of the clip in seconds
          'clip_end': pl.Float32,    # End time of the clip in seconds
          'annotation_status': pl.List(pl.Int32),  # Per-class annotation status array
          'confidence_predictions': pl.List(pl.Float32),  # Per-class confidence scores array
          'label_strength': pl.List(pl.Int32),  # Per-class label strength array (0=weak, 1=strong)
          'embedding_index': pl.Int64,  # Index into embeddings array
          'created_at': pl.Datetime  # When clip was created
        }
      )
      
      # Annotations table - stores human labels
      self.annotations_df = pl.DataFrame(
        schema={
          'annotation_id': pl.Utf8,  # Primary key: unique identifier for each annotation
          'clip_id': pl.Utf8,  # Foreign key to clips table
          'class_name': pl.Utf8,  # Name of the class being annotated
          'label': pl.Utf8,  # Label: 'present', 'not_present', 'uncertain'
          'annotated_at': pl.Datetime  # When annotation was made
        }
      )
      
  def add_file_and_clips(self, 
                        file_name: str, 
                        file_path: str, 
                        duration_sec: float, 
                        sampling_rate: int,
                        window_size: float,
                        embedding_start_index: int = None) -> str:
    """
    Add a file and create clips for it based on window size.
    
    Args:
        file_name: Name of the audio file (without path)
        file_path: Full path to the audio file
        duration_sec: Total duration of the audio file in seconds
        sampling_rate: Audio sampling rate in Hz
        window_size: Size of clips to create in seconds
        embedding_start_index: Starting index for embeddings (None if no embeddings yet)
        
    Returns:
        file_id: The unique identifier for the added file
    """
    # Generate unique file ID
    import uuid
    file_id = str(uuid.uuid4())
    
    # Ensure data types
    duration_sec_float32 = np.float32(duration_sec)
    sampling_rate_int32 = np.int32(sampling_rate)
    
    # Add file to files table
    new_file = pl.DataFrame({
        'file_id': [file_id],
        'file_name': [file_name],
        'file_path': [file_path],
        'duration_sec': [duration_sec_float32],
        'sampling_rate': [sampling_rate_int32],
        'created_at': [datetime.now()]
    })
    
    self.files_df = pl.concat([self.files_df, new_file])
    
    # Create clips for this file
    clip_start = 0.0
    embedding_index = embedding_start_index or 0
    
    while clip_start < duration_sec:
        clip_end = min(clip_start + window_size, duration_sec)
        clip_id = str(uuid.uuid4())
        
        # Initialize per-class annotation status (4 = unreviewed), confidence (0.5 = neutral), and label strength (0 = weak)
        initial_annotation_status = [np.int32(4)] * self.num_classes
        initial_confidence = [np.float32(0.5)] * self.num_classes
        initial_label_strength = [np.int32(0)] * self.num_classes
        
        new_clip = pl.DataFrame({
            'clip_id': [clip_id],
            'file_id': [file_id],
            'clip_start': [np.float32(clip_start)],
            'clip_end': [np.float32(clip_end)],
            'annotation_status': [initial_annotation_status],
            'confidence_predictions': [initial_confidence],
            'label_strength': [initial_label_strength],
            'embedding_index': [embedding_index],
            'created_at': [datetime.now()]
        })
        
        self.clips_df = pl.concat([self.clips_df, new_clip])
        
        clip_start += window_size
        embedding_index += 1
    
    return file_id

  def save_db(self, file_path: str) -> None:
    """Save database using three-table structure."""
    from pathlib import Path
    base_path = Path(file_path).parent
    
    # Ensure the directory exists
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Save the three tables
    self.files_df.write_parquet(base_path / "files.parquet")
    self.clips_df.write_parquet(base_path / "clips.parquet") 
    self.annotations_df.write_parquet(base_path / "annotations.parquet")
    
    # Create metadata file
    import json
    metadata = {
        "database_format": "three_table_v1",
        "tables": ["files.parquet", "clips.parquet", "annotations.parquet"],
        "created_at": str(datetime.now()),
        "num_files": len(self.files_df),
        "num_clips": len(self.clips_df), 
        "num_annotations": len(self.annotations_df),
        "num_classes": self.num_classes
    }
    
    with open(base_path / "database_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
  
  def load_db(self, file_path: str) -> None:
    """Load the three-table database format."""
    from pathlib import Path
    base_path = Path(file_path).parent
    
    # Load the three tables
    files_path = base_path / "files.parquet"
    clips_path = base_path / "clips.parquet"
    annotations_path = base_path / "annotations.parquet"
    database_info_path = base_path / "database_info.json"
    
    if files_path.exists() and clips_path.exists() and annotations_path.exists():
        # Load three-table format
        self.files_df = pl.read_parquet(files_path)
        self.clips_df = pl.read_parquet(clips_path)
        self.annotations_df = pl.read_parquet(annotations_path)
        
        # Load metadata if available
        if database_info_path.exists():
            import json
            with open(database_info_path, 'r') as f:
                metadata = json.load(f)
                print(f"✓ Loaded database: {metadata.get('num_files', 0)} files, "
                      f"{metadata.get('num_clips', 0)} clips, "
                      f"{metadata.get('num_annotations', 0)} annotations")
        
    else:
        raise FileNotFoundError(f"Three-table database not found at {base_path}. "
                              f"Expected files: files.parquet, clips.parquet, annotations.parquet")

  
  def populate_scores(self, scores: List[float]):
      """Update confidence predictions for the first class with new scores."""
      if len(scores) != len(self.clips_df):
        raise ValueError(f"Length of scores ({len(scores)}) must match clips count ({len(self.clips_df)})")
      
      if any(score > self.score_max or score < self.score_min for score in scores):
        print(f"Warning: Some scores are outside the expected range [{self.score_min}, {self.score_max}]")
      
      # Update confidence predictions for first class
      confidence_vectors = self.clips_df['confidence_predictions'].to_list()
      for i, score in enumerate(scores):
          if i < len(confidence_vectors) and confidence_vectors[i]:
              confidence_vectors[i][0] = np.float32(score)
      
      self.clips_df = self.clips_df.with_columns(pl.Series("confidence_predictions", confidence_vectors))
  
  def populate_multiclass_predictions(self, predictions: List[List[float]]):
      """
      Populate confidence predictions with multiclass prediction vectors.
      
      Args:
          predictions: List of prediction vectors, one per clip
      """
      if len(predictions) != len(self.clips_df):
          raise ValueError(f"Length of predictions ({len(predictions)}) must match clips count ({len(self.clips_df)})")
      
      # Convert to float32 lists
      predictions_float32 = [[np.float32(score) for score in pred_vec] for pred_vec in predictions]
      self.clips_df = self.clips_df.with_columns(pl.Series("confidence_predictions", predictions_float32))
  
  def populate_embedding_indices(self, embedding_indices: List[int]):
      """
      Populate the embedding_index column for existing clips.
      
      Args:
          embedding_indices: List of embedding indices, one per clip (can contain None for missing embeddings)
      """
      if len(embedding_indices) != len(self.clips_df):
          raise ValueError(f"Length of embedding indices ({len(embedding_indices)}) must match clips count ({len(self.clips_df)})")
      
      self.clips_df = self.clips_df.with_columns(pl.Series("embedding_index", embedding_indices))
  
  def auto_populate_embedding_indices(self):
      """
      Automatically populate embedding indices for existing clips.
      Assumes embeddings are ordered the same as clips in the database (0, 1, 2, ...).
      """
      num_clips = len(self.clips_df)
      indices = list(range(num_clips))
      self.populate_embedding_indices(indices)
      print(f"✓ Populated embedding indices for {num_clips} clips (0 to {num_clips-1})")
  
  def populate_embedding_indices_by_order(self, embeddings_count: int):
      """
      Populate embedding indices based on clip order, handling cases where 
      there might be fewer embeddings than clips.
      
      Args:
          embeddings_count: Number of embeddings available
      """
      num_clips = len(self.clips_df)
      
      if embeddings_count >= num_clips:
          # Enough embeddings for all clips
          indices = list(range(num_clips))
      else:
          # Fewer embeddings than clips - assign to first N clips
          indices = list(range(embeddings_count)) + [None] * (num_clips - embeddings_count)
          print(f"Warning: Only {embeddings_count} embeddings for {num_clips} clips. "
                f"Last {num_clips - embeddings_count} clips will have no embedding.")
      
      self.populate_embedding_indices(indices)
      print(f"✓ Populated embedding indices: {embeddings_count} clips with embeddings, "
            f"{num_clips - min(embeddings_count, num_clips)} without")
  
  def update_class_scores_and_annotations(self, class_index: int):
      """
      Update the score and annotation columns based on a specific class index.
      This method is provided for backwards compatibility but doesn't modify the underlying tables.
      
      Args:
          class_index: Index of the class to extract scores and annotations for
      """
      if class_index >= self.num_classes:
          raise ValueError(f"Class index {class_index} is out of range for {self.num_classes} classes")
      
      # This method is called by Active Learning to set up the current class view
      # The actual score/annotation extraction happens in the df property
      # Just store the current class index for the property to use
      self._current_class_index = class_index
  
  def update_class_annotation(self, clip_mask, class_index: int, annotation_value: int):
      """
      Update annotation for a specific class and clip.
      This method is provided for backwards compatibility and updates the clips table directly.
      
      Args:
          clip_mask: Boolean mask identifying the clip to update (polars Series)
          class_index: Index of the class to update
          annotation_value: New annotation value (0, 1, 3, or 4)
      """
      if class_index >= self.num_classes:
          raise ValueError(f"Class index {class_index} is out of range for {self.num_classes} classes")
      
      # Update annotation status in clips table for the matching rows
      # This is a simplified version that updates the clips table directly
      if hasattr(clip_mask, 'to_list'):
          mask_values = clip_mask.to_list()
      else:
          mask_values = clip_mask
      
      # Update annotation_status arrays for matching clips
      current_annotations = self.clips_df['annotation_status'].to_list()
      current_strengths = self.clips_df['label_strength'].to_list()
      
      for i, should_update in enumerate(mask_values):
          if should_update and i < len(current_annotations):
              if len(current_annotations[i]) > class_index:
                  current_annotations[i][class_index] = annotation_value
                  
                  # Update label strength
                  if annotation_value in [0, 1]:  # Present or Not Present = strong label
                      current_strengths[i][class_index] = 1
                  elif annotation_value == 4:  # Unreviewed = reset to weak
                      current_strengths[i][class_index] = 0
      
      # Update the clips dataframe with the modified arrays
      self.clips_df = self.clips_df.with_columns([
          pl.Series(current_annotations).alias('annotation_status'),
          pl.Series(current_strengths).alias('label_strength')
      ])
  
  def get_strong_labels_mask(self, class_index: int = None):
      """
      Get a boolean mask for clips with strong labels.
      
      Args:
          class_index: If provided, check only for that class. If None, check if any class has strong labels.
      
      Returns:
          Boolean mask indicating clips with strong labels
      """
      if class_index is not None:
          if class_index >= self.num_classes:
              raise ValueError(f"Class index {class_index} is out of range for {self.num_classes} classes")
          # Check if the specific class has a strong label (value = 1)
          return self.df['label_strength'].list.get(class_index) == 1
      else:
          # Check if any class has a strong label (sum > 0 means at least one class has strong label)
          return self.df['label_strength'].list.eval(pl.element().sum() > 0)
  
  def get_weak_labels_mask(self, class_index: int = None):
      """
      Get a boolean mask for clips with weak labels (not explicitly annotated).
      
      Args:
          class_index: If provided, check only for that class. If None, check if all classes have weak labels.
      
      Returns:
          Boolean mask indicating clips with weak labels
      """
      if class_index is not None:
          if class_index >= self.num_classes:
              raise ValueError(f"Class index {class_index} is out of range for {self.num_classes} classes")
          # Check if the specific class has a weak label (value = 0)
          return self.df['label_strength'].list.get(class_index) == 0
      else:
          # Check if all classes have weak labels (sum == 0 means all classes are weak)
          return self.df['label_strength'].list.eval(pl.element().sum() == 0)
  
  def get_label_statistics(self):
      """
      Get statistics about annotations across all classes using the new annotations table.
      
      Returns:
          Dictionary with label statistics
      """
      total_clips = len(self.clips_df)
      
      # Get annotation counts from the annotations table
      annotation_counts = {}
      if len(self.annotations_df) > 0:
          # Count by label type
          label_counts = self.annotations_df.group_by("label").agg(pl.count("annotation_id").alias("count"))
          for row in label_counts.iter_rows():
              annotation_counts[row[0]] = row[1]
      
      # Count annotated vs unannotated clips
      annotated_clip_ids = set()
      if len(self.annotations_df) > 0:
          annotated_clip_ids = set(self.annotations_df["clip_id"].to_list())
      
      annotated_clips = len(annotated_clip_ids)
      unannotated_clips = total_clips - annotated_clips
      
      # Per-class statistics - count annotations for each class name
      class_stats = {}
      if len(self.annotations_df) > 0:
          class_counts = self.annotations_df.group_by(["class_name", "label"]).agg(pl.count("annotation_id").alias("count"))
          
          # Organize by class
          for row in class_counts.iter_rows():
              class_name = row[0]
              label = row[1] 
              count = row[2]
              
              if class_name not in class_stats:
                  class_stats[class_name] = {"present": 0, "not_present": 0, "uncertain": 0}
              
              if label in class_stats[class_name]:
                  class_stats[class_name][label] = count
      
      # Convert to the format expected by the frontend
      # "Strong labels" = present + not_present annotations (definitive annotations)
      # "Weak labels" = uncertain annotations or no annotations
      definitive_annotations = annotation_counts.get("present", 0) + annotation_counts.get("not_present", 0)
      uncertain_annotations = annotation_counts.get("uncertain", 0)
      
      # Per-class statistics in the expected format
      class_stats_formatted = {}
      class_names = list(set(self.annotations_df["class_name"].to_list())) if len(self.annotations_df) > 0 else []
      
      for i in range(self.num_classes):
          class_key = f"class_{i}"
          class_name = class_names[i] if i < len(class_names) else None
          
          if class_name and class_name in class_stats:
              # Strong labels = present + not_present for this class
              strong_count = class_stats[class_name].get("present", 0) + class_stats[class_name].get("not_present", 0)
              # Weak labels = uncertain for this class
              weak_count = class_stats[class_name].get("uncertain", 0)
          else:
              strong_count = 0
              weak_count = 0
              
          class_stats_formatted[class_key] = {
              "strong_labels": strong_count,
              "weak_labels": weak_count
          }
      
      return {
          "total_clips": total_clips,
          "clips_with_strong_labels": definitive_annotations,  # Clips with definitive annotations
          "clips_with_only_weak_labels": uncertain_annotations,  # Clips with only uncertain annotations
          "per_class_statistics": class_stats_formatted
      }
  
  def find_similar_clips(self, embeddings_array: np.ndarray, query_embedding: np.ndarray, 
                        k: int = 10, annotation_filter: int = None):
      """
      Find similar clips using cosine similarity.
      
      Args:
          embeddings_array: The loaded embeddings array (shape: [n_clips, embedding_dim])
          query_embedding: Query vector (shape: [embedding_dim])
          k: Number of similar clips to return
          annotation_filter: If provided, only search within clips with this annotation value
      
      Returns:
          tuple: (similar_clips_df, similarities, original_indices)
      """
      if len(query_embedding) != self.embedding_dim:
          raise ValueError(f"Query embedding dimension should be {self.embedding_dim}, got {len(query_embedding)}")
      
      # Get clips that have embeddings
      clips_with_embeddings = self.df.filter(pl.col("embedding_index").is_not_null())
      
      # Apply annotation filter if specified
      if annotation_filter is not None:
          clips_with_embeddings = clips_with_embeddings.filter(
              pl.col("annotation") == annotation_filter
          )
      
      if len(clips_with_embeddings) == 0:
          return pl.DataFrame(), np.array([]), np.array([])
      
      # Get embedding indices
      embedding_indices = clips_with_embeddings['embedding_index'].to_list()
      
      # Validate indices are within bounds
      max_index = embeddings_array.shape[0] - 1
      valid_indices = [idx for idx in embedding_indices if idx is not None and 0 <= idx <= max_index]
      
      if len(valid_indices) == 0:
          return pl.DataFrame(), np.array([]), np.array([])
      
      # Extract corresponding embeddings
      clip_embeddings = embeddings_array[valid_indices]
      
      # Compute cosine similarities
      query_norm = np.linalg.norm(query_embedding)
      if query_norm == 0:
          raise ValueError("Query embedding has zero norm")
      
      embedding_norms = np.linalg.norm(clip_embeddings, axis=1)
      # Avoid division by zero
      nonzero_mask = embedding_norms > 0
      similarities = np.zeros(len(clip_embeddings))
      
      if np.any(nonzero_mask):
          similarities[nonzero_mask] = np.dot(clip_embeddings[nonzero_mask], query_embedding) / (
              embedding_norms[nonzero_mask] * query_norm
          )
      
      # Get top k
      k = min(k, len(similarities))
      top_k_local_indices = np.argsort(similarities)[-k:][::-1]
      top_similarities = similarities[top_k_local_indices]
      
      # Map back to original DataFrame indices
      valid_clips = clips_with_embeddings.filter(
          pl.col("embedding_index").is_in(valid_indices)
      )
      
      # Return top k clips
      similar_clips = valid_clips[top_k_local_indices]
      original_indices = [valid_indices[i] for i in top_k_local_indices]
      
      return similar_clips, top_similarities, np.array(original_indices)
  
  def export_wav_clips(self, export_path, annotation_slug, sr=None):
    """
    Export annotated audio clips as WAV files using the three-table structure.
    
    Args:
        export_path (str): Directory path where the WAV files will be saved.
        annotation_slug (str): String to add to filenames as an annotation identifier.
        sr (int, optional): Sampling rate for the exported files. If None, uses the original sampling rate.
    
    Returns:
        tuple: (num_positive_exported, num_negative_exported, num_uncertain_exported) - Count of exported clips by type
    """
    import os
    import librosa
    import soundfile as sf
    import json
    from datetime import datetime
    
    # Create the export directory if it doesn't exist
    os.makedirs(export_path, exist_ok=True)
    
    # Get all clips that have annotations using the annotations table
    annotated_clip_ids = set()
    if len(self.annotations_df) > 0:
        annotated_clip_ids = set(self.annotations_df["clip_id"].to_list())
    
    # Get clips with files information for annotated clips only
    all_clips_with_files = self.get_clips_with_files()
    all_annotated_clips = all_clips_with_files.filter(
        pl.col("clip_id").is_in(list(annotated_clip_ids))
    )
    
    num_total = len(all_annotated_clips)
    print(f"Found {num_total} annotated clips for export.")
    
    
    # Track successful exports and metadata
    clips_exported = 0
    export_metadata = {
        "export_info": {
            "export_date": datetime.now().isoformat(),
            "annotation_slug": annotation_slug,
            "export_path": export_path,
            "total_clips_exported": 0
        },
        "clips": []
    }
    
    def export_single_clip(row_dict):
        """Helper function to export a single clip using the new three-table structure"""
        nonlocal clips_exported
        
        # Extract clip details
        clip_id = row_dict['clip_id']
        file_path = row_dict['file_path']
        file_name = row_dict['file_name']
        clip_start = row_dict['clip_start']
        clip_end = row_dict['clip_end']
        original_sr = row_dict['sampling_rate']
        
        # Get annotations for this clip
        clip_annotations = self.annotations_df.filter(pl.col("clip_id") == clip_id)
        
        # Build annotations by class name
        annotations_by_class = {}
        if len(clip_annotations) > 0:
            for ann_row in clip_annotations.iter_rows():
                annotation_id, clip_id_db, class_name, label, annotated_at = ann_row
                annotations_by_class[class_name] = label
        
        # Determine positive classes (those labeled as "present")
        positive_classes = [class_name for class_name, label in annotations_by_class.items() if label == "present"]
        
        # Skip if no annotations at all
        if not annotations_by_class:
            print(f"Skipping clip {clip_id} - no annotations found")
            return
        
        try:
            # Load audio data for the clip
            audio, _ = librosa.load(file_path, sr=original_sr, offset=clip_start, duration=clip_end-clip_start)
            
            # Resample if needed
            target_sr = sr or original_sr
            if target_sr != original_sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
                export_sr = target_sr
            else:
                export_sr = original_sr
            
            # Create filename based on positive classes
            if positive_classes:
                classes_str = "+".join(positive_classes)
                output_filename = f"{file_name}_{clip_start:.1f}-{classes_str}.wav"
            else:
                # No positive classes - this is a negative example
                output_filename = f"{file_name}_{clip_start:.1f}-empty.wav"
            
            output_path = os.path.join(export_path, output_filename)
            
            # Save as WAV file
            sf.write(output_path, audio, export_sr)
            clips_exported += 1
            
            print(f"Exported clip {clips_exported}: {output_filename}")
            
            # Add comprehensive metadata for this clip
            clip_metadata = {
                "filename": output_filename,
                "original_file": file_name,
                "file_path": file_path,
                "clip_id": clip_id,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "annotation_slug": annotation_slug,
                "annotations": annotations_by_class,  # All annotations for this clip
                "positive_classes": positive_classes,  # Classes labeled as present
                "sampling_rate": export_sr
            }
            export_metadata["clips"].append(clip_metadata)
            
        except Exception as e:
            print(f"Error exporting clip {clip_id} from {file_name}: {str(e)}")
    
    # Export all annotated clips  
    positive_count = 0
    negative_count = 0
    uncertain_count = 0
    
    for i, row in enumerate(all_annotated_clips.iter_rows(named=True)):
        print(f"Processing clip {i+1}/{num_total}: {row['file_name']} {row['clip_start']}-{row['clip_end']}")
        
        # Count annotation types for this clip
        clip_annotations = self.annotations_df.filter(pl.col("clip_id") == row['clip_id'])
        has_present = False
        has_uncertain = False
        
        if len(clip_annotations) > 0:
            for ann_row in clip_annotations.iter_rows():
                _, _, _, label, _ = ann_row
                if label == "present":
                    has_present = True
                elif label == "uncertain":
                    has_uncertain = True
        
        if has_present:
            positive_count += 1
        elif has_uncertain:
            uncertain_count += 1
        else:
            negative_count += 1
        
        export_single_clip(row)
        
        if (i+1) % 10 == 0:
            print(f"Exported {clips_exported}/{num_total} clips...")
    
    # Update total count in metadata
    export_metadata["export_info"]["total_clips_exported"] = clips_exported
    export_metadata["export_info"]["positive_clips"] = positive_count
    export_metadata["export_info"]["negative_clips"] = negative_count  
    export_metadata["export_info"]["uncertain_clips"] = uncertain_count
    
    # Export metadata JSON file
    metadata_path = f"{export_path}/export_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)
    
    print(f"Exported {clips_exported} clips total:")
    print(f"  - Positive clips: {positive_count}")
    print(f"  - Negative clips: {negative_count}")
    print(f"  - Uncertain clips: {uncertain_count}")
    print(f"  - Metadata saved to: {metadata_path}")
    
    return (positive_count, negative_count, uncertain_count)
  
  # Convenience methods for querying the new structure
  def get_files_df(self) -> pl.DataFrame:
      """Get the files DataFrame."""
      return self.files_df
      
  def get_clips_df(self) -> pl.DataFrame:
      """Get the clips DataFrame."""
      return self.clips_df
      
  def get_annotations_df(self) -> pl.DataFrame:
      """Get the annotations DataFrame."""
      return self.annotations_df
      
  def get_clips_with_files(self) -> pl.DataFrame:
      """Get clips joined with file information."""
      return self.clips_df.join(self.files_df, on="file_id", how="left")
      
  def get_clips_with_annotations(self, class_name: str = None) -> pl.DataFrame:
      """Get clips with their annotations, optionally filtered by class."""
      annotations = self.annotations_df
      if class_name:
          annotations = annotations.filter(pl.col("class_name") == class_name)
          
      return self.clips_df.join(
          annotations,
          on="clip_id",
          how="left"
      ).join(
          self.files_df,
          on="file_id", 
          how="left"
      )
  
  def get_clip_by_legacy_info(self, file_path: str, clip_start: float, clip_end: float) -> pl.DataFrame:
      """Find clip using legacy file path and time information."""
      return self.clips_df.join(
          self.files_df,
          on="file_id",
          how="inner"
      ).filter(
          (pl.col("file_path") == file_path) &
          (pl.col("clip_start") == clip_start) &
          (pl.col("clip_end") == clip_end)
      )
  
  def add_annotation(self, clip_id: str, class_name: str, label: str, annotator_id: str = "user") -> None:
      """Add or update an annotation for a specific clip and class."""
      import uuid
      from datetime import datetime
      
      print(f"DEBUG: add_annotation called with clip_id={clip_id}, class_name={class_name}, label={label}")
      print(f"DEBUG: Current annotations_df columns: {self.annotations_df.columns}")
      print(f"DEBUG: Current annotations_df schema: {self.annotations_df.schema}")
      
      try:
          # Remove existing annotation for this clip and class
          print(f"DEBUG: Current annotations_df length: {len(self.annotations_df)}")
          self.annotations_df = self.annotations_df.filter(
              ~((pl.col("clip_id") == clip_id) & (pl.col("class_name") == class_name))
          )
          print(f"DEBUG: After filtering, annotations_df length: {len(self.annotations_df)}")
          
          # Add new annotation - match the existing schema
          new_annotation = pl.DataFrame({
              "annotation_id": [str(uuid.uuid4())],
              "clip_id": [clip_id],
              "class_name": [class_name], 
              "label": [label],
              "annotated_at": [datetime.now()]
          })
          
          print(f"DEBUG: Created new annotation DataFrame with {len(new_annotation)} rows")
          print(f"DEBUG: New annotation columns: {new_annotation.columns}")
          print(f"DEBUG: New annotation schema: {new_annotation.schema}")
          self.annotations_df = pl.concat([self.annotations_df, new_annotation])
          print(f"DEBUG: Final annotations_df length: {len(self.annotations_df)}")
          
      except Exception as e:
          print(f"ERROR in add_annotation: {e}")
          raise
  
  def get_clip_id_by_legacy_info(self, file_path: str, clip_start: float, clip_end: float) -> str:
      """Get clip_id using legacy file path and time information."""
      result = self.get_clip_by_legacy_info(file_path, clip_start, clip_end)
      if len(result) == 0:
          raise ValueError(f"No clip found for {file_path} at {clip_start}-{clip_end}")
      return result["clip_id"][0]
  
  @property
  def df(self):
      """Backwards compatibility property that creates a legacy-style DataFrame view."""
      # Join clips with files to get the legacy format
      legacy_df = self.clips_df.join(self.files_df, on="file_id", how="left")
      
      # Get the current class index (default to 0)
      class_index = getattr(self, '_current_class_index', 0)
      
      # Extract scores for the current class
      if "confidence_predictions" in legacy_df.columns:
          legacy_df = legacy_df.with_columns(
              pl.col("confidence_predictions").list.get(class_index).fill_null(0.0).alias("score")
          )
      elif "predictions" in legacy_df.columns:
          legacy_df = legacy_df.with_columns(
              pl.col("predictions").list.get(class_index).fill_null(0.0).alias("score")
          )
      else:
          legacy_df = legacy_df.with_columns(pl.lit(0.0).alias("score"))
      
      # Extract annotation status for the current class
      if "annotation_status" in legacy_df.columns:
          legacy_df = legacy_df.with_columns(
              pl.col("annotation_status").list.get(class_index).fill_null(4).alias("annotation")
          )
      else:
          legacy_df = legacy_df.with_columns(pl.lit(4).alias("annotation"))
      
      return legacy_df