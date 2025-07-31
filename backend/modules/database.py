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
      Initialize the audio prediction and embedding database with polars.
      
      Args:
          embedding_dim: Dimension size of the embeddings
          num_classes: Number of classes for multiclass prediction
      """
      self.score_min = 0.0
      self.score_max = 1.0
      self.embedding_dim = embedding_dim
      self.num_classes = num_classes
      self.df = pl.DataFrame(
        schema={
          'file_name': pl.Utf8,
          'file_path': pl.Utf8,
          'duration_sec': pl.Float32,
          'clip_start': pl.Float32,
          'clip_end': pl.Float32,
          'sampling_rate': pl.Int32,
          'score': pl.Float32,
          'annotation': pl.Int32,
          'predictions': pl.List(pl.Float32),  # Vector of predictions for all classes
          'annotation_status': pl.List(pl.Int32),  # Vector of annotation status for all classes
          'label_strength': pl.List(pl.Int32),  # Vector indicating strong (1) vs weak (0) labels
          'embedding_index': pl.Int64,  # Index into embeddings array
          'created_at': pl.Datetime
        }
      )
      #'embedding': pl.List(pl.Float32),
      #'metadata': pl.Struct,  # For storing additional information
  def add_clip_row(self, 
                   file_name: str, 
                   file_path: str, 
                   duration_sec: float, 
                   clip_start: float,
                   clip_end: float,
                   sampling_rate: int,
                   embedding_index: int = None) -> None:
                    #embedding: List[float],
                    #metadata: Dict[str, Any] = None) -> None:
    """
    Add an audio embedding to the database.
    
    Args:
        file_name: Unique identifier for the audio clip
        file_path: Path to the audio file
        duration_sec: Duration in seconds
        clip_start: Start time of the audio clip in seconds
        clip_end: End time of the audio clip in seconds
        sampling_rate: Audio sampling rate in Hz
        score: Predicted classifier score 0-1
        annotation: Clip annotation state[ 0: target sound not in clip,
                                           1: target sound in clip,
                                           3: reviewed but uncertain if the target sound is in the clip,
                                           4: not yet reviewed]
        NOT YET IMPLEMENTED embedding: Embedding vector for the audio clip
        NOT YET IMPLEMENTED metadata: Additional information about the clip
    """
    # Check score input 
    #if score > self.score_max or score < self.score_min:
    #   raise ValueError(f"Scores should be between {self.score_min} and {self.score_max}")
    
    # Check embedding input
    #if len(embedding) != self.embedding_dim:
    #    raise ValueError(f"Embedding dimension should be {self.embedding_dim}")
        
    #if metadata is None:
    #    metadata = {}
    
    # ensure data types
    duration_sec_float32 = np.float32(duration_sec)
    clip_start_float32 = np.float32(clip_start)
    clip_end_float32 = np.float32(clip_end)
    sampling_rate_int32 = np.int32(sampling_rate)
    score_float32 = np.float32(2.0)
    annotation_int32 = np.int32(4)
    
    # Initialize multiclass vectors
    # predictions: initialized to 0.5 for all classes (neutral prediction)
    initial_predictions = [np.float32(0.5)] * self.num_classes
    # annotation_status: initialized to 4 for all classes (not yet reviewed)
    initial_annotation_status = [np.int32(4)] * self.num_classes
    # label_strength: initialized to 0 for all classes (weak/unlabeled)
    initial_label_strength = [np.int32(0)] * self.num_classes
    
    new_row = pl.DataFrame({
        'file_name': [file_name],
        'file_path': [file_path],
        'duration_sec': [duration_sec_float32],
        'clip_start': [clip_start_float32],
        'clip_end': [clip_end_float32],
        'sampling_rate': [sampling_rate_int32],
        'score': [score_float32], #initialized at a non-real value for initial testing
        'annotation': [annotation_int32],
        'predictions': [initial_predictions],
        'annotation_status': [initial_annotation_status],
        'label_strength': [initial_label_strength],
        'embedding_index': [embedding_index],
        'created_at': [datetime.now()]
    })

    self.df = pl.concat([self.df, new_row])

  def save_db(self, file_path: str) -> None:
    """Save the database to a file."""
    self.df.write_parquet(file_path)
  
  def load_db(self, file_path: str) -> None:
    """Load the database from a file."""
    if Path(file_path).is_file():
        self.df = pl.read_parquet(file_path)
        
        # Handle backward compatibility - add new columns if they don't exist
        if 'predictions' not in self.df.columns:
            # Initialize predictions with the current score for all classes
            predictions_list = []
            for _ in range(len(self.df)):
                pred_vector = [np.float32(0.5)] * self.num_classes
                predictions_list.append(pred_vector)
            self.df = self.df.with_columns(pl.Series("predictions", predictions_list))
        
        if 'annotation_status' not in self.df.columns:
            # Initialize annotation_status with current annotation for first class, 4 for others
            annotation_status_list = []
            current_annotations = self.df['annotation'].to_list()
            for annotation in current_annotations:
                status_vector = [np.int32(4)] * self.num_classes
                status_vector[0] = np.int32(annotation)  # Use current annotation for first class
                annotation_status_list.append(status_vector)
            self.df = self.df.with_columns(pl.Series("annotation_status", annotation_status_list))
        
        if 'label_strength' not in self.df.columns:
            # Initialize label_strength with 1 for first class if annotated, 0 for others
            label_strength_list = []
            current_annotations = self.df['annotation'].to_list()
            for annotation in current_annotations:
                strength_vector = [np.int32(0)] * self.num_classes
                # If the clip has been annotated (not 4=unreviewed), mark first class as strong
                if annotation != 4:
                    strength_vector[0] = np.int32(1)
                label_strength_list.append(strength_vector)
            self.df = self.df.with_columns(pl.Series("label_strength", label_strength_list))
        
        if 'embedding_index' not in self.df.columns:
            # Initialize embedding_index as None for all existing clips
            null_indices = [None] * len(self.df)
            self.df = self.df.with_columns(pl.Series("embedding_index", null_indices))
    else:
        raise FileNotFoundError(f"Database file {file_path} not found")
  
  def populate_scores(self, scores: List[float]):
      if len(scores) != len(self.df):
        raise ValueError(f"Length of new_values ({len(scores)}) must match DataFrame length ({len(self.df)})")
      
      if any(score > self.score_max or score < self.score_min for score in scores):
        print(f"Warning: Some scores are outside the expected range [{self.score_min}, {self.score_max}]")
      
      scores_float32 = np.float32(scores)
      self.df = self.df.with_columns(pl.Series("score", scores_float32))
  
  def populate_multiclass_predictions(self, predictions: List[List[float]]):
      """
      Populate the predictions column with multiclass prediction vectors.
      
      Args:
          predictions: List of prediction vectors, one per clip
      """
      if len(predictions) != len(self.df):
          raise ValueError(f"Length of predictions ({len(predictions)}) must match DataFrame length ({len(self.df)})")
      
      # Convert to float32 lists
      predictions_float32 = [[np.float32(score) for score in pred_vec] for pred_vec in predictions]
      self.df = self.df.with_columns(pl.Series("predictions", predictions_float32))
  
  def populate_embedding_indices(self, embedding_indices: List[int]):
      """
      Populate the embedding_index column for existing clips.
      
      Args:
          embedding_indices: List of embedding indices, one per clip (can contain None for missing embeddings)
      """
      if len(embedding_indices) != len(self.df):
          raise ValueError(f"Length of embedding indices ({len(embedding_indices)}) must match DataFrame length ({len(self.df)})")
      
      self.df = self.df.with_columns(pl.Series("embedding_index", embedding_indices))
  
  def auto_populate_embedding_indices(self):
      """
      Automatically populate embedding indices for existing clips.
      Assumes embeddings are ordered the same as clips in the database (0, 1, 2, ...).
      """
      num_clips = len(self.df)
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
      num_clips = len(self.df)
      
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
      
      Args:
          class_index: Index of the class to extract scores and annotations for
      """
      if class_index >= self.num_classes:
          raise ValueError(f"Class index {class_index} is out of range for {self.num_classes} classes")
      
      # Extract scores for the specific class
      scores = self.df['predictions'].list.get(class_index)
      annotations = self.df['annotation_status'].list.get(class_index)
      
      # Update the score and annotation columns
      self.df = self.df.with_columns([
          scores.alias("score"),
          annotations.alias("annotation")
      ])
  
  def update_class_annotation(self, clip_mask, class_index: int, annotation_value: int):
      """
      Update annotation for a specific class and clip.
      
      Args:
          clip_mask: Boolean mask identifying the clip to update
          class_index: Index of the class to update
          annotation_value: New annotation value (0, 1, 3, or 4)
      """
      if class_index >= self.num_classes:
          raise ValueError(f"Class index {class_index} is out of range for {self.num_classes} classes")
      
      # Get current annotation status and label strength vectors
      annotation_vectors = self.df['annotation_status'].to_list()
      label_strength_vectors = self.df['label_strength'].to_list()
      
      # Update the specific class annotation for rows matching the mask
      mask_values = clip_mask.to_list() if hasattr(clip_mask, 'to_list') else clip_mask
      
      for i, should_update in enumerate(mask_values):
          if should_update and i < len(annotation_vectors):
              if len(annotation_vectors[i]) > class_index:
                  annotation_vectors[i][class_index] = np.int32(annotation_value)
                  
                  # Update label strength: mark as strong (1) if annotated as present (1) or absent (0)
                  # Keep as weak (0) if uncertain (3) or unreviewed (4)
                  if annotation_value in [0, 1]:  # Present or Not Present = strong label
                      label_strength_vectors[i][class_index] = np.int32(1)
                  elif annotation_value == 4:  # Unreviewed = reset to weak
                      label_strength_vectors[i][class_index] = np.int32(0)
                  # annotation_value == 3 (uncertain) keeps the current strength value
      
      # Update the DataFrame
      self.df = self.df.with_columns([
          pl.Series("annotation_status", annotation_vectors),
          pl.Series("label_strength", label_strength_vectors)
      ])
      
      # Also update the single annotation column for the current class
      self.update_class_scores_and_annotations(class_index)
  
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
      Get statistics about strong vs weak labels across all classes.
      
      Returns:
          Dictionary with label statistics
      """
      total_clips = len(self.df)
      
      def safe_count_true(mask):
          """Helper to count True values in a mask, handling different data types"""
          try:
              return int(mask.sum())
          except:
              return mask.to_list().count(True)
      
      # Count clips with strong labels (any class)
      strong_mask = self.get_strong_labels_mask()
      strong_clips = safe_count_true(strong_mask)
      
      # Count clips with only weak labels (all classes)
      weak_mask = self.get_weak_labels_mask()
      weak_clips = safe_count_true(weak_mask)
      
      # Per-class statistics
      class_stats = {}
      for i in range(self.num_classes):
          strong_mask_i = self.get_strong_labels_mask(i)
          weak_mask_i = self.get_weak_labels_mask(i)
          
          class_strong = safe_count_true(strong_mask_i)
          class_weak = safe_count_true(weak_mask_i)
          
          class_stats[f"class_{i}"] = {
              "strong_labels": class_strong,
              "weak_labels": class_weak
          }
      
      return {
          "total_clips": total_clips,
          "clips_with_strong_labels": strong_clips,
          "clips_with_only_weak_labels": weak_clips,
          "per_class_statistics": class_stats
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
    Export annotated audio clips as WAV files with label strength information.
    
    Args:
        export_path (str): Directory path where the WAV files will be saved.
        annotation_slug (str): String to add to filenames as an annotation identifier.
        sr (int, optional): Sampling rate for the exported files. If None, uses the original sampling rate.
    
    Returns:
        tuple: (num_positive_exported, num_negative_exported) - Count of positive and negative clips exported
    """
    import os
    import librosa
    import soundfile as sf
    import json
    from datetime import datetime
    
    # Create the export directory if it doesn't exist
    os.makedirs(export_path, exist_ok=True)
    
    # Get annotated clips (positive, negative, and uncertain)
    positive_clips = self.df.filter(pl.col("annotation") == 1)
    negative_clips = self.df.filter(pl.col("annotation") == 0)
    uncertain_clips = self.df.filter(pl.col("annotation") == 3)
    
    num_positive = len(positive_clips)
    num_negative = len(negative_clips)
    num_uncertain = len(uncertain_clips)
    
    print(f"Found {num_positive} positive clips, {num_negative} negative clips, and {num_uncertain} uncertain clips for export.")
    
    # Track successful exports and metadata
    positive_exported = 0
    negative_exported = 0
    uncertain_exported = 0
    export_metadata = {
        "export_info": {
            "export_date": datetime.now().isoformat(),
            "annotation_slug": annotation_slug,
            "export_path": export_path,
            "total_clips_exported": 0
        },
        "clips": []
    }
    
    def export_single_clip(row, annotation_type, annotation_slug):
        """Helper function to export a single clip with complete multiclass metadata"""
        # Extract clip details
        file_path = row['file_path']
        file_name = row['file_name']
        clip_start = row['clip_start']
        clip_end = row['clip_end']
        original_sr = row['sampling_rate']
        
        # Get multiclass information from database
        predictions = row.get('predictions', [])
        annotation_status = row.get('annotation_status', [])
        label_strength = row.get('label_strength', [])
        current_score = row.get('score', 0.0)
        
        # Create binary label vector and strength vector for all classes
        if predictions and len(predictions) > 0:
            # Multiclass case - use vectors from database
            labels_vector = []
            strength_vector = []
            scores_vector = predictions if isinstance(predictions, list) else [current_score]
            
            # Build labels and strengths for each class
            for i in range(len(predictions)):
                if i < len(annotation_status):
                    ann_status = annotation_status[i]
                    if ann_status == 1:  # Present
                        labels_vector.append(1)
                    elif ann_status == 0:  # Not present  
                        labels_vector.append(0)
                    elif ann_status == 3:  # Uncertain
                        labels_vector.append(1)  # Treat uncertain as positive but weak
                    else:  # Unreviewed or other
                        labels_vector.append(0)
                else:
                    labels_vector.append(0)
                
                # Set strength: strong (1) for definitive annotations, weak (0) for uncertain
                if i < len(label_strength):
                    strength_vector.append(label_strength[i])
                elif i < len(annotation_status) and annotation_status[i] == 3:
                    strength_vector.append(0)  # Uncertain = weak
                elif i < len(annotation_status) and annotation_status[i] in [0, 1]:
                    strength_vector.append(1)  # Definitive = strong
                else:
                    strength_vector.append(0)  # Default to weak for unreviewed
        else:
            # Single class case - legacy support
            labels_vector = [1 if annotation_type == 1 else 0]
            strength_vector = [1 if annotation_type in [0, 1] else 0]  # Strong for definitive, weak for uncertain
            scores_vector = [current_score]
        
        # Load audio data for the clip
        audio, _ = librosa.load(file_path, sr=original_sr, offset=clip_start, duration=clip_end-clip_start)
        
        # Resample if needed
        if cfg.TARGET_SR is not None and cfg.TARGET_SR != original_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=cfg.TARGET_SR)
            export_sr = cfg.TARGET_SR
        else:
            export_sr = original_sr
        
        # Create simplified filename: original_name_clipstart-annotation_slug-strong.wav
        # Only use -strong tag to indicate this contains strong labels for the specified class
        output_filename = f"{file_name}_{clip_start:.1f}-{annotation_slug}-strong.wav"
        output_path = f"{export_path}/{output_filename}"
        
        # Save as WAV file
        sf.write(output_path, audio, export_sr)
        
        # Add comprehensive metadata for this clip
        clip_metadata = {
            "filename": output_filename,
            "original_file": file_name,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "annotation_slug": annotation_slug,
            "labels": labels_vector,              # Full binary vector for all classes
            "label_strengths": strength_vector,   # Strength vector for all classes  
            "scores": scores_vector,              # Prediction scores for all classes
            "primary_annotation": annotation_type, # Original single annotation for reference
            "sampling_rate": export_sr
        }
        export_metadata["clips"].append(clip_metadata)
        
        return output_filename
    
    # Export positive clips (annotation = 1)
    for i, row in enumerate(positive_clips.iter_rows(named=True)):
        try:
            export_single_clip(row, 1, annotation_slug)
            positive_exported += 1
            
            if (i+1) % 10 == 0:
                print(f"Exported {i+1}/{num_positive} positive clips...")
        
        except Exception as e:
            print(f"Error exporting positive clip {row['file_name']}: {str(e)}")
    
    # Export negative clips (annotation = 0)
    for i, row in enumerate(negative_clips.iter_rows(named=True)):
        try:
            export_single_clip(row, 0, "empty")
            negative_exported += 1
            
            if (i+1) % 10 == 0:
                print(f"Exported {i+1}/{num_negative} negative clips...")
        
        except Exception as e:
            print(f"Error exporting negative clip {row['file_name']}: {str(e)}")
    
    # Export uncertain clips (annotation = 3) 
    for i, row in enumerate(uncertain_clips.iter_rows(named=True)):
        try:
            export_single_clip(row, 3, "uncertain")
            uncertain_exported += 1
            
            if (i+1) % 10 == 0:
                print(f"Exported {i+1}/{num_uncertain} uncertain clips...")
        
        except Exception as e:
            print(f"Error exporting uncertain clip {row['file_name']}: {str(e)}")
    
    # Update total count in metadata
    total_exported = positive_exported + negative_exported + uncertain_exported
    export_metadata["export_info"]["total_clips_exported"] = total_exported
    export_metadata["export_info"]["positive_clips"] = positive_exported
    export_metadata["export_info"]["negative_clips"] = negative_exported
    export_metadata["export_info"]["uncertain_clips"] = uncertain_exported
    
    # Add class map information for proper label vector interpretation
    # Try to get class map from self (database object) if available
    if hasattr(self, 'class_map') and self.class_map:
        export_metadata["class_map"] = self.class_map
    else:
        # Default single class map
        export_metadata["class_map"] = {annotation_slug: 0}
    
    # Export metadata JSON file
    metadata_path = f"{export_path}/export_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)
    
    print(f"Exported {total_exported} clips total:")
    print(f"  - {positive_exported} positive clips")
    print(f"  - {negative_exported} negative clips") 
    print(f"  - {uncertain_exported} uncertain clips")
    print(f"  - Metadata saved to: {metadata_path}")
    
    return (positive_exported, negative_exported, uncertain_exported)
  
  def empty_method(self):
      pass