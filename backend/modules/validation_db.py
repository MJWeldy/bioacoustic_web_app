import polars as pl
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import uuid
import os
import json
import re
import librosa
import soundfile as sf
import threading
import time

class ValidationDB:
    """
    Simple validation database that expects user-provided strata column.
    Much simpler than the complex temporal/spatial grouping approach.

    Includes smart queue system for non-blocking saves:
    - Saves happen in background thread
    - Multiple save requests collapse into single save of latest state
    - Thread-safe with locking to prevent corruption
    """

    def __init__(self):
        """Initialize validation database with three core tables."""

        # Project save location tracking
        self.project_base_path = None
        self.project_name = None

        # Smart queue system for non-blocking saves
        self._save_pending = False  # Flag indicating save is needed
        self._save_lock = threading.Lock()  # Prevents concurrent saves AND dataframe modifications
        self._shutdown = False  # Flag to stop worker thread
        self._save_worker_thread = None  # Background thread for saves
        self._start_save_worker()  # Start the background worker

        # Predictions table - stores classifier predictions
        self.predictions_df = pl.DataFrame(
            schema={
                'prediction_id': pl.Utf8,  # Primary key
                'filename': pl.Utf8,       # Audio filename
                'start_time': pl.Float32,  # Start time in seconds
                'end_time': pl.Float32,    # End time in seconds
                'species_name': pl.Utf8,   # Species/class name
                'confidence': pl.Float32,  # Model confidence [0-1]
                'model_name': pl.Utf8,     # Source model name
                'audio_file_path': pl.Utf8, # Full path to audio file
                'strata_id': pl.Utf8,      # Foreign key to strata
                'strata': pl.Utf8,         # User-provided strata grouping
                'created_at': pl.Datetime  # Import timestamp
            }
        )

        # Validation annotations table - stores human validation decisions
        self.validation_annotations_df = pl.DataFrame(
            schema={
                'annotation_id': pl.Utf8,        # Primary key
                'prediction_id': pl.Utf8,        # FK to predictions
                'filename': pl.Utf8,             # Audio filename (denormalized)
                'start_time': pl.Float32,        # Start time (denormalized)
                'end_time': pl.Float32,          # End time (denormalized)
                'species_name': pl.Utf8,         # Species name (denormalized)
                'original_confidence': pl.Float32, # Original model confidence
                'validation_state': pl.Utf8,    # confirmed/rejected/uncertain/skipped
                'validation_confidence': pl.Int32, # User confidence 1-5
                'annotator_id': pl.Utf8,         # Who made the annotation
                'validated_at': pl.Datetime,     # When annotation was made
                'strata_id': pl.Utf8,           # FK to strata
                'notes': pl.Utf8                # Optional user notes
            }
        )

        # Strata definitions table - defines validation groupings
        self.strata_definitions_df = pl.DataFrame(
            schema={
                'strata_id': pl.Utf8,         # Primary key
                'strata_name': pl.Utf8,       # Human-readable name
                'strata_type': pl.Utf8,       # user_provided
                'confidence_threshold': pl.Float32,  # Minimum confidence threshold for this strata
                'created_at': pl.Datetime     # When strata was defined
            }
        )

        # Validation progress table - tracks progress per strata/species
        self.validation_progress_df = pl.DataFrame(
            schema={
                'strata_id': pl.Utf8,         # FK to strata
                'strata_name': pl.Utf8,       # Denormalized strata name
                'species_name': pl.Utf8,      # Species being validated
                'total_clips': pl.Int32,      # Total clips for this strata/species
                'validated_clips': pl.Int32,  # Number validated so far
                'confirmed_clips': pl.Int32,  # Number confirmed
                'rejected_clips': pl.Int32,   # Number rejected
                'uncertain_clips': pl.Int32,  # Number marked uncertain
                'skipped_clips': pl.Int32,    # Number skipped
                'target_confirmations': pl.Int32, # Target number of confirmations
                'is_completed': pl.Boolean,   # Manually marked as complete
                'last_updated': pl.Datetime   # Last progress update
            }
        )

    def clear_data(self):
        """
        Clear all data from the validation database.
        Thread-safe operation that resets all dataframes to empty state.
        Called when loading a new dataset with replace_existing=True.
        """
        with self._save_lock:
            print("INFO: Clearing validation database data")

            # Clear predictions table
            self.predictions_df = pl.DataFrame(
                schema={
                    'prediction_id': pl.Utf8,
                    'filename': pl.Utf8,
                    'start_time': pl.Float32,
                    'end_time': pl.Float32,
                    'species_name': pl.Utf8,
                    'confidence': pl.Float32,
                    'model_name': pl.Utf8,
                    'audio_file_path': pl.Utf8,
                    'strata_id': pl.Utf8,
                    'strata': pl.Utf8,
                    'created_at': pl.Datetime
                }
            )

            # Clear validation annotations table
            self.validation_annotations_df = pl.DataFrame(
                schema={
                    'annotation_id': pl.Utf8,
                    'prediction_id': pl.Utf8,
                    'filename': pl.Utf8,
                    'start_time': pl.Float32,
                    'end_time': pl.Float32,
                    'species_name': pl.Utf8,
                    'original_confidence': pl.Float32,
                    'validation_state': pl.Utf8,
                    'validation_confidence': pl.Int32,
                    'annotator_id': pl.Utf8,
                    'validated_at': pl.Datetime,
                    'strata_id': pl.Utf8,
                    'notes': pl.Utf8
                }
            )

            # Clear strata definitions table
            self.strata_definitions_df = pl.DataFrame(
                schema={
                    'strata_id': pl.Utf8,
                    'strata_name': pl.Utf8,
                    'strata_type': pl.Utf8,
                    'confidence_threshold': pl.Float32,
                    'created_at': pl.Datetime
                }
            )

            # Clear validation progress table
            self.validation_progress_df = pl.DataFrame(
                schema={
                    'strata_id': pl.Utf8,
                    'strata_name': pl.Utf8,
                    'species_name': pl.Utf8,
                    'total_clips': pl.Int32,
                    'validated_clips': pl.Int32,
                    'confirmed_clips': pl.Int32,
                    'rejected_clips': pl.Int32,
                    'uncertain_clips': pl.Int32,
                    'skipped_clips': pl.Int32,
                    'target_confirmations': pl.Int32,
                    'is_completed': pl.Boolean,
                    'last_updated': pl.Datetime
                }
            )

            print("INFO: Validation database cleared successfully")

    def acquire_lock(self):
        """Acquire the database lock for thread-safe operations."""
        self._save_lock.acquire()

    def release_lock(self):
        """Release the database lock."""
        self._save_lock.release()

    def _start_save_worker(self):
        """Start the background worker thread for smart queue saves."""
        if self._save_worker_thread is None or not self._save_worker_thread.is_alive():
            self._shutdown = False
            self._save_worker_thread = threading.Thread(target=self._save_worker, daemon=True)
            self._save_worker_thread.start()
            print("INFO: Smart queue save worker started")

    def _save_worker(self):
        """
        Background worker that processes save requests.

        Smart queue logic:
        - Waits for save_pending flag
        - When set, clones small dataframes and captures references
        - Releases lock BEFORE writing to disk (non-blocking)
        - After save, checks if another save is needed (state changed during save)
        - This ensures latest state is always saved without queuing all intermediate states
        """
        while not self._shutdown:
            # Check if save is needed
            if self._save_pending:
                # Acquire lock BRIEFLY to capture current state
                with self._save_lock:
                    # Reset flag before saving (new changes during save will re-set it)
                    self._save_pending = False

                    # Clone small dataframes that change frequently (fast operation)
                    # These are cloned so we can write them after releasing the lock
                    annotations_snapshot = self.validation_annotations_df.clone()
                    progress_snapshot = self.validation_progress_df.clone()

                    # Just capture references for large dataframes that rarely change
                    # (they'll only be written if files don't exist yet)
                    predictions_snapshot = self.predictions_df
                    strata_defs_snapshot = self.strata_definitions_df

                    # Capture project info
                    base_path = self.project_base_path
                    proj_name = self.project_name

                # Lock is released here! Annotations can continue while we write to disk

                # Perform the actual save WITHOUT holding the lock
                if base_path and proj_name:
                    try:
                        save_start = time.time()
                        result = self._save_dataframes_to_disk(
                            base_path,
                            proj_name,
                            predictions_snapshot,
                            annotations_snapshot,
                            strata_defs_snapshot,
                            progress_snapshot
                        )
                        save_duration = (time.time() - save_start) * 1000

                        if result.get('status') == 'success':
                            print(f"INFO: Smart queue auto-save completed in {save_duration:.1f}ms")
                        else:
                            print(f"WARNING: Smart queue auto-save failed: {result.get('message')}")
                    except Exception as e:
                        print(f"ERROR: Smart queue auto-save exception: {e}")

                # After save completes, check if another save is needed
                # (state may have changed during save operation)
                if self._save_pending:
                    print("INFO: State changed during save, will save again")
                    continue

            # Sleep briefly to avoid busy-waiting
            time.sleep(0.1)

        print("INFO: Smart queue save worker stopped")

    def _save_dataframes_to_disk(
        self,
        base_path: str,
        project_name: str,
        predictions_df,
        annotations_df,
        strata_defs_df,
        progress_df
    ) -> Dict[str, Any]:
        """
        Write dataframes to disk without holding the lock.

        This is called by the background worker with cloned/snapshot dataframes.
        """
        from pathlib import Path
        from datetime import datetime
        import json

        try:
            base_dir = Path(base_path)
            project_dir = base_dir / project_name
            project_dir.mkdir(parents=True, exist_ok=True)

            # Define file paths
            predictions_path = project_dir / "predictions.parquet"
            annotations_path = project_dir / "annotations.parquet"
            strata_defs_path = project_dir / "strata_definitions.parquet"
            progress_path = project_dir / "validation_progress.parquet"
            metadata_path = project_dir / "project_metadata.json"

            # Save dataframes (only if they have data)
            # Predictions and strata_defs: only write if file doesn't exist
            # (they don't change during validation, only during initial load)
            if len(predictions_df) > 0 and not predictions_path.exists():
                predictions_df.write_parquet(predictions_path)

            # Annotations and progress: always write (they change frequently)
            if len(annotations_df) > 0:
                annotations_df.write_parquet(annotations_path)

            if len(strata_defs_df) > 0 and not strata_defs_path.exists():
                strata_defs_df.write_parquet(strata_defs_path)

            if len(progress_df) > 0:
                progress_df.write_parquet(progress_path)

            # Update metadata (read existing if present to preserve created_at)
            existing_metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        existing_metadata = json.load(f)
                except:
                    pass

            metadata = {
                "project_name": project_name,
                "created_at": existing_metadata.get("created_at", datetime.now().isoformat()),
                "last_saved": datetime.now().isoformat(),
                "total_predictions": len(predictions_df),
                "total_annotations": len(annotations_df),
                "total_strata": len(strata_defs_df),
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return {
                'status': 'success',
                'message': 'Validation database saved successfully',
                'project_path': str(project_dir)
            }

        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'message': f'Failed to save validation database: {str(e)}',
                'traceback': traceback.format_exc()
            }

    def wait_for_pending_save(self, timeout: float = 10.0) -> bool:
        """
        Wait for any pending save operation to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if save completed or no save pending, False if timeout
        """
        if not self._save_pending:
            return True

        print("INFO: Waiting for pending save to complete...")
        start_time = time.time()

        while self._save_pending and (time.time() - start_time) < timeout:
            time.sleep(0.1)

        if self._save_pending:
            print(f"WARNING: Save did not complete within {timeout}s timeout")
            return False
        else:
            print("INFO: Pending save completed")
            return True

    def shutdown_save_worker(self):
        """Stop the background save worker thread gracefully."""
        if not self._save_worker_thread:
            return

        self._shutdown = True

        # Always wait for thread to finish
        if self._save_worker_thread.is_alive():
            if self._save_pending:
                print("INFO: Waiting for final save to complete...")
            self._save_worker_thread.join(timeout=10)
            if self._save_worker_thread.is_alive():
                print("WARNING: Save worker thread did not stop within timeout")
            else:
                print("INFO: Smart queue save worker shutdown complete")

        # Clear thread reference
        self._save_worker_thread = None

    def get_save_status(self) -> Dict[str, Any]:
        """
        Get current status of the smart queue save system.

        Returns:
            Dict with save_pending flag and worker status
        """
        return {
            'save_pending': self._save_pending,
            'worker_alive': self._save_worker_thread.is_alive() if self._save_worker_thread else False,
            'project_configured': bool(self.project_base_path and self.project_name)
        }

    def load_predictions_from_csv(self,
                                  file_path: str,
                                  format_type: str = 'auto',
                                  model_name: str = 'unknown',
                                  audio_directory: str = None,
                                  replace_existing: bool = False) -> Dict[str, Any]:
        """
        Load predictions from CSV file in wide or long format.
        Expects a 'strata' column with user-defined groupings.

        Args:
            file_path: Path to predictions CSV file
            format_type: 'wide', 'long', or 'auto' for auto-detection
            model_name: Name of the model that generated predictions
            audio_directory: Directory containing audio files
            replace_existing: If True, clear existing predictions before loading new ones

        Returns:
            Dict with load status and summary statistics
        """
        try:
            # Clear existing predictions if requested
            if replace_existing:
                print("DEBUG: Clearing existing predictions")
                self.predictions_df = pl.DataFrame(
                    schema={
                        'prediction_id': pl.Utf8,
                        'filename': pl.Utf8,
                        'start_time': pl.Float32,
                        'end_time': pl.Float32,
                        'species_name': pl.Utf8,
                        'confidence': pl.Float32,
                        'model_name': pl.Utf8,
                        'audio_file_path': pl.Utf8,
                        'strata_id': pl.Utf8,
                        'strata': pl.Utf8,
                        'created_at': pl.Datetime
                    }
                )

            # Load the CSV file with proper schema inference for floating point data
            df = pl.read_csv(
                file_path,
                infer_schema_length=10000,  # Scan more rows for better type inference
                null_values=['', 'null', 'NULL', 'None', 'NaN']
            )

            print(f"DEBUG: Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            print(f"DEBUG: Columns: {df.columns}")

            # Auto-detect format if requested
            if format_type == 'auto':
                format_type = self._detect_format(df)

            print(f"DEBUG: Detected format: {format_type}")

            # Convert to long format if needed
            if format_type == 'wide':
                print("DEBUG: Converting from wide to long format")
                df_long = self._convert_wide_to_long(df)
            else:
                print("DEBUG: Data already in long format, no conversion needed")
                df_long = df

            print(f"DEBUG: After format conversion - Columns: {df_long.columns}")
            print(f"DEBUG: After format conversion - Row count: {len(df_long)}")

            # Validate required columns
            required_cols = ['filename', 'start_time', 'end_time', 'species_name', 'confidence']
            missing_cols = [col for col in required_cols if col not in df_long.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Check for strata column - add default if missing
            if 'strata' not in df_long.columns:
                print("Warning: No 'strata' column found. Adding default strata 'all_data'")
                df_long = df_long.with_columns(pl.lit('all_data').alias('strata'))
            else:
                # Show sample of strata values for debugging
                unique_strata = df_long['strata'].unique().to_list()
                print(f"DEBUG: Found strata column with {len(unique_strata)} unique values: {unique_strata[:10]}")

                # Check for null or empty values
                null_count = df_long.filter(pl.col('strata').is_null()).height
                empty_count = df_long.filter(pl.col('strata') == '').height
                if null_count > 0:
                    print(f"WARNING: Found {null_count} rows with null strata values")
                if empty_count > 0:
                    print(f"WARNING: Found {empty_count} rows with empty strata values")

                # Replace null/empty strata with 'all_data'
                if null_count > 0 or empty_count > 0:
                    print("DEBUG: Replacing null/empty strata values with 'all_data'")
                    df_long = df_long.with_columns(
                        pl.when((pl.col('strata').is_null()) | (pl.col('strata') == ''))
                        .then(pl.lit('all_data'))
                        .otherwise(pl.col('strata'))
                        .alias('strata')
                    )

            # Generate prediction IDs and add metadata
            prediction_ids = [str(uuid.uuid4()) for _ in range(len(df_long))]
            current_time = datetime.now()

            # Add audio file paths if directory provided
            audio_paths = []
            if audio_directory:
                audio_paths = self._link_audio_files(df_long['filename'].to_list(), audio_directory)
            else:
                audio_paths = [''] * len(df_long)

            # Create predictions dataframe
            predictions_new = pl.DataFrame({
                'prediction_id': prediction_ids,
                'filename': df_long['filename'],
                'start_time': df_long['start_time'].cast(pl.Float32),
                'end_time': df_long['end_time'].cast(pl.Float32),
                'species_name': df_long['species_name'],
                'confidence': df_long['confidence'].cast(pl.Float32),
                'model_name': [model_name] * len(df_long),
                'audio_file_path': audio_paths,
                'strata_id': [''] * len(df_long),  # Will be populated when strata are created
                'strata': df_long['strata'],  # Include the strata column
                'created_at': [current_time] * len(df_long)
            })

            # Ensure current predictions_df schema is correct before concat
            self._ensure_schema_compliance()

            # Append to existing predictions
            self.predictions_df = pl.concat([self.predictions_df, predictions_new])
            
            # Deduplicate to prevent multiple loads of same data
            # Keep existing records (first) to preserve prediction_ids and annotation links
            self.predictions_df = self.predictions_df.unique(
                subset=['filename', 'start_time', 'end_time', 'species_name', 'model_name'],
                keep='first'
            )

            # Generate summary statistics
            total_predictions = len(predictions_new)
            unique_files = len(predictions_new['filename'].unique())
            unique_species = len(predictions_new['species_name'].unique())
            species_list = sorted(predictions_new['species_name'].unique().to_list())
            confidence_range = [
                float(predictions_new['confidence'].min()),
                float(predictions_new['confidence'].max())
            ]
            audio_files_linked = sum(1 for path in audio_paths if path)
            format_detected = format_type

            # Count unique strata
            unique_strata = len(predictions_new['strata'].unique())
            strata_list = sorted(predictions_new['strata'].unique().to_list())

            return {
                'status': 'success',
                'total_predictions': total_predictions,
                'unique_files': unique_files,
                'unique_species': unique_species,
                'species_list': species_list,
                'confidence_range': confidence_range,
                'audio_files_linked': audio_files_linked,
                'format_detected': format_detected,
                'unique_strata': unique_strata,
                'strata_list': strata_list
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def _ensure_schema_compliance(self):
        """Ensure all dataframes have the expected columns and types."""
        # Define expected schemas
        schemas = {
            'predictions_df': {
                'prediction_id': pl.Utf8,
                'filename': pl.Utf8,
                'start_time': pl.Float32,
                'end_time': pl.Float32,
                'species_name': pl.Utf8,
                'confidence': pl.Float32,
                'model_name': pl.Utf8,
                'audio_file_path': pl.Utf8,
                'strata_id': pl.Utf8,
                'strata': pl.Utf8,
                'created_at': pl.Datetime
            },
            'validation_annotations_df': {
                'annotation_id': pl.Utf8,
                'prediction_id': pl.Utf8,
                'filename': pl.Utf8,
                'start_time': pl.Float32,
                'end_time': pl.Float32,
                'species_name': pl.Utf8,
                'original_confidence': pl.Float32,
                'validation_state': pl.Utf8,
                'validation_confidence': pl.Int32,
                'annotator_id': pl.Utf8,
                'validated_at': pl.Datetime,
                'strata_id': pl.Utf8,
                'notes': pl.Utf8
            },
            'strata_definitions_df': {
                'strata_id': pl.Utf8,
                'strata_name': pl.Utf8,
                'strata_type': pl.Utf8,
                'confidence_threshold': pl.Float32,
                'created_at': pl.Datetime
            },
            'validation_progress_df': {
                'strata_id': pl.Utf8,
                'strata_name': pl.Utf8,
                'species_name': pl.Utf8,
                'total_clips': pl.Int32,
                'validated_clips': pl.Int32,
                'confirmed_clips': pl.Int32,
                'rejected_clips': pl.Int32,
                'uncertain_clips': pl.Int32,
                'skipped_clips': pl.Int32,
                'target_confirmations': pl.Int32,
                'is_completed': pl.Boolean,
                'last_updated': pl.Datetime
            }
        }

        for df_name, schema in schemas.items():
            df = getattr(self, df_name)
            modified = False
            
            for col_name, col_type in schema.items():
                if col_name not in df.columns:
                    # Add missing column with nulls/defaults
                    if col_type == pl.Boolean:
                        default_val = False
                    elif col_type in [pl.Int32, pl.Int64]:
                        default_val = 0
                    elif col_type in [pl.Float32, pl.Float64]:
                        default_val = 0.0
                    else:
                        default_val = None
                        
                    df = df.with_columns(pl.lit(default_val).alias(col_name).cast(col_type))
                    modified = True
                else:
                    # Ensure correct type
                    if df.schema[col_name] != col_type:
                        df = df.with_columns(pl.col(col_name).cast(col_type))
                        modified = True
            
            if modified:
                setattr(self, df_name, df)

    def _detect_format(self, df: pl.DataFrame) -> str:
        """Detect if CSV is in wide or long format."""
        # Look for common wide format indicators
        potential_species_cols = [col for col in df.columns
                                 if col not in ['filename', 'start_time', 'end_time', 'strata']]

        # Check if we have numeric columns that could be species confidences
        numeric_cols = []
        for col in potential_species_cols:
            try:
                df.select(pl.col(col).cast(pl.Float32))
                numeric_cols.append(col)
            except:
                pass

        print(f"DEBUG: Identified metadata columns: {['filename', 'start_time', 'end_time']}")
        print(f"DEBUG: Identified species confidence columns: {numeric_cols}")

        if len(numeric_cols) > 3:  # Multiple species columns suggest wide format
            return 'wide'
        elif 'species_name' in df.columns and 'confidence' in df.columns:
            return 'long'
        else:
            # Default to wide format if unclear
            return 'wide'

    def _convert_wide_to_long(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert wide format (species as columns) to long format."""
        # Identify metadata columns and species columns
        # Check if strata column exists, if not add it with default values
        has_strata = 'strata' in df.columns
        if not has_strata:
            print("DEBUG: Wide format CSV missing 'strata' column, adding default 'all_data'")
            df = df.with_columns(pl.lit('all_data').alias('strata'))

        metadata_cols = ['filename', 'start_time', 'end_time', 'strata']
        species_cols = [col for col in df.columns if col not in metadata_cols]

        # Filter species columns to only include those with numeric data
        valid_species_cols = []
        for col in species_cols:
            try:
                # Try to cast column to float - if successful, it's a confidence column
                test_series = df.select(pl.col(col).cast(pl.Float64))
                valid_species_cols.append(col)
            except:
                # Skip non-numeric columns
                continue

        print(f"DEBUG: Converting {len(valid_species_cols)} species columns to long format")
        print(f"DEBUG: Strata column present: {has_strata}")

        # Melt the dataframe
        df_long = df.melt(
            id_vars=metadata_cols,
            value_vars=valid_species_cols,
            variable_name='species_name',
            value_name='confidence'
        )

        # For wide format data, replace null confidence values with 0.0 instead of removing them
        # This maintains balance across species since every clip should have a prediction for every species
        df_long = df_long.with_columns(
            pl.col('confidence').fill_null(0.0)
        )

        # Ensure confidence is float type
        df_long = df_long.with_columns(pl.col('confidence').cast(pl.Float32))

        # Filter out invalid strata values (Excel errors, null values, etc.)
        print("DEBUG: Before filtering - unique strata:", df_long['strata'].unique().to_list()[:10])
        rows_before = len(df_long)

        # Replace problematic values with 'all_data' instead of filtering them out completely
        df_long = df_long.with_columns(
            pl.when(
                (pl.col('strata').is_null()) |
                (pl.col('strata') == '') |
                (pl.col('strata') == '#REF!') |
                (pl.col('strata') == '#N/A') |
                (pl.col('strata') == '#VALUE!') |
                (pl.col('strata') == '#DIV/0!') |
                (pl.col('strata').str.starts_with('#'))
            )
            .then(pl.lit('all_data'))
            .otherwise(pl.col('strata'))
            .alias('strata')
        )

        rows_after = len(df_long)
        print(f"DEBUG: After strata cleaning - rows: {rows_after} (removed {rows_before - rows_after})")
        print("DEBUG: After filtering - unique strata:", df_long['strata'].unique().to_list()[:10])

        return df_long

    def _link_audio_files(self, filenames: List[str], audio_directory: str) -> List[str]:
        """Link audio files to predictions by finding matching files."""
        audio_paths = []
        audio_dir_path = Path(audio_directory)

        if not audio_dir_path.exists():
            print(f"Warning: Audio directory {audio_directory} not found")
            return [''] * len(filenames)

        # Create a mapping of base filenames to full paths
        audio_files = {}
        for audio_file in audio_dir_path.rglob('*'):
            if audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.aiff', '.m4a']:
                audio_files[audio_file.name] = str(audio_file)

        # Match filenames to audio files
        for filename in filenames:
            if filename in audio_files:
                audio_paths.append(audio_files[filename])
            else:
                audio_paths.append('')

        linked_count = sum(1 for path in audio_paths if path)
        print(f"DEBUG: Linked {linked_count}/{len(filenames)} audio files")

        return audio_paths

    def load_unvalidated_clips(self,
                               audio_directory: str,
                               clip_window_length: float,
                               target_classes: List[str],
                               strata_file: str = None,
                               use_filename_as_strata: bool = False,
                               replace_existing: bool = False) -> Dict[str, Any]:
        """
        Load unvalidated clips by subdividing audio files into fixed-length windows.
        Creates "predictions" with confidence 0.0 for each target class.

        Args:
            audio_directory: Directory containing audio files
            clip_window_length: Length of each clip window in seconds
            target_classes: List of target class names for validation
            strata_file: Optional path to CSV file with 'filename' and 'strata' columns
            use_filename_as_strata: If True, use audio filename as strata (keeps clips sequential per file)
            replace_existing: If True, clear existing predictions before loading

        Returns:
            Dict with load status and summary statistics
        """
        try:
            # Clear existing predictions if requested
            if replace_existing:
                print("DEBUG: Clearing existing predictions")
                self.predictions_df = pl.DataFrame(
                    schema={
                        'prediction_id': pl.Utf8,
                        'filename': pl.Utf8,
                        'start_time': pl.Float32,
                        'end_time': pl.Float32,
                        'species_name': pl.Utf8,
                        'confidence': pl.Float32,
                        'model_name': pl.Utf8,
                        'audio_file_path': pl.Utf8,
                        'strata_id': pl.Utf8,
                        'strata': pl.Utf8,
                        'created_at': pl.Datetime
                    }
                )

            audio_dir = Path(audio_directory)
            if not audio_dir.exists():
                raise ValueError(f"Audio directory not found: {audio_directory}")

            # Load strata mapping from CSV if provided
            strata_mapping = {}
            if strata_file and strata_file.strip():
                strata_path = Path(strata_file)
                if not strata_path.exists():
                    raise ValueError(f"Strata file not found: {strata_file}")

                print(f"DEBUG: Loading strata mapping from {strata_file}")
                strata_df = pl.read_csv(strata_file)

                # Validate required columns
                if 'filename' not in strata_df.columns or 'strata' not in strata_df.columns:
                    raise ValueError("Strata CSV must contain 'filename' and 'strata' columns")

                # Create filename -> strata mapping
                for row in strata_df.iter_rows(named=True):
                    strata_mapping[row['filename']] = row['strata']

                print(f"DEBUG: Loaded strata mapping for {len(strata_mapping)} files")

            # Find all audio files
            audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg']
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(audio_dir.rglob(f'*{ext}'))

            if not audio_files:
                raise ValueError(f"No audio files found in {audio_directory}")

            print(f"DEBUG: Found {len(audio_files)} audio files")

            # Process each audio file
            parsed_data = []
            failed_files = []
            total_clips = 0
            unmapped_files = []

            for audio_file in audio_files:
                try:
                    # Get audio duration
                    info = sf.info(str(audio_file))
                    duration = info.duration
                    filename = audio_file.name

                    # Calculate number of clips
                    num_clips = int(np.ceil(duration / clip_window_length))
                    total_clips += num_clips

                    # Determine strata value based on options
                    strata_value = 'all_data'  # Default

                    if use_filename_as_strata:
                        # Use the filename itself as the strata (keeps clips sequential per file)
                        strata_value = filename
                    elif strata_mapping:
                        # Use mapping from CSV file
                        if filename in strata_mapping:
                            strata_value = strata_mapping[filename]
                        else:
                            # Track unmapped files for warning
                            unmapped_files.append(filename)
                            strata_value = 'unmapped'

                    # Create clips for each target class
                    for clip_idx in range(num_clips):
                        clip_start = clip_idx * clip_window_length
                        clip_end = min((clip_idx + 1) * clip_window_length, duration)

                        for target_class in target_classes:
                            parsed_data.append({
                                'filename': filename,
                                'audio_file_path': str(audio_file),
                                'start_time': clip_start,
                                'end_time': clip_end,
                                'species_name': target_class,
                                'confidence': 0.0,  # Unvalidated clips have 0 confidence
                                'strata': strata_value
                            })

                except Exception as e:
                    print(f"WARNING: Failed to process {audio_file}: {e}")
                    failed_files.append(str(audio_file))
                    continue

            if not parsed_data:
                raise ValueError("No valid clips could be generated from audio files")

            print(f"DEBUG: Generated {len(parsed_data)} clip-class combinations")

            # Create dataframe from parsed data
            df_long = pl.DataFrame(parsed_data)

            # Generate prediction IDs and add metadata
            prediction_ids = [str(uuid.uuid4()) for _ in range(len(df_long))]
            current_time = datetime.now()

            # Create predictions dataframe
            predictions_new = pl.DataFrame({
                'prediction_id': prediction_ids,
                'filename': df_long['filename'],
                'start_time': df_long['start_time'].cast(pl.Float32),
                'end_time': df_long['end_time'].cast(pl.Float32),
                'species_name': df_long['species_name'],
                'confidence': df_long['confidence'].cast(pl.Float32),
                'model_name': ['unvalidated'] * len(df_long),
                'audio_file_path': df_long['audio_file_path'],
                'strata_id': [''] * len(df_long),
                'strata': df_long['strata'],
                'created_at': [current_time] * len(df_long)
            })

            # Ensure current predictions_df schema is correct before concat
            self._ensure_schema_compliance()

            # Append to existing predictions
            self.predictions_df = pl.concat([self.predictions_df, predictions_new])
            
            # Deduplicate to prevent multiple loads of same data
            # Keep existing records (first) to preserve prediction_ids and annotation links
            self.predictions_df = self.predictions_df.unique(
                subset=['filename', 'start_time', 'end_time', 'species_name', 'model_name'],
                keep='first'
            )

            # Generate summary statistics
            total_predictions = len(predictions_new)
            unique_files = len(predictions_new['filename'].unique())
            unique_species = len(predictions_new['species_name'].unique())
            species_list = sorted(predictions_new['species_name'].unique().to_list())
            unique_strata = len(predictions_new['strata'].unique())
            strata_list = sorted(predictions_new['strata'].unique().to_list())

            # Warn about unmapped files if any
            if unmapped_files:
                print(f"WARNING: {len(unmapped_files)} files not found in strata mapping. Using 'unmapped' as strata.")
                print(f"Unmapped files: {unmapped_files[:10]}{'...' if len(unmapped_files) > 10 else ''}")

            result = {
                'status': 'success',
                'total_predictions': total_predictions,
                'total_clips': total_clips,
                'unique_files': unique_files,
                'unique_species': unique_species,
                'species_list': species_list,
                'confidence_range': [0.0, 0.0],
                'audio_files_linked': unique_files,
                'format_detected': 'unvalidated_clips',
                'unique_strata': unique_strata,
                'strata_list': strata_list,
                'failed_files': len(failed_files),
                'clip_window_length': clip_window_length
            }

            # Add unmapped files warning if applicable
            if unmapped_files:
                result['unmapped_files'] = len(unmapped_files)
                result['unmapped_files_list'] = unmapped_files[:20]  # Return first 20 for display

            return result

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def create_strata(self, clear_existing: bool = True, confidence_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Create validation strata using the user-provided strata column.
        Much simpler than the old approach - just uses the strata values directly.

        Args:
            clear_existing: If True, clear existing strata definitions and progress
                          before creating new ones (default: True)
            confidence_threshold: Minimum confidence threshold for this strata (default: 0.0)

        Returns:
            Summary of created strata
        """
        try:
            # Ensure schema compliance before operating on dataframes
            self._ensure_schema_compliance()

            if len(self.predictions_df) == 0:
                return {
                    'status': 'error',
                    'message': 'No predictions loaded. Load predictions first.'
                }

            # Clear existing strata definitions and progress if requested
            if clear_existing:
                print("DEBUG: Clearing existing strata definitions and progress")
                self.strata_definitions_df = pl.DataFrame(
                    schema={
                        'strata_id': pl.Utf8,
                        'strata_name': pl.Utf8,
                        'strata_type': pl.Utf8,
                        'confidence_threshold': pl.Float32,
                        'created_at': pl.Datetime
                    }
                )
                self.validation_progress_df = pl.DataFrame(
                    schema={
                        'strata_id': pl.Utf8,
                        'strata_name': pl.Utf8,
                        'species_name': pl.Utf8,
                        'total_clips': pl.Int32,
                        'validated_clips': pl.Int32,
                        'confirmed_clips': pl.Int32,
                        'rejected_clips': pl.Int32,
                        'uncertain_clips': pl.Int32,
                        'skipped_clips': pl.Int32,
                        'target_confirmations': pl.Int32,
                        'is_completed': pl.Boolean,
                        'last_updated': pl.Datetime
                    }
                )
                # Reset all strata_ids in predictions
                self.predictions_df = self.predictions_df.with_columns(
                    pl.lit('').alias('strata_id')
                )

            # Use all predictions (no confidence filtering at strata creation)
            filtered_df = self.predictions_df

            print(f"DEBUG: predictions_df has {len(filtered_df)} rows")
            print(f"DEBUG: predictions_df columns: {filtered_df.columns}")

            # Check if strata column exists
            if 'strata' not in filtered_df.columns:
                return {
                    'status': 'error',
                    'message': 'No strata column found in predictions. Load predictions with strata information first.'
                }

            # Get unique strata from the strata column
            unique_strata = filtered_df.select('strata').unique().to_series().to_list()
            print(f"DEBUG: Found {len(unique_strata)} unique strata: {unique_strata}")

            # Check for problematic strata values
            null_strata_count = filtered_df.filter(pl.col('strata').is_null()).height
            empty_strata_count = filtered_df.filter(pl.col('strata') == '').height
            if null_strata_count > 0:
                print(f"WARNING: Found {null_strata_count} predictions with null strata")
            if empty_strata_count > 0:
                print(f"WARNING: Found {empty_strata_count} predictions with empty strata")

            # Create strata definitions and validation progress records
            strata_mapping = {}
            new_strata_records = []
            new_progress_records = []

            for strata_name in unique_strata:
                strata_id = str(uuid.uuid4())
                strata_mapping[strata_name] = strata_id

                # Add to strata definitions
                new_strata_records.append({
                    'strata_id': strata_id,
                    'strata_name': strata_name,
                    'strata_type': 'user_provided',
                    'confidence_threshold': confidence_threshold,
                    'created_at': datetime.now()
                })

                # Get predictions for this strata
                strata_predictions = filtered_df.filter(pl.col('strata') == strata_name)

                # Create progress records for each species in this strata
                species_counts = strata_predictions.group_by('species_name').agg([
                    pl.count().alias('total_clips')
                ])

                for species_row in species_counts.iter_rows(named=True):
                    new_progress_records.append({
                        'strata_id': strata_id,
                        'strata_name': strata_name,
                        'species_name': species_row['species_name'],
                        'total_clips': int(species_row['total_clips']),  # Convert to Python int (Int32 compatible)
                        'validated_clips': 0,
                        'confirmed_clips': 0,
                        'rejected_clips': 0,
                        'uncertain_clips': 0,
                        'skipped_clips': 0,
                        'target_confirmations': 1,
                        'is_completed': False,
                        'last_updated': datetime.now()
                    })

                # Update predictions with strata IDs
                prediction_ids = strata_predictions['prediction_id'].to_list()
                self.predictions_df = self.predictions_df.with_columns(
                    pl.when(pl.col('prediction_id').is_in(prediction_ids))
                    .then(pl.lit(strata_id))
                    .otherwise(pl.col('strata_id'))
                    .alias('strata_id')
                )

            # Add strata definitions to dataframe
            if new_strata_records:
                new_strata_df = pl.DataFrame(new_strata_records)
                # Ensure correct data types to match schema
                new_strata_df = new_strata_df.with_columns([
                    pl.col('confidence_threshold').cast(pl.Float32)
                ])
                
                # Robust concatenation: ensure schemas match
                for col in new_strata_df.columns:
                    if col not in self.strata_definitions_df.columns:
                        self.strata_definitions_df = self.strata_definitions_df.with_columns(
                            pl.lit(None).alias(col).cast(new_strata_df.schema[col])
                        )
                
                self.strata_definitions_df = pl.concat([self.strata_definitions_df, new_strata_df])

            # Add progress records to dataframe
            if new_progress_records:
                new_progress_df = pl.DataFrame(new_progress_records)
                # Ensure correct data types to match schema
                new_progress_df = new_progress_df.with_columns([
                    pl.col('total_clips').cast(pl.Int32),
                    pl.col('validated_clips').cast(pl.Int32),
                    pl.col('confirmed_clips').cast(pl.Int32),
                    pl.col('rejected_clips').cast(pl.Int32),
                    pl.col('uncertain_clips').cast(pl.Int32),
                    pl.col('skipped_clips').cast(pl.Int32),
                    pl.col('target_confirmations').cast(pl.Int32)
                ])
                
                # Robust concatenation: ensure schemas match
                for col in new_progress_df.columns:
                    if col not in self.validation_progress_df.columns:
                        self.validation_progress_df = self.validation_progress_df.with_columns(
                            pl.lit(None).alias(col).cast(new_progress_df.schema[col])
                        )
                        
                self.validation_progress_df = pl.concat([self.validation_progress_df, new_progress_df])

            # Generate summary
            strata_summary = []
            for strata_name in unique_strata:
                strata_records = [r for r in new_progress_records if r['strata_name'] == strata_name]
                total_predictions = sum(r['total_clips'] for r in strata_records)
                unique_species = len(set(r['species_name'] for r in strata_records))

                # Count unique files in this strata
                unique_files_count = filtered_df.filter(pl.col('strata') == strata_name).select('filename').n_unique()

                strata_summary.append({
                    'strata_group': strata_name,
                    'total_predictions': int(total_predictions),
                    'unique_species': int(unique_species),
                    'unique_files': int(unique_files_count)
                })

            return {
                'status': 'success',
                'strata_created': len(strata_mapping),
                'total_predictions_assigned': sum(r['total_predictions'] for r in strata_summary),
                'strata_summary': strata_summary
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to create strata: {str(e)}'
            }

    def get_strata_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all created strata."""
        if len(self.strata_definitions_df) == 0:
            return []

        summary = []
        for strata_row in self.strata_definitions_df.iter_rows(named=True):
            strata_id = strata_row['strata_id']
            strata_name = strata_row['strata_name']

            # Get progress for this strata
            progress_records = self.validation_progress_df.filter(
                pl.col('strata_id') == strata_id
            )

            if len(progress_records) > 0:
                total_clips = progress_records['total_clips'].sum()
                validated_clips = progress_records['validated_clips'].sum()
                species_count = len(progress_records)

                summary.append({
                    'strata_id': strata_id,
                    'strata_name': strata_name,
                    'total_clips': total_clips,
                    'validated_clips': validated_clips,
                    'species_count': species_count,
                    'completion_percentage': (validated_clips / total_clips * 100) if total_clips > 0 else 0
                })

        return summary

    def save_validation_database(self, base_path: str, project_name: str = None) -> Dict[str, Any]:
        """
        Save validation database to files for later loading.

        Args:
            base_path: Directory to save validation project files
            project_name: Optional project name (defaults to timestamp)
        Returns:
            Dict with save status and file paths
        """
        try:
            base_dir = Path(base_path)

            # Validate path (cross-platform)
            if not base_dir.is_absolute():
                import platform
                system = platform.system()
                if system == "Windows":
                    example = "C:\\Users\\YourName\\validation_projects"
                else:
                    example = "/home/user/validation_projects"

                raise ValueError(
                    f"Save location must be an absolute path\n"
                    f"You provided: {base_path}\n"
                    f"Example for {system}: {example}"
                )

            # Create directory with better error handling
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                raise ValueError(
                    f"Permission denied: Cannot create directory at {base_path}\n"
                    f"Please check:\n"
                    f"1. You have write permissions for this location\n"
                    f"2. The parent directory exists\n"
                    f"3. You have sufficient disk space"
                )
            except OSError as e:
                raise ValueError(
                    f"Failed to create directory at {base_path}\n"
                    f"Error: {e}\n"
                    f"Please ensure the path is valid and accessible"
                )

            if project_name is None:
                project_name = f"validation_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Store project location for auto-save
            self.project_base_path = str(base_path)
            self.project_name = project_name

            project_dir = base_dir / project_name
            project_dir.mkdir(exist_ok=True)

            # Save each dataframe as parquet
            predictions_path = project_dir / "predictions.parquet"
            annotations_path = project_dir / "annotations.parquet"
            strata_defs_path = project_dir / "strata_definitions.parquet"
            progress_path = project_dir / "validation_progress.parquet"
            metadata_path = project_dir / "project_metadata.json"

            # Save dataframes (only if they have data)
            if len(self.predictions_df) > 0:
                self.predictions_df.write_parquet(predictions_path)

            if len(self.validation_annotations_df) > 0:
                self.validation_annotations_df.write_parquet(annotations_path)

            if len(self.strata_definitions_df) > 0:
                self.strata_definitions_df.write_parquet(strata_defs_path)

            if len(self.validation_progress_df) > 0:
                self.validation_progress_df.write_parquet(progress_path)

            # Save project metadata (preserve created_at if it exists)
            existing_metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        existing_metadata = json.load(f)
                except:
                    pass

            metadata = {
                "project_name": project_name,
                "created_at": existing_metadata.get("created_at", datetime.now().isoformat()),
                "last_saved": datetime.now().isoformat(),
                "total_predictions": len(self.predictions_df),
                "total_annotations": len(self.validation_annotations_df),
                "total_strata": len(self.strata_definitions_df),
                "file_paths": {
                    "predictions": str(predictions_path) if len(self.predictions_df) > 0 else None,
                    "annotations": str(annotations_path) if len(self.validation_annotations_df) > 0 else None,
                    "strata_definitions": str(strata_defs_path) if len(self.strata_definitions_df) > 0 else None,
                    "validation_progress": str(progress_path) if len(self.validation_progress_df) > 0 else None
                }
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return {
                'status': 'success',
                'project_name': project_name,
                'project_path': str(project_dir),
                'metadata_path': str(metadata_path),
                'files_saved': {
                    'predictions': len(self.predictions_df) > 0,
                    'annotations': len(self.validation_annotations_df) > 0,
                    'strata_definitions': len(self.strata_definitions_df) > 0,
                    'validation_progress': len(self.validation_progress_df) > 0
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to save validation database: {str(e)}'
            }

    def auto_save(self) -> bool:
        """
        Request a non-blocking auto-save via smart queue system.

        Returns True if save was requested, False if no save path configured.

        Smart queue behavior:
        - Sets save_pending flag immediately and returns
        - Background worker will save latest state
        - Multiple rapid calls collapse into single save of most recent state
        - User can continue annotating without waiting
        """
        # Ensure worker thread is running
        if self._save_worker_thread is None or not self._save_worker_thread.is_alive():
            print("WARNING: Save worker thread not running, restarting...")
            self._start_save_worker()

        if self.project_base_path and self.project_name:
            self._save_pending = True
            return True
        else:
            print("WARNING: Auto-save requested but no project path configured")
            return False

    def load_validation_database(self, project_path: str) -> Dict[str, Any]:
        """
        Load validation database from saved files.

        Args:
            project_path: Path to validation project directory or metadata file
        Returns:
            Dict with load status and summary
        """
        try:
            # Handle both directory path and metadata file path
            if project_path.endswith('.json'):
                metadata_path = Path(project_path)
                project_dir = metadata_path.parent
            else:
                project_dir = Path(project_path)
                metadata_path = project_dir / "project_metadata.json"

            if not metadata_path.exists():
                return {
                    'status': 'error',
                    'message': f'Project metadata not found at {metadata_path}'
                }

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Reset current dataframes
            self.__init__()

            # Store project location for auto-save (after init)
            self.project_base_path = str(project_dir.parent)
            self.project_name = metadata.get('project_name')

            # Load each dataframe if file exists
            # Strategy: Check file in project dir first (robust to moves), then fallback to metadata path
            file_paths = metadata.get('file_paths', {})
            
            # Helper to load parquet with fallback
            def load_table(filename, metadata_key):
                # 1. Try direct file in project directory
                direct_path = project_dir / filename
                if direct_path.exists():
                    print(f"DEBUG: Loading {filename} from {direct_path}")
                    return pl.read_parquet(direct_path)
                
                # 2. Try path from metadata
                meta_path_str = file_paths.get(metadata_key)
                if meta_path_str:
                    meta_path = Path(meta_path_str)
                    if meta_path.exists():
                        print(f"DEBUG: Loading {filename} from metadata path: {meta_path}")
                        return pl.read_parquet(meta_path)
                
                return None

            # Load tables
            preds = load_table("predictions.parquet", "predictions")
            if preds is not None:
                self.predictions_df = preds
                # Deduplicate loaded predictions
                self.predictions_df = self.predictions_df.unique(
                    subset=['filename', 'start_time', 'end_time', 'species_name', 'model_name'],
                    keep='first'
                )

            anns = load_table("annotations.parquet", "annotations")
            if anns is not None:
                self.validation_annotations_df = anns

            strata = load_table("strata_definitions.parquet", "strata_definitions")
            if strata is not None:
                self.strata_definitions_df = strata

            prog = load_table("validation_progress.parquet", "validation_progress")
            if prog is not None:
                # Schema migration: add missing columns if from an older version
                if 'is_completed' not in prog.columns:
                    prog = prog.with_columns(pl.lit(False).alias('is_completed'))
                
                if 'last_updated' not in prog.columns:
                    prog = prog.with_columns(pl.lit(datetime.now()).alias('last_updated'))
                    
                self.validation_progress_df = prog

            # Ensure all schemas are compliant after loading
            self._ensure_schema_compliance()

            return {
                'status': 'success',
                'project_name': metadata.get('project_name'),
                'created_at': metadata.get('created_at'),
                'total_predictions': len(self.predictions_df),
                'total_annotations': len(self.validation_annotations_df),
                'total_strata': len(self.strata_definitions_df),
                'files_loaded': {
                    'predictions': len(self.predictions_df) > 0,
                    'annotations': len(self.validation_annotations_df) > 0,
                    'strata_definitions': len(self.strata_definitions_df) > 0,
                    'validation_progress': len(self.validation_progress_df) > 0
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to load validation database: {str(e)}'
            }

    def toggle_strata_completion(self, strata_id: str, species_name: str, is_completed: bool) -> Dict[str, Any]:
        """
        Mark a strata/species combination as complete or incomplete.

        Args:
            strata_id: The strata ID
            species_name: The species name
            is_completed: Whether to mark as complete (True) or incomplete (False)
        Returns:
            Dict with status and updated progress
        """
        # Acquire lock for thread-safe dataframe modification
        with self._save_lock:
            try:
                print(f"DEBUG: toggle_strata_completion called with strata_id={strata_id}, species={species_name}, is_completed={is_completed}")
                print(f"DEBUG: validation_progress_df has {len(self.validation_progress_df)} rows")
                print(f"DEBUG: validation_progress_df columns: {self.validation_progress_df.columns}")

                # Check if record exists first
                existing = self.validation_progress_df.filter(
                    (pl.col('strata_id') == strata_id) & (pl.col('species_name') == species_name)
                )
                print(f"DEBUG: Found {len(existing)} matching records")

                if len(existing) == 0:
                    return {
                        'status': 'error',
                        'message': f'No progress record found for strata_id={strata_id}, species={species_name}'
                    }

                # Update completion status for this strata/species
                self.validation_progress_df = self.validation_progress_df.with_columns(
                    pl.when(
                        (pl.col('strata_id') == strata_id) & (pl.col('species_name') == species_name)
                    )
                    .then(pl.lit(is_completed))
                    .otherwise(pl.col('is_completed'))
                    .alias('is_completed'),

                    pl.when(
                        (pl.col('strata_id') == strata_id) & (pl.col('species_name') == species_name)
                    )
                    .then(pl.lit(datetime.now()))
                    .otherwise(pl.col('last_updated'))
                    .alias('last_updated')
                )

                # Get updated progress record
                progress_record = self.validation_progress_df.filter(
                    (pl.col('strata_id') == strata_id) & (pl.col('species_name') == species_name)
                )

                progress_dict = progress_record.to_dicts()[0]
                print(f"DEBUG: Updated progress record: {progress_dict}")

                return {
                    'status': 'success',
                    'message': f'Strata marked as {"complete" if is_completed else "incomplete"}',
                    'progress': progress_dict
                }

            except Exception as e:
                import traceback
                print(f"ERROR in toggle_strata_completion: {str(e)}")
                print(f"ERROR traceback:\n{traceback.format_exc()}")
                return {
                    'status': 'error',
                    'message': f'Failed to update completion status: {str(e)}'
                }

    def list_validation_projects(self, base_path: str) -> Dict[str, Any]:
        """
        List available validation projects in a directory.

        Args:
            base_path: Directory to search for validation projects
        Returns:
            Dict with list of available projects
        """
        try:
            base_dir = Path(base_path)
            if not base_dir.exists():
                return {
                    'status': 'success',
                    'projects': []
                }

            projects = []
            for item in base_dir.iterdir():
                if item.is_dir():
                    metadata_path = item / "project_metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)

                            projects.append({
                                'project_name': metadata.get('project_name', item.name),
                                'created_at': metadata.get('created_at', metadata.get('last_saved', '')),
                                'project_path': str(item),
                                'total_predictions': metadata.get('total_predictions', 0),
                                'total_annotations': metadata.get('total_annotations', 0),
                                'total_strata': metadata.get('total_strata', 0)
                            })
                        except:
                            # Skip invalid metadata files
                            continue

            # Sort by creation date (newest first), handling None values
            projects.sort(key=lambda x: x.get('created_at') or '', reverse=True)

            return {
                'status': 'success',
                'projects': projects
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to list validation projects: {str(e)}'
            }