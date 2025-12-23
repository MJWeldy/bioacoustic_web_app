import polars as pl
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import uuid
import os
import json

class ValidationDB:
    """
    Database management for acoustic classifier validation workflows.
    Handles prediction ingestion, strata management, and validation tracking.
    """

    def __init__(self):
        """Initialize validation database with three core tables."""

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

        # Strata progress table - tracks validation progress per strata/species
        self.strata_progress_df = pl.DataFrame(
            schema={
                'strata_id': pl.Utf8,           # Primary key (compound with species)
                'strata_name': pl.Utf8,         # Human-readable strata name
                'species_name': pl.Utf8,        # Species name
                'total_clips': pl.Int32,        # Total predictions in strata
                'reviewed_clips': pl.Int32,     # Number reviewed (any state)
                'confirmed_clips': pl.Int32,    # Number confirmed as correct
                'rejected_clips': pl.Int32,     # Number rejected as incorrect
                'uncertain_clips': pl.Int32,    # Number marked uncertain
                'skipped_clips': pl.Int32,      # Number skipped
                'clips_above_threshold': pl.Int32, # Clips >= confidence threshold
                'threshold_value': pl.Float32,  # Confidence threshold used
                'completion_status': pl.Utf8,   # incomplete/completed/target_met
                'target_confirmations': pl.Int32, # Required confirmations (1,2,3)
                'last_updated': pl.Datetime     # Last progress update
            }
        )

        # Strata definitions table - defines how strata are created
        self.strata_definitions_df = pl.DataFrame(
            schema={
                'strata_id': pl.Utf8,           # Primary key
                'strata_name': pl.Utf8,         # Human-readable name
                'definition_type': pl.Utf8,     # temporal/spatial/mixed
                'grouping_columns': pl.List(pl.Utf8), # Columns used for grouping
                'date_column': pl.Utf8,         # Column for temporal grouping
                'temporal_unit': pl.Utf8,       # week/month/season/year
                'spatial_columns': pl.List(pl.Utf8), # Columns for spatial grouping
                'created_at': pl.Datetime       # When strata was defined
            }
        )

    def load_predictions_from_csv(self,
                                  file_path: str,
                                  format_type: str = 'auto',
                                  model_name: str = 'unknown',
                                  audio_directory: str = None) -> Dict[str, Any]:
        """
        Load predictions from CSV file in wide or long format.

        Args:
            file_path: Path to predictions CSV file
            format_type: 'wide', 'long', or 'auto' for auto-detection
            model_name: Name of the model that generated predictions
            audio_directory: Directory containing audio files

        Returns:
            Dict with load status and summary statistics
        """
        try:
            # Load the CSV file with proper schema inference for floating point data
            df = pl.read_csv(
                file_path,
                infer_schema_length=10000,  # Scan more rows for better type inference
                ignore_errors=False,        # We want to catch parsing errors
                null_values=["", "NA", "NULL", "null", "None"]  # Handle various null representations
            )

            # Auto-detect format if needed
            if format_type == 'auto':
                format_type = self._detect_format(df)

            # Convert to long format if needed
            if format_type == 'wide':
                df_long = self._convert_wide_to_long(df)
            else:
                df_long = df

            # Validate required columns
            required_cols = ['filename', 'start_time', 'end_time', 'species_name', 'confidence']
            missing_cols = [col for col in required_cols if col not in df_long.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Check for strata column - add default if missing
            if 'strata' not in df_long.columns:
                print("Warning: No 'strata' column found. Adding default strata 'all_data'")
                df_long = df_long.with_columns(pl.lit('all_data').alias('strata'))

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

            # Append to existing predictions
            self.predictions_df = pl.concat([self.predictions_df, predictions_new])

            # Generate summary statistics
            summary = {
                'status': 'success',
                'total_predictions': len(predictions_new),
                'unique_files': df_long['filename'].n_unique(),
                'unique_species': df_long['species_name'].n_unique(),
                'species_list': df_long['species_name'].unique().to_list(),
                'confidence_range': [
                    float(df_long['confidence'].min()),
                    float(df_long['confidence'].max())
                ],
                'format_detected': format_type,
                'audio_files_linked': sum(1 for path in audio_paths if path)
            }

            return summary

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def _detect_format(self, df: pl.DataFrame) -> str:
        """Detect if dataframe is in wide or long format."""
        # Look for common indicators
        has_species_column = 'species_name' in df.columns or 'species' in df.columns
        has_confidence_column = 'confidence' in df.columns

        # Count numeric columns (potential species confidence columns)
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]

        if has_species_column and has_confidence_column:
            return 'long'
        elif len(numeric_cols) > 3:  # Likely species confidence columns
            return 'wide'
        else:
            return 'long'  # Default assumption

    def _convert_wide_to_long(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert wide format predictions to long format."""
        # Identify metadata columns (non-species confidence columns)
        metadata_cols = []
        species_cols = []

        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['filename', 'file', 'start_time', 'start', 'end_time', 'end', 'offset', 'begin_time', 'begin_datetime', 'end_datetime']:
                metadata_cols.append(col)
            elif col_lower in ['species_name', 'species', 'confidence', 'score']:
                metadata_cols.append(col)  # These are already in long format
            else:
                # Try to identify if this could be a species confidence column
                try:
                    # Check if column contains numeric data that could be confidence scores
                    sample_values = df[col].head(100).drop_nulls()
                    if len(sample_values) > 0:
                        # Try to cast to float to see if it's numeric
                        df_test = sample_values.cast(pl.Float64, strict=False)
                        if df_test.null_count() < len(sample_values):  # Most values can be converted
                            # Check if values are in confidence range [0, 1] or reasonable scores
                            numeric_values = df_test.drop_nulls()
                            if len(numeric_values) > 0:
                                min_val = float(numeric_values.min())
                                max_val = float(numeric_values.max())
                                # Confidence scores are typically 0-1, but some models use 0-100 or other ranges
                                if min_val >= 0 and max_val <= 100:
                                    species_cols.append(col)
                                else:
                                    metadata_cols.append(col)
                            else:
                                metadata_cols.append(col)
                        else:
                            metadata_cols.append(col)
                    else:
                        metadata_cols.append(col)
                except:
                    # If casting fails, treat as metadata
                    metadata_cols.append(col)

        print(f"DEBUG: Identified metadata columns: {metadata_cols}")
        print(f"DEBUG: Identified species confidence columns: {species_cols}")

        # If no species columns identified, fall back to numeric columns
        if not species_cols:
            for col in df.columns:
                if col not in metadata_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    species_cols.append(col)

        if not species_cols:
            raise ValueError("No species confidence columns identified. Please check your data format.")

        # Ensure confidence columns are cast to float
        df_cast = df.clone()
        for col in species_cols:
            try:
                df_cast = df_cast.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                )
            except:
                print(f"Warning: Could not cast column {col} to float, skipping")
                species_cols.remove(col)
                metadata_cols.append(col)

        # Melt the dataframe
        df_long = df_cast.melt(
            id_vars=metadata_cols,
            value_vars=species_cols,
            variable_name='species_name',
            value_name='confidence'
        )

        # Remove rows with null confidence values
        df_long = df_long.filter(pl.col('confidence').is_not_null())

        # Standardize column names
        col_mapping = {}
        for col in df_long.columns:
            col_lower = col.lower()
            if col_lower in ['file']:
                col_mapping[col] = 'filename'
            elif col_lower in ['start', 'begin_time']:
                col_mapping[col] = 'start_time'
            elif col_lower in ['end']:
                col_mapping[col] = 'end_time'

        if col_mapping:
            df_long = df_long.rename(col_mapping)

        # Ensure we have required columns with defaults if missing
        required_cols = ['filename', 'start_time', 'end_time', 'species_name', 'confidence']
        for req_col in required_cols:
            if req_col not in df_long.columns:
                if req_col == 'start_time':
                    df_long = df_long.with_columns(pl.lit(0.0).alias('start_time'))
                elif req_col == 'end_time':
                    df_long = df_long.with_columns(pl.lit(1.0).alias('end_time'))
                else:
                    raise ValueError(f"Required column '{req_col}' not found and cannot be inferred")

        return df_long

    def _link_audio_files(self, filenames: List[str], audio_directory: str) -> List[str]:
        """Link prediction filenames to actual audio files."""
        audio_paths = []
        audio_dir = Path(audio_directory)

        # Create a mapping of available audio files
        audio_files = {}
        if audio_dir.exists():
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                for audio_file in audio_dir.rglob(f'*{ext}'):
                    audio_files[audio_file.stem] = str(audio_file)

        # Match filenames to audio files
        for filename in filenames:
            # Try exact match first
            stem = Path(filename).stem
            if stem in audio_files:
                audio_paths.append(audio_files[stem])
            else:
                # Try partial matches
                found = False
                for audio_stem, audio_path in audio_files.items():
                    if stem in audio_stem or audio_stem in stem:
                        audio_paths.append(audio_path)
                        found = True
                        break
                if not found:
                    audio_paths.append('')

        return audio_paths

    def create_strata(self,
                      strata_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create validation strata based on grouping criteria.

        Args:
            strata_definition: Dict defining how to group predictions into strata

        Returns:
            Summary of created strata
        """
        try:
            definition_type = strata_definition.get('type', 'temporal')
            grouping_columns = strata_definition.get('grouping_columns', ['filename'])
            temporal_unit = strata_definition.get('temporal_unit', 'week')

            # Extract grouping information from filenames or metadata
            df_with_groups = self._extract_grouping_info(
                self.predictions_df,
                grouping_columns,
                temporal_unit
            )

            # Create strata groups
            strata_groups = df_with_groups.group_by(['strata_group']).agg([
                pl.count().alias('total_predictions'),
                pl.col('species_name').n_unique().alias('unique_species'),
                pl.col('filename').n_unique().alias('unique_files')
            ])

            # Generate strata IDs and update predictions
            strata_mapping = {}
            new_strata_defs = []
            new_progress_records = []

            for group_row in strata_groups.iter_rows(named=True):
                strata_id = str(uuid.uuid4())
                strata_name = group_row['strata_group']
                strata_mapping[strata_name] = strata_id

                # Add to strata definitions
                new_strata_defs.append({
                    'strata_id': strata_id,
                    'strata_name': strata_name,
                    'definition_type': definition_type,
                    'grouping_columns': grouping_columns,
                    'temporal_unit': temporal_unit,
                    'created_at': datetime.now()
                })

                # Create progress records for each species in this strata
                if strata_name is None:
                    strata_predictions = df_with_groups.filter(
                        pl.col('strata_group').is_null()
                    )
                else:
                    strata_predictions = df_with_groups.filter(
                        pl.col('strata_group') == strata_name
                    )

                species_counts = strata_predictions.group_by('species_name').agg([
                    pl.count().alias('total_clips')
                ])

                for species_row in species_counts.iter_rows(named=True):
                    new_progress_records.append({
                        'strata_id': strata_id,
                        'strata_name': strata_name,
                        'species_name': species_row['species_name'],
                        'total_clips': species_row['total_clips'],
                        'reviewed_clips': 0,
                        'confirmed_clips': 0,
                        'rejected_clips': 0,
                        'uncertain_clips': 0,
                        'skipped_clips': 0,
                        'clips_above_threshold': 0,
                        'threshold_value': 0.5,
                        'completion_status': 'incomplete',
                        'target_confirmations': 1,
                        'last_updated': datetime.now()
                    })

            # Update predictions with strata IDs
            for strata_name, strata_id in strata_mapping.items():
                if strata_name is None:
                    mask = df_with_groups['strata_group'].is_null()
                else:
                    mask = df_with_groups['strata_group'] == strata_name
                prediction_ids = df_with_groups.filter(mask)['prediction_id'].to_list()

                # Update predictions dataframe
                self.predictions_df = self.predictions_df.with_columns(
                    pl.when(pl.col('prediction_id').is_in(prediction_ids))
                    .then(pl.lit(strata_id))
                    .otherwise(pl.col('strata_id'))
                    .alias('strata_id')
                )

            # Add new records to dataframes
            if new_strata_defs:
                new_strata_df = pl.DataFrame(new_strata_defs)
                self.strata_definitions_df = pl.concat([self.strata_definitions_df, new_strata_df])

            if new_progress_records:
                new_progress_df = pl.DataFrame(new_progress_records)
                self.strata_progress_df = pl.concat([self.strata_progress_df, new_progress_df])

            return {
                'status': 'success',
                'strata_created': len(strata_groups),
                'total_predictions_assigned': len(df_with_groups),
                'strata_summary': strata_groups.to_dicts()
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def _extract_grouping_info(self,
                               df: pl.DataFrame,
                               grouping_columns: List[str],
                               temporal_unit: str) -> pl.DataFrame:
        """Extract grouping information from predictions for strata creation."""
        # Start with predictions dataframe
        df_groups = df.clone()

        # Extract temporal information from filename if needed
        if temporal_unit in ['week', 'month', 'year']:
            # Assume filename contains date information
            # Example: Site_001_20231015_... -> extract date
            df_groups = df_groups.with_columns([
                pl.col('filename').str.extract(r'(\d{8})', 1).alias('date_str')
            ])

            # Convert to date and extract temporal unit
            df_groups = df_groups.with_columns([
                pl.col('date_str').str.strptime(pl.Date, '%Y%m%d', strict=False).alias('date')
            ])

            if temporal_unit == 'week':
                df_groups = df_groups.with_columns([
                    pl.col('date').dt.strftime('%Y-W%U').alias('temporal_group')
                ])
            elif temporal_unit == 'month':
                df_groups = df_groups.with_columns([
                    pl.col('date').dt.strftime('%Y-%m').alias('temporal_group')
                ])
            elif temporal_unit == 'year':
                df_groups = df_groups.with_columns([
                    pl.col('date').dt.strftime('%Y').alias('temporal_group')
                ])

        # Extract site information from filename
        # Example: Site_001_... -> Site_001
        df_groups = df_groups.with_columns([
            pl.col('filename').str.extract(r'(Site_\d+)', 1).alias('site_group')
        ])

        # Combine grouping columns to create strata
        group_parts = []
        if 'site_group' in df_groups.columns:
            group_parts.append('site_group')
        if 'temporal_group' in df_groups.columns:
            group_parts.append('temporal_group')

        if group_parts:
            df_groups = df_groups.with_columns([
                pl.concat_str(group_parts, separator='_').alias('strata_group')
            ])
        else:
            # Fallback to filename-based grouping
            df_groups = df_groups.with_columns([
                pl.col('filename').str.extract(r'^([^_]+_[^_]+)', 1).alias('strata_group')
            ])

        return df_groups

    def get_strata_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all strata and their validation progress."""
        if len(self.strata_progress_df) == 0:
            return []

        summary = self.strata_progress_df.group_by(['strata_id', 'strata_name']).agg([
            pl.col('species_name').count().alias('species_count'),
            pl.col('total_clips').sum().alias('total_clips'),
            pl.col('reviewed_clips').sum().alias('reviewed_clips'),
            pl.col('confirmed_clips').sum().alias('confirmed_clips'),
            pl.col('completion_status').first().alias('overall_status')
        ])

        return summary.to_dicts()

    def save_validation_database(self, base_path: str) -> Dict[str, Any]:
        """Save validation database to files."""
        try:
            base_dir = Path(base_path)
            base_dir.mkdir(parents=True, exist_ok=True)

            # Save each table
            self.predictions_df.write_parquet(base_dir / 'predictions.parquet')
            self.validation_annotations_df.write_parquet(base_dir / 'validation_annotations.parquet')
            self.strata_progress_df.write_parquet(base_dir / 'strata_progress.parquet')
            self.strata_definitions_df.write_parquet(base_dir / 'strata_definitions.parquet')

            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'total_predictions': len(self.predictions_df),
                'total_annotations': len(self.validation_annotations_df),
                'total_strata': len(self.strata_definitions_df),
                'database_version': '1.0'
            }

            with open(base_dir / 'validation_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            return {
                'status': 'success',
                'saved_to': str(base_dir),
                'files_created': 5
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

    def load_validation_database(self, base_path: str) -> Dict[str, Any]:
        """Load validation database from files."""
        try:
            base_dir = Path(base_path)

            # Load each table if it exists
            if (base_dir / 'predictions.parquet').exists():
                self.predictions_df = pl.read_parquet(base_dir / 'predictions.parquet')

            if (base_dir / 'validation_annotations.parquet').exists():
                self.validation_annotations_df = pl.read_parquet(base_dir / 'validation_annotations.parquet')

            if (base_dir / 'strata_progress.parquet').exists():
                self.strata_progress_df = pl.read_parquet(base_dir / 'strata_progress.parquet')

            if (base_dir / 'strata_definitions.parquet').exists():
                self.strata_definitions_df = pl.read_parquet(base_dir / 'strata_definitions.parquet')

            return {
                'status': 'success',
                'loaded_from': str(base_dir),
                'predictions_count': len(self.predictions_df),
                'annotations_count': len(self.validation_annotations_df),
                'strata_count': len(self.strata_definitions_df)
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }