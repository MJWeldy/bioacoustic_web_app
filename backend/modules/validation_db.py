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

class ValidationDB:
    """
    Simple validation database that expects user-provided strata column.
    Much simpler than the complex temporal/spatial grouping approach.
    """

    def __init__(self):
        """Initialize validation database with three core tables."""

        # Project save location tracking
        self.project_base_path = None
        self.project_name = None

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
                'last_updated': pl.Datetime   # Last progress update
            }
        )

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
        df_long = df_long.filter(
            (pl.col('strata').is_not_null()) &
            (pl.col('strata') != '#REF!') &
            (pl.col('strata') != '#N/A') &
            (pl.col('strata') != '#VALUE!') &
            (pl.col('strata') != '#DIV/0!') &
            (pl.col('strata') != '') &
            (~pl.col('strata').str.starts_with('#'))
        )
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

    def _parse_pnw_cnet_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Parse PNW-CNet filename format used by NWFP acoustic monitoring program.

        Filename format: JCN_40758-12_20220614_123409_part_001.png
        - JCN: 3-letter study region code
        - 40758: 5-digit sampling site
        - 12: 2-digit sampling station
        - 20220614: 8-digit date (YYYYMMDD)
        - 123409: 6-digit time (HHMMSS)
        - part_001: part number (12-second windows)

        Args:
            filename: Filename to parse (with or without extension)

        Returns:
            Dict with parsed metadata or None if parsing fails
        """
        # Remove file extension if present
        base_name = Path(filename).stem

        # Pattern for PNW-CNet filename format
        # Region (3 letters) _ Site (5 digits) - Station (1-2 digits) _ Date (8 digits) _ Time (6 digits) _ part _ PartNum (3 digits)
        pattern = r'^([A-Z]{3})_(\d{5})-(\d{1,2})_(\d{8})_(\d{6})_part_(\d{3})'

        match = re.match(pattern, base_name)
        if not match:
            return None

        region = match.group(1)
        site = match.group(2)
        station = match.group(3).zfill(2)  # Zero-pad to 2 digits for consistency
        date_str = match.group(4)  # YYYYMMDD
        time_str = match.group(5)  # HHMMSS
        part_num = int(match.group(6))

        # Parse date and time
        try:
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            hour = int(time_str[0:2])
            minute = int(time_str[2:4])
            second = int(time_str[4:6])

            recording_datetime = datetime(year, month, day, hour, minute, second)
        except ValueError:
            # Invalid date/time values
            return None

        # Calculate ISO week number
        iso_calendar = recording_datetime.isocalendar()
        week_number = iso_calendar[1]  # Week number (1-53)
        week_str = f"W{week_number:02d}"

        # Calculate clip start and end times
        # Each part is a 12-second window
        # part_001 = 0-12, part_002 = 12-24, etc.
        clip_start = (part_num - 1) * 12.0
        clip_end = part_num * 12.0

        # Create site-station identifier and week-based combinations
        site_station = f"{site}-{station}"
        region_week = f"{region}-{week_str}"
        site_week = f"{site}-{week_str}"
        site_station_week = f"{site_station}-{week_str}"

        return {
            'region': region,
            'site': site,
            'station': station,
            'site_station': site_station,
            'date': date_str,
            'time': time_str,
            'datetime': recording_datetime,
            'week_number': week_number,
            'week_str': week_str,
            'region_week': region_week,
            'site_week': site_week,
            'site_station_week': site_station_week,
            'part_number': part_num,
            'clip_start': clip_start,
            'clip_end': clip_end,
            'strata': site_station  # Use site-station as default strata
        }

    def load_pnw_cnet_predictions(self,
                                   file_path: str,
                                   model_name: str = 'PNW-CNet',
                                   audio_directory: str = None,
                                   replace_existing: bool = False,
                                   strata_field: str = 'site_station') -> Dict[str, Any]:
        """
        Load predictions from PNW-CNet default output format.
        Parses filenames to extract metadata and calculate clip times.
        Supports both single files and directories (with recursive search).

        Args:
            file_path: Path to PNW-CNet predictions CSV file or directory
            model_name: Name of the model (default: 'PNW-CNet')
            audio_directory: Directory containing audio files
            replace_existing: If True, clear existing predictions before loading
            strata_field: Which field to use for strata ('site_station', 'site', 'region', etc.)

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

            # Check if path is a file or directory
            path = Path(file_path)
            csv_files = []

            if not path.exists():
                raise ValueError(
                    f"Path does not exist: {file_path}\n"
                    f"Please check that:\n"
                    f"1. The path is correct and not a typo\n"
                    f"2. The directory/file is accessible\n"
                    f"3. Any network drives are mounted\n"
                    f"4. You have read permissions for this location"
                )

            if path.is_file():
                csv_files = [path]
                print(f"DEBUG: Loading single file: {file_path}")
            elif path.is_dir():
                # Recursively find all CSV files
                csv_files = list(path.rglob('*.csv'))
                print(f"DEBUG: Found {len(csv_files)} CSV files in directory: {file_path}")
            else:
                raise ValueError(f"Path exists but is neither a file nor a directory: {file_path}")

            if not csv_files:
                raise ValueError(f"No CSV files found in: {file_path}")

            # Load and combine all CSV files
            all_dfs = []
            failed_files = []

            for csv_file in csv_files:
                try:
                    df = pl.read_csv(
                        str(csv_file),
                        infer_schema_length=10000,
                        null_values=['', 'null', 'NULL', 'None', 'NaN']
                    )

                    # Verify it has the Filename column
                    if 'Filename' in df.columns:
                        all_dfs.append(df)
                        print(f"DEBUG: Loaded {len(df)} rows from {csv_file.name}")
                    else:
                        print(f"WARNING: Skipping {csv_file.name} - no 'Filename' column")
                        failed_files.append(str(csv_file))

                except Exception as e:
                    print(f"WARNING: Failed to load {csv_file.name}: {e}")
                    failed_files.append(str(csv_file))
                    continue

            if not all_dfs:
                error_msg = "No valid PNW-CNet prediction files could be loaded.\n"
                if failed_files:
                    error_msg += f"\nFailed to load {len(failed_files)} file(s):\n"
                    for f in failed_files[:5]:  # Show first 5 failures
                        error_msg += f"  - {f}\n"
                    if len(failed_files) > 5:
                        error_msg += f"  ... and {len(failed_files) - 5} more\n"
                error_msg += "\nPNW-CNet CSV files must have:\n"
                error_msg += "  1. A 'Filename' column (case-sensitive, capital F)\n"
                error_msg += "  2. Species names as additional column headers\n"
                error_msg += "  3. Confidence values (0-1) in the cells\n"
                raise ValueError(error_msg)

            # Combine all dataframes
            if len(all_dfs) == 1:
                df = all_dfs[0]
            else:
                # Concatenate all dataframes, ensuring consistent columns
                df = pl.concat(all_dfs, how='vertical_relaxed')

            print(f"DEBUG: Combined total: {len(df)} rows and {len(df.columns)} columns")
            print(f"DEBUG: Columns: {df.columns}")

            # Verify 'Filename' column exists
            if 'Filename' not in df.columns:
                raise ValueError("PNW-CNet format requires 'Filename' column")

            # Identify species columns (all columns except 'Filename')
            species_cols = [col for col in df.columns if col != 'Filename']

            print(f"DEBUG: Identified {len(species_cols)} species columns")

            # Parse filenames and extract metadata
            parsed_data = []
            failed_parses = []

            for row in df.iter_rows(named=True):
                filename = row['Filename']
                parsed = self._parse_pnw_cnet_filename(filename)

                if parsed is None:
                    failed_parses.append(filename)
                    continue

                # Create a record for each species with its confidence
                for species_name in species_cols:
                    confidence = row[species_name]

                    # Skip null/NaN values or convert to 0.0
                    if confidence is None or (isinstance(confidence, float) and np.isnan(confidence)):
                        confidence = 0.0
                    else:
                        confidence = float(confidence)

                    # Select strata value based on strata_field
                    if strata_field == 'site_station':
                        strata_value = parsed['site_station']
                    elif strata_field == 'site':
                        strata_value = parsed['site']
                    elif strata_field == 'region':
                        strata_value = parsed['region']
                    elif strata_field == 'region_week':
                        strata_value = parsed['region_week']
                    elif strata_field == 'site_week':
                        strata_value = parsed['site_week']
                    elif strata_field == 'site_station_week':
                        strata_value = parsed['site_station_week']
                    else:
                        strata_value = parsed['site_station']  # default

                    parsed_data.append({
                        'filename': filename,
                        'start_time': parsed['clip_start'],
                        'end_time': parsed['clip_end'],
                        'species_name': species_name,
                        'confidence': confidence,
                        'strata': strata_value,
                        'region': parsed['region'],
                        'site': parsed['site'],
                        'station': parsed['station'],
                        'site_station': parsed['site_station'],
                        'recording_date': parsed['date'],
                        'recording_time': parsed['time'],
                        'part_number': parsed['part_number']
                    })

            if failed_parses:
                print(f"WARNING: Failed to parse {len(failed_parses)} filenames")
                print(f"First few failures: {failed_parses[:5]}")

            if not parsed_data:
                error_msg = "No valid data could be parsed from PNW-CNet files.\n"
                if failed_parses:
                    error_msg += f"\nFailed to parse {len(failed_parses)} filename(s):\n"
                    for f in failed_parses[:5]:  # Show first 5 failures
                        error_msg += f"  - {f}\n"
                    if len(failed_parses) > 5:
                        error_msg += f"  ... and {len(failed_parses) - 5} more\n"
                error_msg += "\nExpected filename format:\n"
                error_msg += "  REGION_SITE-STATION_YYYYMMDD_HHMMSS_part_NNN.png\n"
                error_msg += "Example:\n"
                error_msg += "  JCN_40758-12_20220614_123409_part_001.png\n"
                error_msg += "\nWhere:\n"
                error_msg += "  - REGION: 3-letter region code (e.g., JCN)\n"
                error_msg += "  - SITE: 5-digit site number (e.g., 40758)\n"
                error_msg += "  - STATION: 1-2 digit station number (e.g., 12)\n"
                error_msg += "  - DATE: YYYYMMDD format\n"
                error_msg += "  - TIME: HHMMSS format\n"
                error_msg += "  - PART: 3-digit part number (e.g., 001)\n"
                raise ValueError(error_msg)

            # Create dataframe from parsed data
            df_long = pl.DataFrame(parsed_data)

            # Generate prediction IDs and add metadata
            prediction_ids = [str(uuid.uuid4()) for _ in range(len(df_long))]
            current_time = datetime.now()

            # Link audio files if directory provided
            audio_paths = []
            if audio_directory:
                # For PNW-CNet, convert image filenames to audio filenames
                # Image: JCN_40758-12_20220614_123409_part_001.png
                # Audio: JCN_40758-12_20220614_123409.wav
                audio_filenames = []
                for fn in df_long['filename'].to_list():
                    # Remove extension and _part_NNN suffix
                    base_name = Path(fn).stem  # JCN_40758-12_20220614_123409_part_001
                    # Remove _part_NNN pattern (matches _part_ followed by 3 digits)
                    audio_base = re.sub(r'_part_\d{3}$', '', base_name)
                    audio_filenames.append(audio_base + '.wav')

                audio_paths = self._link_audio_files(audio_filenames, audio_directory)
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
                'strata_id': [''] * len(df_long),
                'strata': df_long['strata'],
                'created_at': [current_time] * len(df_long)
            })

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
                'format_detected': 'pnw_cnet',
                'unique_strata': unique_strata,
                'strata_list': strata_list,
                'failed_parses': len(failed_parses),
                'strata_field_used': strata_field,
                'csv_files_loaded': len(all_dfs),
                'csv_files_failed': len(failed_files)
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

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

    def load_density_estimation_clips(self,
                                      audio_directory: str,
                                      clip_length: float,
                                      target_class: str,
                                      sampling_interval: int = 60,
                                      clips_per_interval: int = 5,
                                      replace_existing: bool = False) -> Dict[str, Any]:
        """
        Load clips for call density estimation using systematic temporal sampling.

        Args:
            audio_directory: Directory containing audio files
            clip_length: Length of each clip in seconds
            target_class: Target class for density estimation
            sampling_interval: Time interval for systematic sampling in seconds
            clips_per_interval: Number of clips to sample per interval
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

            for audio_file in audio_files:
                try:
                    # Get audio duration
                    info = sf.info(str(audio_file))
                    duration = info.duration
                    filename = audio_file.name

                    # Calculate number of sampling intervals
                    num_intervals = int(np.floor(duration / sampling_interval))

                    # Sample clips at regular intervals
                    for interval_idx in range(num_intervals):
                        interval_start = interval_idx * sampling_interval
                        interval_end = min((interval_idx + 1) * sampling_interval, duration)

                        # Sample clips_per_interval clips within this interval
                        for clip_idx in range(clips_per_interval):
                            # Evenly distribute clips within the interval
                            clip_offset = (interval_end - interval_start) / (clips_per_interval + 1) * (clip_idx + 1)
                            clip_start = interval_start + clip_offset

                            # Ensure clip doesn't exceed file duration
                            if clip_start + clip_length <= duration:
                                clip_end = clip_start + clip_length
                                total_clips += 1

                                parsed_data.append({
                                    'filename': filename,
                                    'audio_file_path': str(audio_file),
                                    'start_time': clip_start,
                                    'end_time': clip_end,
                                    'species_name': target_class,
                                    'confidence': 0.0,  # Unvalidated clips
                                    'strata': filename  # Use filename as strata for density estimation
                                })

                except Exception as e:
                    print(f"WARNING: Failed to process {audio_file}: {e}")
                    failed_files.append(str(audio_file))
                    continue

            if not parsed_data:
                raise ValueError("No valid clips could be generated from audio files")

            print(f"DEBUG: Generated {len(parsed_data)} clips for density estimation")

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
                'model_name': ['density_estimation'] * len(df_long),
                'audio_file_path': df_long['audio_file_path'],
                'strata_id': [''] * len(df_long),
                'strata': df_long['strata'],
                'created_at': [current_time] * len(df_long)
            })

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

            return {
                'status': 'success',
                'total_predictions': total_predictions,
                'total_clips': total_clips,
                'unique_files': unique_files,
                'unique_species': 1,
                'species_list': [target_class],
                'confidence_range': [0.0, 0.0],
                'audio_files_linked': unique_files,
                'format_detected': 'density_estimation',
                'unique_strata': unique_files,
                'strata_list': sorted(predictions_new['strata'].unique().to_list()),
                'failed_files': len(failed_files),
                'clip_length': clip_length,
                'sampling_interval': sampling_interval,
                'clips_per_interval': clips_per_interval
            }

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
                        'last_updated': pl.Datetime
                    }
                )
                # Reset all strata_ids in predictions
                self.predictions_df = self.predictions_df.with_columns(
                    pl.lit('').alias('strata_id')
                )

            # Use all predictions (no confidence filtering at strata creation)
            filtered_df = self.predictions_df

            # Get unique strata from the strata column
            unique_strata = filtered_df.select('strata').unique().to_series().to_list()
            print(f"DEBUG: Found {len(unique_strata)} unique strata: {unique_strata}")

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

            # Validate path
            if not base_dir.is_absolute():
                raise ValueError(
                    f"Save location must be an absolute path starting with '/'\n"
                    f"You provided: {base_path}\n"
                    f"Example: /home/user/validation_projects"
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

            # Save project metadata
            metadata = {
                "project_name": project_name,
                "created_at": datetime.now().isoformat(),
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
        Automatically save project using stored path (if available).
        Returns True if save succeeded, False otherwise.
        """
        if self.project_base_path and self.project_name:
            try:
                result = self.save_validation_database(self.project_base_path, self.project_name)
                return result.get('status') == 'success'
            except Exception as e:
                print(f"Auto-save failed: {e}")
                return False
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
                self.validation_progress_df = prog

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
                                'project_name': metadata.get('project_name'),
                                'created_at': metadata.get('created_at'),
                                'project_path': str(item),
                                'total_predictions': metadata.get('total_predictions', 0),
                                'total_annotations': metadata.get('total_annotations', 0),
                                'total_strata': metadata.get('total_strata', 0)
                            })
                        except:
                            # Skip invalid metadata files
                            continue

            # Sort by creation date (newest first)
            projects.sort(key=lambda x: x['created_at'], reverse=True)

            return {
                'status': 'success',
                'projects': projects
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to list validation projects: {str(e)}'
            }