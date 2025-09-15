"""
Shared test fixtures and utilities for the bioacoustic web app test suite.

This module provides reusable fixtures using only the Python standard library
and existing dependencies to avoid adding new test dependencies.
"""

import pytest
import tempfile
import numpy as np
import polars as pl
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import uuid


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_audio_data():
    """Generate synthetic audio data for testing."""
    # Create a simple sine wave for testing
    sample_rate = 22050
    duration = 5.0  # seconds
    frequency = 440  # Hz (A note)

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5  # Amplitude 0.5

    return audio.astype(np.float32), sample_rate


@pytest.fixture
def mock_embeddings():
    """Generate mock embedding vectors for testing."""
    # Standard embedding dimensions for different models
    embedding_dim = 1280  # Common for PERCH
    num_embeddings = 10

    # Generate random embeddings with realistic distribution
    embeddings = np.random.normal(0, 0.1, (num_embeddings, embedding_dim)).astype(np.float32)

    return embeddings


@pytest.fixture
def mock_class_map():
    """Create a mock class mapping for testing."""
    return {
        "NOWA_song": 0,
        "CEWA_call": 1,
        "WIWA_song": 2,
        "background": 3
    }


@pytest.fixture
def mock_audio_db():
    """Create a mock Audio_DB instance for testing."""
    with patch('modules.database.Audio_DB') as mock_db_class:
        mock_db = Mock()
        mock_db.num_classes = 3
        mock_db.embedding_dim = 1280
        mock_db.score_min = 0.0
        mock_db.score_max = 1.0

        # Mock the DataFrames
        mock_db.files_df = pl.DataFrame(schema={
            'file_id': pl.Utf8,
            'file_name': pl.Utf8,
            'file_path': pl.Utf8,
            'duration_sec': pl.Float32,
            'sampling_rate': pl.Int32,
            'created_at': pl.Datetime
        })

        mock_db.clips_df = pl.DataFrame(schema={
            'clip_id': pl.Utf8,
            'file_id': pl.Utf8,
            'clip_start': pl.Float32,
            'clip_end': pl.Float32,
            'annotation_status': pl.List(pl.Int32),
            'confidence_predictions': pl.List(pl.Float32),
            'label_strength': pl.List(pl.Int32),
            'embedding_start_index': pl.Int32,
            'created_at': pl.Datetime
        })

        mock_db.annotations_df = pl.DataFrame(schema={
            'annotation_id': pl.Utf8,
            'clip_id': pl.Utf8,
            'class_name': pl.Utf8,
            'label': pl.Utf8,
            'annotated_at': pl.Datetime,
            'annotator_id': pl.Utf8
        })

        mock_db_class.return_value = mock_db
        yield mock_db


@pytest.fixture
def mock_file_operations():
    """Mock all file I/O operations to prevent real file access."""
    with patch('soundfile.read') as mock_sf_read, \
         patch('librosa.resample') as mock_resample, \
         patch('polars.DataFrame.write_parquet') as mock_write_parquet, \
         patch('polars.read_parquet') as mock_read_parquet, \
         patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('builtins.open', create=True) as mock_open:

        # Configure default return values
        mock_sf_read.return_value = (np.zeros(22050, dtype=np.float32), 22050)
        mock_resample.return_value = np.zeros(22050, dtype=np.float32)
        mock_exists.return_value = True

        yield {
            'sf_read': mock_sf_read,
            'resample': mock_resample,
            'write_parquet': mock_write_parquet,
            'read_parquet': mock_read_parquet,
            'exists': mock_exists,
            'mkdir': mock_mkdir,
            'open': mock_open
        }


@pytest.fixture
def mock_tensorflow_operations():
    """Mock TensorFlow operations to avoid model loading."""
    with patch('tensorflow.keras.models.load_model') as mock_load_model, \
         patch('tensorflow_hub.load') as mock_hub_load, \
         patch('tensorflow.function') as mock_tf_function:

        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.random.random((10, 3))  # Mock predictions
        mock_load_model.return_value = mock_model

        # Mock TensorFlow Hub model
        mock_hub_model = Mock()
        mock_hub_model.return_value = np.random.random((10, 1280))  # Mock embeddings
        mock_hub_load.return_value = mock_hub_model

        # Mock tf.function decorator
        mock_tf_function.side_effect = lambda func: func

        yield {
            'load_model': mock_load_model,
            'hub_load': mock_hub_load,
            'tf_function': mock_tf_function,
            'model': mock_model,
            'hub_model': mock_hub_model
        }


@pytest.fixture
def sample_dataset_config():
    """Create a sample dataset configuration for testing."""
    return {
        "audio_folder": "/mock/path/to/audio",
        "class_map": [
            {"name": "NOWA_song", "value": 0},
            {"name": "CEWA_call", "value": 1},
            {"name": "WIWA_song", "value": 2}
        ],
        "backend_model": "PERCH",
        "save_path": "/mock/path/to/save",
        "pretrained_classifier_path": None,
        "window_size": 5.0
    }


@pytest.fixture
def sample_clips_data():
    """Generate sample clip data for testing."""
    clips = []
    for i in range(5):
        clips.append({
            "clip_id": f"clip_{i}",
            "file_name": f"test_file_{i}.wav",
            "file_path": f"/mock/path/test_file_{i}.wav",
            "clip_start": float(i * 5),
            "clip_end": float((i + 1) * 5),
            "score": np.random.random(),
            "annotation_status": [4, 4, 4],  # Unreviewed
            "confidence_predictions": [np.random.random() for _ in range(3)]
        })
    return clips


def create_mock_polars_dataframe(data=None, schema=None):
    """Create a mock Polars DataFrame for testing."""
    mock_df = Mock(spec=pl.DataFrame)

    if data:
        mock_df.__len__.return_value = len(data)
        mock_df.iter_rows.return_value = iter(data)
        mock_df.to_dicts.return_value = data
    else:
        mock_df.__len__.return_value = 0
        mock_df.iter_rows.return_value = iter([])
        mock_df.to_dicts.return_value = []

    if schema:
        mock_df.columns = list(schema.keys())

    return mock_df


def assert_no_real_io_calls():
    """Utility function to verify no real I/O operations occurred during tests."""
    # This can be extended to check for any unpatched I/O operations
    pass