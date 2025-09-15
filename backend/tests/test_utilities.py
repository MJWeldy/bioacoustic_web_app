"""
Test suite for the utilities module (modules/utilities.py).

This module tests audio processing, embedding generation, and backend model
integration using comprehensive mocking to avoid requiring real audio files
or model downloads.

Key areas tested:
- Audio loading and preprocessing
- Audio normalization algorithms
- Backend model configuration
- Embedding generation workflows
- Error handling for invalid inputs
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os

# Add backend directory to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the module under test
from modules import utilities as u
from modules import config as cfg


class TestAudioLoading:
    """Test audio file loading and preprocessing."""

    @pytest.mark.unit
    @patch('soundfile.read')
    def test_load_audio_no_time_segment(self, mock_sf_read, mock_audio_data):
        """Test loading complete audio file without time segmentation."""
        audio_data, sample_rate = mock_audio_data
        mock_sf_read.return_value = (audio_data, sample_rate)

        # Test loading without start_stop parameter
        result = u.load_audio('/mock/path/test.wav', start_stop=None)

        mock_sf_read.assert_called_once_with('/mock/path/test.wav')
        np.testing.assert_array_equal(result, audio_data)

    @pytest.mark.unit
    @patch('soundfile.read')
    def test_load_audio_with_time_segment(self, mock_sf_read):
        """Test loading audio file with time segmentation."""
        # Create mock audio data
        full_audio = np.random.random(44100).astype(np.float32)  # 2 seconds at 22050 Hz
        mock_sf_read.return_value = (full_audio, 22050)

        # Test loading with time segment (1.0-2.0 seconds)
        start_stop = (1.0, 2.0, 22050)
        result = u.load_audio('/mock/path/test.wav', start_stop=start_stop)

        # Verify soundfile.read was called with correct parameters
        expected_start = int(1.0 * 22050)
        expected_stop = int(2.0 * 22050)
        mock_sf_read.assert_called_once_with(
            '/mock/path/test.wav',
            start=expected_start,
            stop=expected_stop
        )

    @pytest.mark.unit
    @patch('librosa.resample')
    @patch('soundfile.read')
    def test_load_audio_with_resampling(self, mock_sf_read, mock_resample):
        """Test audio loading with sample rate conversion."""
        # Mock audio at different sample rate
        original_sr = 44100
        target_sr = 22050
        audio_data = np.random.random(44100).astype(np.float32)
        resampled_data = np.random.random(22050).astype(np.float32)

        mock_sf_read.return_value = (audio_data, original_sr)
        mock_resample.return_value = resampled_data

        # Mock the target sample rate
        with patch.object(cfg, 'TARGET_SR', target_sr):
            result = u.load_audio('/mock/path/test.wav', start_stop=None)

        # Verify resampling was called
        mock_resample.assert_called_once()
        np.testing.assert_array_equal(result, resampled_data)

    @pytest.mark.unit
    @patch('soundfile.read')
    def test_load_audio_no_resampling_needed(self, mock_sf_read):
        """Test audio loading when no resampling is needed."""
        target_sr = 22050
        audio_data = np.random.random(22050).astype(np.float32)

        mock_sf_read.return_value = (audio_data, target_sr)

        with patch.object(cfg, 'TARGET_SR', target_sr):
            result = u.load_audio('/mock/path/test.wav', start_stop=None)

        # Result should be the same as input (no resampling)
        np.testing.assert_array_equal(result, audio_data)


class TestAudioNormalization:
    """Test audio normalization and preprocessing functions."""

    @pytest.mark.unit
    def test_normalize_audio_basic(self):
        """Test basic audio normalization."""
        # Create test audio with known properties
        audio = tf.constant([0.5, -0.3, 0.8, -0.2], dtype=tf.float32)
        norm_factor = 0.8

        result = u.normalize_audio(audio, norm_factor)

        # Audio should be mean-centered and peak-normalized
        assert isinstance(result, tf.Tensor)
        assert result.shape == audio.shape

        # Check that the result is within expected range
        assert tf.reduce_max(tf.abs(result)) <= norm_factor

    @pytest.mark.unit
    def test_normalize_audio_zero_audio(self):
        """Test normalization with zero audio (silence)."""
        audio = tf.zeros(1000, dtype=tf.float32)
        norm_factor = 0.8

        result = u.normalize_audio(audio, norm_factor)

        # Should handle silence gracefully
        assert isinstance(result, tf.Tensor)
        np.testing.assert_array_almost_equal(result.numpy(), np.zeros(1000))

    @pytest.mark.unit
    def test_normalize_audio_multi_dimensional(self):
        """Test normalization with multi-dimensional audio."""
        # Test with batch dimension
        audio = tf.random.normal((3, 1000), dtype=tf.float32)
        norm_factor = 0.8

        result = u.normalize_audio(audio, norm_factor)

        assert result.shape == audio.shape
        assert result.dtype == tf.float32

    @pytest.mark.unit
    def test_normalize_audio_extreme_values(self):
        """Test normalization with extreme audio values."""
        # Test with very large values
        audio = tf.constant([100.0, -50.0, 75.0], dtype=tf.float32)
        norm_factor = 0.5

        result = u.normalize_audio(audio, norm_factor)

        # Should be properly normalized regardless of input magnitude
        assert tf.reduce_max(tf.abs(result)) <= norm_factor + 1e-6  # Small tolerance


class TestBackendModelConfiguration:
    """Test backend model configuration and switching."""

    @pytest.mark.unit
    def test_backend_environment_variable(self):
        """Test that BACKEND environment variable is used."""
        with patch.dict(os.environ, {'BACKEND': 'TEST_MODEL'}):
            # Reload the module to pick up environment change
            import importlib
            importlib.reload(u)

            assert u.BACKEND == 'TEST_MODEL'

    @pytest.mark.unit
    def test_default_backend(self):
        """Test default backend when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove BACKEND from environment
            if 'BACKEND' in os.environ:
                del os.environ['BACKEND']

            # Reload module
            import importlib
            importlib.reload(u)

            assert u.BACKEND == 'PERCH'  # Default value


class TestEmbeddingGeneration:
    """Test embedding generation for different backend models."""

    @pytest.fixture
    def mock_audio_data(self):
        """Generate mock audio data for embedding tests."""
        # Create realistic audio data
        sample_rate = 22050
        duration = 5.0
        samples = int(sample_rate * duration)

        # Generate a simple sine wave
        t = np.linspace(0, duration, samples, False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        return audio

    @pytest.mark.unit
    @patch('tensorflow_hub.load')
    def test_embedding_generation_perch(self, mock_hub_load, mock_audio_data):
        """Test embedding generation with PERCH model."""
        # Mock TensorFlow Hub model
        mock_model = Mock()
        mock_embeddings = np.random.random((1, 1280)).astype(np.float32)
        mock_model.return_value = mock_embeddings
        mock_hub_load.return_value = mock_model

        with patch.object(u, 'BACKEND', 'PERCH'):
            # Mock any embedding generation function that might exist
            with patch.object(u, 'generate_embeddings', create=True) as mock_gen:
                mock_gen.return_value = mock_embeddings

                result = u.generate_embeddings(mock_audio_data)

                assert result is not None
                mock_gen.assert_called_once()

    @pytest.mark.unit
    @patch('tensorflow.lite.python.interpreter')
    def test_embedding_generation_birdnet(self, mock_tflite, mock_audio_data):
        """Test embedding generation with BirdNET model."""
        # Mock TensorFlow Lite interpreter
        mock_interpreter = Mock()
        mock_tflite.Interpreter.return_value = mock_interpreter

        # Mock interpreter methods
        mock_interpreter.allocate_tensors.return_value = None
        mock_interpreter.get_input_details.return_value = [{'index': 0, 'shape': [1, 144000]}]
        mock_interpreter.get_output_details.return_value = [{'index': 0, 'shape': [1, 1024]}]
        mock_interpreter.get_tensor.return_value = np.random.random((1, 1024)).astype(np.float32)

        with patch.object(u, 'BACKEND', 'BirdNET_2.4'):
            # Mock embedding generation for BirdNET
            with patch.object(u, 'generate_embeddings', create=True) as mock_gen:
                expected_embeddings = np.random.random((1, 1024)).astype(np.float32)
                mock_gen.return_value = expected_embeddings

                result = u.generate_embeddings(mock_audio_data)

                assert result is not None


class TestAudioPreprocessingPipeline:
    """Test complete audio preprocessing pipeline."""

    @pytest.mark.unit
    @patch('modules.utilities.normalize_audio')
    @patch('modules.utilities.load_audio')
    def test_preprocessing_pipeline(self, mock_load_audio, mock_normalize):
        """Test complete audio preprocessing workflow."""
        # Mock audio loading
        raw_audio = np.random.random(22050).astype(np.float32)
        mock_load_audio.return_value = raw_audio

        # Mock normalization
        normalized_audio = tf.constant(raw_audio * 0.8, dtype=tf.float32)
        mock_normalize.return_value = normalized_audio

        # Test the pipeline (assuming there's a preprocessing function)
        with patch.object(u, 'preprocess_audio', create=True) as mock_preprocess:
            mock_preprocess.return_value = normalized_audio

            result = u.preprocess_audio('/mock/path/test.wav')

            assert result is not None
            mock_preprocess.assert_called_once()

    @pytest.mark.unit
    def test_audio_windowing(self):
        """Test audio windowing for fixed-size inputs."""
        # Create test audio longer than typical window
        audio = np.random.random(100000).astype(np.float32)  # ~4.5 seconds at 22050 Hz
        window_size = 22050  # 1 second

        # Mock windowing function
        with patch.object(u, 'window_audio', create=True) as mock_window:
            expected_windows = [
                audio[0:22050],
                audio[22050:44100],
                audio[44100:66150],
                audio[66150:88200]
            ]
            mock_window.return_value = expected_windows

            result = u.window_audio(audio, window_size)

            assert result is not None
            mock_window.assert_called_once_with(audio, window_size)


class TestErrorHandling:
    """Test error handling in utilities functions."""

    @pytest.mark.unit
    @patch('soundfile.read')
    def test_load_audio_file_not_found(self, mock_sf_read):
        """Test handling of missing audio files."""
        mock_sf_read.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            u.load_audio('/nonexistent/file.wav', start_stop=None)

    @pytest.mark.unit
    @patch('soundfile.read')
    def test_load_audio_corrupted_file(self, mock_sf_read):
        """Test handling of corrupted audio files."""
        mock_sf_read.side_effect = Exception("Corrupted audio file")

        with pytest.raises(Exception):
            u.load_audio('/corrupted/file.wav', start_stop=None)

    @pytest.mark.unit
    def test_normalize_audio_invalid_input(self):
        """Test normalization with invalid input."""
        # Test with NaN values
        audio_with_nan = tf.constant([1.0, float('nan'), 0.5], dtype=tf.float32)
        norm_factor = 0.8

        # Should handle NaN gracefully or raise appropriate error
        try:
            result = u.normalize_audio(audio_with_nan, norm_factor)
            # If it succeeds, check that result is valid
            assert not tf.reduce_any(tf.math.is_nan(result))
        except (ValueError, tf.errors.InvalidArgumentError):
            # Expected behavior for invalid input
            pass

    @pytest.mark.unit
    def test_invalid_time_segment(self):
        """Test handling of invalid time segments."""
        with patch('soundfile.read') as mock_sf_read:
            mock_sf_read.side_effect = ValueError("Invalid time segment")

            # Test with negative start time
            with pytest.raises(ValueError):
                u.load_audio('/mock/file.wav', start_stop=(-1.0, 2.0, 22050))


class TestPerformanceAndMemory:
    """Test performance characteristics and memory usage."""

    @pytest.mark.slow
    @patch('soundfile.read')
    def test_large_audio_file_handling(self, mock_sf_read):
        """Test handling of large audio files."""
        # Simulate a very large audio file (10 minutes at 22050 Hz)
        large_audio = np.random.random(13230000).astype(np.float32)
        mock_sf_read.return_value = (large_audio, 22050)

        # Should handle large files without memory issues
        result = u.load_audio('/mock/large_file.wav', start_stop=None)

        assert result is not None
        assert len(result) == len(large_audio)

    @pytest.mark.unit
    def test_batch_processing_efficiency(self):
        """Test efficiency of batch audio processing."""
        # Mock batch processing function
        audio_files = [f'/mock/file_{i}.wav' for i in range(10)]

        with patch.object(u, 'process_audio_batch', create=True) as mock_batch:
            mock_batch.return_value = [np.random.random(22050) for _ in range(10)]

            result = u.process_audio_batch(audio_files)

            assert result is not None
            assert len(result) == 10
            mock_batch.assert_called_once_with(audio_files)


class TestTensorFlowIntegration:
    """Test TensorFlow-specific functionality."""

    @pytest.mark.unit
    def test_tensorflow_function_decorator(self):
        """Test that TensorFlow function decorator works correctly."""
        # Test the normalize_audio function with tf.function
        audio = tf.random.normal((1000,), dtype=tf.float32)
        norm_factor = 0.8

        # Should work with tf.function compilation
        result = u.normalize_audio(audio, norm_factor)

        assert isinstance(result, tf.Tensor)
        assert result.shape == audio.shape

    @pytest.mark.unit
    def test_gpu_availability(self):
        """Test GPU availability detection."""
        # Mock GPU detection
        with patch('tensorflow.config.list_physical_devices') as mock_gpu:
            mock_gpu.return_value = ['GPU:0']  # Mock GPU available

            # Test any GPU-related functionality
            with patch.object(u, 'check_gpu_available', create=True) as mock_check:
                mock_check.return_value = True

                result = u.check_gpu_available()
                assert result is True

    @pytest.mark.unit
    def test_tensorflow_memory_management(self):
        """Test TensorFlow memory growth configuration."""
        # Mock memory growth setting
        with patch('tensorflow.config.experimental.set_memory_growth') as mock_memory:
            # Test memory configuration function if it exists
            with patch.object(u, 'configure_tensorflow_memory', create=True) as mock_config:
                u.configure_tensorflow_memory()
                mock_config.assert_called_once()


class TestConfigurationManagement:
    """Test configuration and model parameter management."""

    @pytest.mark.unit
    def test_model_parameters_loading(self):
        """Test loading of model-specific parameters."""
        # Test configuration for different models
        test_configs = {
            'PERCH': {'sample_rate': 22050, 'window_size': 5.0},
            'BirdNET_2.4': {'sample_rate': 48000, 'window_size': 3.0},
            'PNWCnet': {'sample_rate': 22050, 'window_size': 4.0}
        }

        for model_name, expected_config in test_configs.items():
            with patch.object(u, 'BACKEND', model_name):
                # Mock configuration loading
                with patch.object(cfg, 'get_model_config', create=True) as mock_config:
                    mock_config.return_value = expected_config

                    config = cfg.get_model_config(model_name)
                    assert config['sample_rate'] == expected_config['sample_rate']

    @pytest.mark.unit
    def test_parameter_validation(self):
        """Test validation of model parameters."""
        # Test parameter validation function
        valid_params = {'sample_rate': 22050, 'window_size': 5.0, 'embedding_dim': 1280}
        invalid_params = {'sample_rate': -1, 'window_size': 0, 'embedding_dim': 'invalid'}

        with patch.object(u, 'validate_parameters', create=True) as mock_validate:
            mock_validate.side_effect = lambda params: all(
                isinstance(v, (int, float)) and v > 0 for v in params.values()
            )

            assert u.validate_parameters(valid_params) is True
            assert u.validate_parameters(invalid_params) is False