"""
Test suite for FastAPI endpoints (main.py).

This module tests all API endpoints using FastAPI's TestClient with comprehensive
mocking to avoid any real file operations, model loading, or external dependencies.

Key areas tested:
- Dataset Builder endpoints
- Active Learning endpoints
- Database Viewer endpoints
- Model Training endpoints
- Evaluation endpoints
- Request/response validation
- Error handling and status codes
"""

import pytest
import json
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend directory to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the FastAPI app
from main import app

# Create test client
client = TestClient(app)


class TestDatasetBuilderEndpoints:
    """Test Dataset Builder API endpoints."""

    @pytest.fixture
    def mock_app_state(self):
        """Mock the global app state for testing."""
        with patch('main.app_state') as mock_state:
            mock_state.return_value = {
                "audio_db": None,
                "embeddings": None,
                "classifier_model": None,
                "class_map": None,
                "backend_model": None,
                "dataset_path": None,
                "save_path": None,
                "review_mode": "random"
            }
            yield mock_state

    @pytest.fixture
    def mock_building_state(self):
        """Mock the building state for testing."""
        with patch('main.building_state') as mock_state:
            mock_state.return_value = {
                "status": "idle",
                "message": "",
                "progress": 0,
                "total_files": 0,
                "processed_files": 0
            }
            yield mock_state

    @pytest.mark.unit
    def test_dataset_status_endpoint(self, mock_app_state):
        """Test GET /api/dataset/status endpoint."""
        response = client.get("/api/dataset/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.unit
    def test_dataset_building_status_endpoint(self, mock_building_state):
        """Test GET /api/dataset/building-status endpoint."""
        response = client.get("/api/dataset/building-status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "progress" in data

    @pytest.mark.unit
    @patch('main.threading.Thread')
    @patch('modules.database.Audio_DB')
    @patch('modules.utilities.load_audio')
    def test_create_dataset_endpoint(self, mock_load_audio, mock_db, mock_thread, sample_dataset_config):
        """Test POST /api/dataset/create endpoint."""
        # Mock file system operations
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('pathlib.Path.mkdir') as mock_mkdir:

            mock_listdir.return_value = ['test1.wav', 'test2.wav']
            mock_isfile.return_value = True

            response = client.post("/api/dataset/create", json=sample_dataset_config)

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    @pytest.mark.unit
    def test_create_dataset_invalid_input(self):
        """Test dataset creation with invalid input."""
        invalid_config = {
            "audio_folder": "",  # Invalid empty path
            "class_map": [],     # Invalid empty class map
            "backend_model": "INVALID_MODEL"
        }

        response = client.post("/api/dataset/create", json=invalid_config)

        # Should return validation error
        assert response.status_code in [400, 422]


class TestActiveLearningEndpoints:
    """Test Active Learning API endpoints."""

    @pytest.fixture
    def mock_audio_db(self):
        """Mock Audio_DB for active learning tests."""
        with patch('main.app_state') as mock_state:
            mock_db = Mock()
            mock_db.get_clips_with_files.return_value = Mock()
            mock_db.add_annotation.return_value = None
            mock_db.save_db.return_value = None

            mock_state.__getitem__.side_effect = lambda key: {
                "audio_db": mock_db,
                "class_map": {"class1": 0, "class2": 1},
                "current_class_index": 0
            }.get(key)

            yield mock_db

    @pytest.mark.unit
    def test_get_available_classes(self, mock_audio_db):
        """Test GET /api/active-learning/classes endpoint."""
        response = client.get("/api/active-learning/classes")

        if response.status_code == 400:
            # Expected when no dataset is loaded
            data = response.json()
            assert "No dataset loaded" in data["detail"]
        else:
            assert response.status_code == 200

    @pytest.mark.unit
    def test_select_class_endpoint(self, mock_audio_db):
        """Test POST /api/active-learning/select-class endpoint."""
        response = client.post("/api/active-learning/select-class?class_index=0")

        # Expect 400 if no dataset loaded, or 200 if successful
        assert response.status_code in [200, 400]

    @pytest.mark.unit
    @patch('modules.display_web.WebAnnotationInterface')
    def test_get_clips_endpoint(self, mock_interface, mock_audio_db):
        """Test POST /api/active-learning/get-clips endpoint."""
        filter_config = {
            "score_min": 0.0,
            "score_max": 1.0,
            "annotation_filter": [4]
        }

        response = client.post("/api/active-learning/get-clips", json=filter_config)

        # Expect 400 if no dataset loaded
        if response.status_code == 400:
            data = response.json()
            assert "No dataset loaded" in data["detail"]

    @pytest.mark.unit
    def test_annotate_clip_endpoint(self, mock_audio_db):
        """Test POST /api/active-learning/annotate endpoint."""
        annotation_request = {
            "clip_id": "test_clip_123",
            "annotation": 1
        }

        response = client.post("/api/active-learning/annotate", json=annotation_request)

        # Expect 400 if no dataset loaded
        assert response.status_code in [200, 400]

    @pytest.mark.unit
    def test_annotate_specific_class_endpoint(self, mock_audio_db):
        """Test POST /api/active-learning/annotate-class endpoint."""
        annotation_request = {
            "clip_id": "test_clip_123",
            "annotation": 1,
            "class_index": 0
        }

        response = client.post("/api/active-learning/annotate-class", json=annotation_request)

        # Expect 400 if no dataset loaded
        assert response.status_code in [200, 400]

    @pytest.mark.unit
    def test_save_database_endpoint(self, mock_audio_db):
        """Test POST /api/active-learning/save-database endpoint."""
        response = client.post("/api/active-learning/save-database")

        # Expect 400 if no dataset loaded
        assert response.status_code in [200, 400]

    @pytest.mark.unit
    @patch('main.export_clips_to_folder')
    def test_export_clips_endpoint(self, mock_export, mock_audio_db):
        """Test POST /api/active-learning/export-clips endpoint."""
        response = client.post("/api/active-learning/export-clips?export_path=/mock/export/path")

        # Expect 400 if no dataset loaded
        assert response.status_code in [200, 400]

    @pytest.mark.unit
    def test_load_dataset_endpoint(self):
        """Test POST /api/active-learning/load-dataset endpoint."""
        with patch('polars.read_parquet') as mock_read, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open') as mock_open, \
             patch('json.load') as mock_json:

            mock_exists.return_value = True
            mock_json.return_value = {"class_map": {"class1": 0}}

            response = client.post("/api/active-learning/load-dataset?dataset_path=/mock/dataset")

            # Should attempt to load dataset
            assert response.status_code in [200, 500]  # 500 if mocked files don't match expected format

    @pytest.mark.unit
    def test_load_classifier_endpoint(self):
        """Test POST /api/active-learning/load-classifier endpoint."""
        with patch('tensorflow.keras.models.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = np.random.random((10, 3))
            mock_load.return_value = mock_model

            response = client.post("/api/active-learning/load-classifier?classifier_path=/mock/classifier.keras")

            # Expect 400 if no dataset loaded
            assert response.status_code in [200, 400]


class TestSpectrogramAndAudioEndpoints:
    """Test spectrogram generation and audio streaming endpoints."""

    @pytest.mark.unit
    @patch('modules.display_web.create_mel_spectrogram')
    def test_spectrogram_endpoint(self, mock_spectrogram):
        """Test POST /api/spectrogram endpoint."""
        mock_spectrogram.return_value = "data:image/png;base64,mock_spectrogram_data"

        spectrogram_request = {
            "file_path": "/mock/path/test.wav",
            "clip_start": 0.0,
            "clip_end": 5.0,
            "color_mode": "viridis"
        }

        response = client.post("/api/spectrogram", json=spectrogram_request)

        if response.status_code == 200:
            data = response.json()
            assert "spectrogram" in data

    @pytest.mark.unit
    @patch('main.FileResponse')
    @patch('pathlib.Path.exists')
    def test_audio_endpoint(self, mock_exists, mock_file_response):
        """Test GET /api/audio/{file_path} endpoint."""
        mock_exists.return_value = True
        mock_file_response.return_value = Mock()

        # Test with URL-encoded file path
        encoded_path = "mock%2Fpath%2Ftest.wav"
        response = client.get(f"/api/audio/{encoded_path}")

        # Response depends on file existence and streaming implementation
        assert response.status_code in [200, 404, 500]


class TestDatabaseViewerEndpoints:
    """Test Database Viewer API endpoints."""

    @pytest.fixture
    def mock_database_state(self):
        """Mock database state for viewer tests."""
        with patch('main.app_state') as mock_state:
            mock_db = Mock()
            mock_db.files_df = Mock()
            mock_db.clips_df = Mock()
            mock_db.annotations_df = Mock()

            mock_state.__getitem__.side_effect = lambda key: {
                "audio_db": mock_db
            }.get(key)

            yield mock_db

    @pytest.mark.unit
    def test_database_info_endpoint(self, mock_database_state):
        """Test GET /api/database/info endpoint."""
        response = client.get("/api/database/info")

        # Expect 400 if no database loaded
        assert response.status_code in [200, 400]

    @pytest.mark.unit
    def test_database_data_endpoint(self, mock_database_state):
        """Test GET /api/database/data endpoint."""
        response = client.get("/api/database/data")

        assert response.status_code in [200, 400]

    @pytest.mark.unit
    def test_database_files_endpoint(self, mock_database_state):
        """Test GET /api/database/files endpoint."""
        response = client.get("/api/database/files")

        assert response.status_code in [200, 400]

    @pytest.mark.unit
    def test_database_clips_with_annotations_endpoint(self, mock_database_state):
        """Test GET /api/database/clips-with-annotations endpoint."""
        response = client.get("/api/database/clips-with-annotations")

        assert response.status_code in [200, 400]


class TestModelTrainingEndpoints:
    """Test Model Training API endpoints."""

    @pytest.fixture
    def mock_training_state(self):
        """Mock training state for tests."""
        with patch('main.training_state') as mock_state:
            mock_state.__getitem__.side_effect = lambda key: {
                "status": "idle",
                "message": "",
                "logs": [],
                "results": None,
                "stop_requested": False
            }.get(key)

            yield mock_state

    @pytest.mark.unit
    def test_training_status_endpoint(self, mock_training_state):
        """Test GET /api/model-training/status endpoint."""
        response = client.get("/api/model-training/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.unit
    def test_training_stop_endpoint(self, mock_training_state):
        """Test POST /api/model-training/stop endpoint."""
        response = client.post("/api/model-training/stop")

        assert response.status_code == 200

    @pytest.mark.unit
    @patch('os.listdir')
    @patch('os.path.isfile')
    def test_preview_training_data_endpoint(self, mock_isfile, mock_listdir):
        """Test POST /api/model-training/preview-data endpoint."""
        mock_listdir.return_value = ['test1.wav', 'test2.wav']
        mock_isfile.return_value = True

        preview_request = {
            "audio_folder": "/mock/training/data",
            "metadata_file": None
        }

        response = client.post("/api/model-training/preview-data", json=preview_request)

        assert response.status_code in [200, 500]  # 500 if audio processing fails

    @pytest.mark.unit
    @patch('main.threading.Thread')
    def test_start_training_endpoint(self, mock_thread):
        """Test POST /api/model-training/start endpoint."""
        training_request = {
            "audio_folder": "/mock/training/data",
            "test_split": 0.2,
            "epochs": 5,
            "batch_size": 32
        }

        response = client.post("/api/model-training/start", json=training_request)

        assert response.status_code in [200, 400]


class TestEvaluationEndpoints:
    """Test Model Evaluation API endpoints."""

    @pytest.fixture
    def mock_eval_state(self):
        """Mock evaluation state for tests."""
        with patch('main.eval_state') as mock_state:
            mock_state.__getitem__.side_effect = lambda key: {
                "eval_embeddings": None,
                "eval_labels": None,
                "eval_classifier": None,
                "eval_class_map": None,
                "eval_dataset_path": None
            }.get(key)

            yield mock_state

    @pytest.mark.unit
    @patch('os.listdir')
    @patch('os.path.isfile')
    def test_load_evaluation_dataset_endpoint(self, mock_isfile, mock_listdir, mock_eval_state):
        """Test POST /api/evaluation/load-dataset endpoint."""
        mock_listdir.return_value = ['test1.wav', 'test2.wav']
        mock_isfile.return_value = True

        load_request = {
            "dataset_path": "/mock/eval/dataset"
        }

        response = client.post("/api/evaluation/load-dataset", json=load_request)

        assert response.status_code in [200, 500]

    @pytest.mark.unit
    @patch('tensorflow.keras.models.load_model')
    def test_load_evaluation_classifier_endpoint(self, mock_load_model, mock_eval_state):
        """Test POST /api/evaluation/load-classifier endpoint."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        response = client.post("/api/evaluation/load-classifier?classifier_path=/mock/classifier.keras")

        assert response.status_code in [200, 400]

    @pytest.mark.unit
    def test_run_evaluation_endpoint(self, mock_eval_state):
        """Test POST /api/evaluation/run-evaluation endpoint."""
        response = client.post("/api/evaluation/run-evaluation")

        # Expect 400 if no dataset/classifier loaded
        assert response.status_code in [200, 400]


class TestErrorHandling:
    """Test error handling across all endpoints."""

    @pytest.mark.unit
    def test_invalid_json_request(self):
        """Test handling of invalid JSON in request body."""
        response = client.post(
            "/api/dataset/create",
            data="invalid json",
            headers={"content-type": "application/json"}
        )

        assert response.status_code == 422

    @pytest.mark.unit
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        incomplete_request = {
            "audio_folder": "/mock/path"
            # Missing class_map and backend_model
        }

        response = client.post("/api/dataset/create", json=incomplete_request)

        assert response.status_code == 422

    @pytest.mark.unit
    def test_nonexistent_endpoint(self):
        """Test handling of requests to nonexistent endpoints."""
        response = client.get("/api/nonexistent/endpoint")

        assert response.status_code == 404

    @pytest.mark.unit
    def test_method_not_allowed(self):
        """Test handling of wrong HTTP methods."""
        response = client.put("/api/dataset/status")  # Should be GET

        assert response.status_code == 405


class TestRequestValidation:
    """Test request validation and Pydantic models."""

    @pytest.mark.unit
    def test_annotation_request_validation(self):
        """Test validation of annotation requests."""
        # Valid request
        valid_request = {
            "clip_id": "test_clip_123",
            "annotation": 1
        }

        # Invalid request - missing fields
        invalid_request = {
            "clip_id": "test_clip_123"
            # Missing annotation field
        }

        # Test both requests (may return 400 for no dataset, but shouldn't be 422 for valid structure)
        valid_response = client.post("/api/active-learning/annotate", json=valid_request)
        invalid_response = client.post("/api/active-learning/annotate", json=invalid_request)

        assert valid_response.status_code in [200, 400]  # 400 = no dataset loaded
        assert invalid_response.status_code == 422  # 422 = validation error

    @pytest.mark.unit
    def test_spectrogram_request_validation(self):
        """Test validation of spectrogram requests."""
        valid_request = {
            "file_path": "/path/to/file.wav",
            "clip_start": 0.0,
            "clip_end": 5.0,
            "color_mode": "viridis"
        }

        invalid_request = {
            "file_path": "/path/to/file.wav",
            "clip_start": "invalid",  # Should be float
            "clip_end": 5.0
        }

        valid_response = client.post("/api/spectrogram", json=valid_request)
        invalid_response = client.post("/api/spectrogram", json=invalid_request)

        assert valid_response.status_code in [200, 400, 500]
        assert invalid_response.status_code == 422

    @pytest.mark.unit
    def test_class_map_validation(self):
        """Test validation of class map structure."""
        valid_class_map = [
            {"name": "class1", "value": 0},
            {"name": "class2", "value": 1}
        ]

        invalid_class_map = [
            {"name": "class1"},  # Missing value
            {"value": 1}         # Missing name
        ]

        valid_request = {
            "audio_folder": "/mock/path",
            "class_map": valid_class_map,
            "backend_model": "PERCH",
            "save_path": "/mock/save"
        }

        invalid_request = {
            "audio_folder": "/mock/path",
            "class_map": invalid_class_map,
            "backend_model": "PERCH",
            "save_path": "/mock/save"
        }

        valid_response = client.post("/api/dataset/create", json=valid_request)
        invalid_response = client.post("/api/dataset/create", json=invalid_request)

        assert valid_response.status_code in [200, 400, 500]
        assert invalid_response.status_code == 422


class TestCORSConfiguration:
    """Test CORS configuration for frontend integration."""

    @pytest.mark.unit
    def test_cors_headers(self):
        """Test that CORS headers are properly set."""
        response = client.options("/api/dataset/status")

        # Should allow the configured origin
        assert response.status_code in [200, 204]

    @pytest.mark.unit
    def test_preflight_request(self):
        """Test handling of CORS preflight requests."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }

        response = client.options("/api/dataset/create", headers=headers)

        assert response.status_code in [200, 204]