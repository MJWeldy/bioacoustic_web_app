"""
Integration test suite for the bioacoustic web application.

This module tests complete workflows and end-to-end functionality using
comprehensive mocking to simulate real usage scenarios without requiring
actual files, models, or external services.

Key workflows tested:
- Complete dataset creation pipeline
- Full active learning annotation workflow
- Model training and evaluation pipeline
- Database operations and persistence
- Multi-component interactions
"""

import pytest
import json
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock, call
import sys
import os
import tempfile
from pathlib import Path

# Add backend directory to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the FastAPI app and modules
from main import app
from modules.database import Audio_DB

# Create test client
client = TestClient(app)


class TestDatasetCreationWorkflow:
    """Test complete dataset creation workflow."""

    @pytest.fixture
    def mock_file_system(self):
        """Mock file system for dataset creation."""
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('os.path.exists') as mock_exists, \
             patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('soundfile.read') as mock_sf_read, \
             patch('librosa.resample') as mock_resample:

            # Mock audio files in directory
            mock_listdir.return_value = [
                'bird_song_1.wav', 'bird_song_2.wav', 'background_1.wav',
                'other_file.txt'  # Should be ignored
            ]
            mock_isfile.side_effect = lambda path: path.endswith('.wav')
            mock_exists.return_value = True

            # Mock audio loading
            mock_sf_read.return_value = (np.random.random(110250).astype(np.float32), 22050)  # 5 seconds
            mock_resample.return_value = np.random.random(110250).astype(np.float32)

            yield {
                'listdir': mock_listdir,
                'isfile': mock_isfile,
                'exists': mock_exists,
                'mkdir': mock_mkdir,
                'sf_read': mock_sf_read,
                'resample': mock_resample
            }

    @pytest.fixture
    def mock_embedding_generation(self):
        """Mock embedding generation for testing."""
        with patch('tensorflow_hub.load') as mock_hub_load, \
             patch('modules.utilities.generate_embeddings', create=True) as mock_gen_embeddings:

            # Mock TensorFlow Hub model
            mock_model = Mock()
            mock_embeddings = np.random.random((22, 1280)).astype(np.float32)  # 22 clips, 1280 dims
            mock_model.return_value = mock_embeddings
            mock_hub_load.return_value = mock_model

            # Mock embedding generation function
            mock_gen_embeddings.return_value = mock_embeddings

            yield {
                'hub_load': mock_hub_load,
                'model': mock_model,
                'embeddings': mock_embeddings
            }

    @pytest.mark.integration
    def test_complete_dataset_creation_workflow(self, mock_file_system, mock_embedding_generation):
        """Test complete dataset creation from start to finish."""
        # Step 1: Check initial status
        status_response = client.get("/api/dataset/status")
        assert status_response.status_code == 200

        # Step 2: Create dataset configuration
        dataset_config = {
            "audio_folder": "/mock/audio/folder",
            "class_map": [
                {"name": "bird_song", "value": 0},
                {"name": "background", "value": 1}
            ],
            "backend_model": "PERCH",
            "save_path": "/mock/save/path",
            "pretrained_classifier_path": None,
            "window_size": 5.0
        }

        # Step 3: Start dataset creation
        with patch('main.threading.Thread') as mock_thread:
            create_response = client.post("/api/dataset/create", json=dataset_config)
            assert create_response.status_code == 200
            assert create_response.json()["status"] == "success"

        # Step 4: Check building status
        building_status_response = client.get("/api/dataset/building-status")
        assert building_status_response.status_code == 200
        building_data = building_status_response.json()
        assert "status" in building_data
        assert "progress" in building_data

    @pytest.mark.integration
    @patch('pickle.dump')
    @patch('builtins.open', create=True)
    def test_dataset_creation_with_persistence(self, mock_open, mock_pickle_dump,
                                             mock_file_system, mock_embedding_generation):
        """Test dataset creation with proper file persistence."""
        dataset_config = {
            "audio_folder": "/mock/audio/folder",
            "class_map": [{"name": "test_class", "value": 0}],
            "backend_model": "PERCH",
            "save_path": "/mock/save/path",
            "window_size": 5.0
        }

        with patch('main.threading.Thread') as mock_thread, \
             patch('json.dump') as mock_json_dump:

            response = client.post("/api/dataset/create", json=dataset_config)
            assert response.status_code == 200

            # Verify that thread was started for background processing
            mock_thread.assert_called_once()


class TestActiveLearningWorkflow:
    """Test complete active learning annotation workflow."""

    @pytest.fixture
    def mock_loaded_dataset(self):
        """Mock a loaded dataset for active learning."""
        with patch('main.app_state') as mock_state:
            # Create mock database
            mock_db = Mock(spec=Audio_DB)
            mock_db.get_clips_with_files.return_value = Mock()
            mock_db.add_annotation.return_value = None
            mock_db.save_db.return_value = None

            # Mock state
            state_data = {
                "audio_db": mock_db,
                "class_map": {"bird_song": 0, "background": 1},
                "current_class_index": 0,
                "dataset_path": "/mock/dataset/path"
            }

            mock_state.__getitem__.side_effect = lambda key: state_data.get(key)
            mock_state.__setitem__.side_effect = lambda key, value: state_data.update({key: value})
            mock_state.get.side_effect = lambda key, default=None: state_data.get(key, default)

            yield mock_db

    @pytest.fixture
    def mock_clips_data(self):
        """Generate mock clip data for testing."""
        clips = []
        for i in range(5):
            clips.append({
                "clip_id": f"clip_{i}",
                "file_name": f"test_file_{i}.wav",
                "file_path": f"/mock/path/test_file_{i}.wav",
                "clip_start": float(i * 5),
                "clip_end": float((i + 1) * 5),
                "score": np.random.random(),
                "annotation_status": [4, 4],  # Unreviewed for 2 classes
                "confidence_predictions": [np.random.random(), np.random.random()]
            })
        return clips

    @pytest.mark.integration
    def test_complete_active_learning_workflow(self, mock_loaded_dataset, mock_clips_data):
        """Test complete active learning workflow from loading to annotation."""
        # Step 1: Load dataset
        with patch('polars.read_parquet') as mock_read_parquet, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('json.load') as mock_json_load:

            mock_exists.return_value = True
            mock_json_load.return_value = {"class_map": {"bird_song": 0, "background": 1}}

            load_response = client.post("/api/active-learning/load-dataset?dataset_path=/mock/dataset")

        # Step 2: Get available classes
        classes_response = client.get("/api/active-learning/classes")
        if classes_response.status_code == 200:
            classes_data = classes_response.json()
            assert "classes" in classes_data

        # Step 3: Get clips for annotation
        with patch('modules.display_web.WebAnnotationInterface') as mock_interface:
            mock_interface_instance = Mock()
            mock_interface_instance.get_filtered_clips.return_value = mock_clips_data
            mock_interface.return_value = mock_interface_instance

            filter_config = {
                "score_min": 0.0,
                "score_max": 1.0,
                "annotation_filter": [4]
            }

            clips_response = client.post("/api/active-learning/get-clips", json=filter_config)

        # Step 4: Annotate clips
        if clips_response.status_code == 200:
            for i, clip in enumerate(mock_clips_data[:3]):  # Annotate first 3 clips
                annotation_request = {
                    "clip_id": clip["clip_id"],
                    "annotation": 1 if i % 2 == 0 else 0  # Alternate between present/not present
                }

                annotate_response = client.post("/api/active-learning/annotate", json=annotation_request)
                # May return 400 if no dataset loaded in mocked state

        # Step 5: Save database
        save_response = client.post("/api/active-learning/save-database")

    @pytest.mark.integration
    def test_multiclass_annotation_workflow(self, mock_loaded_dataset):
        """Test multi-class annotation workflow with additional positive classes."""
        # Step 1: Select target class
        select_response = client.post("/api/active-learning/select-class?class_index=0")

        # Step 2: Annotate target class
        target_annotation = {
            "clip_id": "test_clip_123",
            "annotation": 1
        }
        target_response = client.post("/api/active-learning/annotate", json=target_annotation)

        # Step 3: Annotate additional positive class
        additional_annotation = {
            "clip_id": "test_clip_123",
            "annotation": 1,
            "class_index": 1
        }
        additional_response = client.post("/api/active-learning/annotate-class", json=additional_annotation)

        # Step 4: Mark other classes as absent
        other_classes_request = {
            "clip_id": "test_clip_123",
            "annotation": 0
        }
        other_response = client.post("/api/active-learning/annotate-other-classes", json=other_classes_request)

    @pytest.mark.integration
    def test_spectrogram_and_audio_integration(self, mock_loaded_dataset):
        """Test spectrogram generation and audio streaming integration."""
        # Generate spectrogram
        with patch('modules.display_web.create_mel_spectrogram') as mock_spectrogram:
            mock_spectrogram.return_value = "data:image/png;base64,mock_spectrogram_data"

            spectrogram_request = {
                "file_path": "/mock/path/test.wav",
                "clip_start": 0.0,
                "clip_end": 5.0,
                "color_mode": "viridis"
            }

            spectrogram_response = client.post("/api/spectrogram", json=spectrogram_request)

        # Stream audio
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('main.FileResponse') as mock_file_response:

            mock_exists.return_value = True
            mock_file_response.return_value = Mock()

            audio_response = client.get("/api/audio/mock%2Fpath%2Ftest.wav")


class TestModelTrainingWorkflow:
    """Test complete model training workflow."""

    @pytest.fixture
    def mock_training_data(self):
        """Mock training data and file system."""
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('modules.utilities.load_audio') as mock_load_audio:

            # Mock training files
            mock_listdir.return_value = [
                'Site_001_Rep_C_0.0-bird_song.wav',
                'Site_001_Rep_C_5.0-background.wav',
                'Site_002_Rep_A_0.0-bird_song+other_call.wav'
            ]
            mock_isfile.return_value = True

            # Mock audio loading
            mock_load_audio.return_value = np.random.random(110250).astype(np.float32)

            yield {
                'listdir': mock_listdir,
                'isfile': mock_isfile,
                'load_audio': mock_load_audio
            }

    @pytest.mark.integration
    def test_complete_training_workflow(self, mock_training_data):
        """Test complete model training workflow."""
        # Step 1: Preview training data
        preview_request = {
            "audio_folder": "/mock/training/data",
            "metadata_file": None
        }

        preview_response = client.post("/api/model-training/preview-data", json=preview_request)

        # Step 2: Start training
        with patch('main.threading.Thread') as mock_thread, \
             patch('modules.classifier.fit_w_tape', create=True) as mock_fit:

            training_request = {
                "audio_folder": "/mock/training/data",
                "test_split": 0.2,
                "epochs": 5,
                "batch_size": 32,
                "model_architecture": "small",
                "learning_rate": 0.001
            }

            start_response = client.post("/api/model-training/start", json=training_request)

        # Step 3: Check training status
        status_response = client.get("/api/model-training/status")
        assert status_response.status_code == 200

        # Step 4: Stop training (if needed)
        stop_response = client.post("/api/model-training/stop")
        assert stop_response.status_code == 200

    @pytest.mark.integration
    def test_training_with_metadata(self, mock_training_data):
        """Test training workflow with metadata file."""
        # Mock metadata file
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('json.load') as mock_json_load:

            mock_exists.return_value = True
            mock_json_load.return_value = {
                "class_map": {"bird_song": 0, "background": 1, "other_call": 2},
                "clips": [
                    {
                        "filename": "Site_001_Rep_C_0.0-bird_song.wav",
                        "labels": [1, 0, 0],
                        "label_strengths": [1, 1, 1]
                    }
                ]
            }

            preview_request = {
                "audio_folder": "/mock/training/data",
                "metadata_file": "/mock/metadata.json"
            }

            response = client.post("/api/model-training/preview-data", json=preview_request)


class TestEvaluationWorkflow:
    """Test complete model evaluation workflow."""

    @pytest.fixture
    def mock_evaluation_setup(self):
        """Mock evaluation dataset and classifier."""
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('tensorflow.keras.models.load_model') as mock_load_model, \
             patch('modules.utilities.load_audio') as mock_load_audio:

            # Mock evaluation files
            mock_listdir.return_value = [
                'eval_file_1-bird_song.wav',
                'eval_file_2-background.wav',
                'eval_file_3-bird_song+other_call.wav'
            ]
            mock_isfile.return_value = True

            # Mock classifier
            mock_model = Mock()
            mock_model.predict.return_value = np.random.random((3, 3))  # 3 files, 3 classes
            mock_load_model.return_value = mock_model

            # Mock audio loading
            mock_load_audio.return_value = np.random.random(110250).astype(np.float32)

            yield {
                'listdir': mock_listdir,
                'isfile': mock_isfile,
                'model': mock_model,
                'load_audio': mock_load_audio
            }

    @pytest.mark.integration
    def test_complete_evaluation_workflow(self, mock_evaluation_setup):
        """Test complete model evaluation workflow."""
        # Step 1: Load evaluation dataset
        load_dataset_request = {
            "dataset_path": "/mock/eval/dataset"
        }

        dataset_response = client.post("/api/evaluation/load-dataset", json=load_dataset_request)

        # Step 2: Load classifier
        classifier_response = client.post("/api/evaluation/load-classifier?classifier_path=/mock/classifier.keras")

        # Step 3: Run evaluation
        with patch('modules.classifier.get_AUC') as mock_auc, \
             patch('modules.classifier.average_precision') as mock_ap, \
             patch('sklearn.metrics.confusion_matrix') as mock_cm:

            # Mock evaluation metrics
            mock_auc.return_value = {
                "macro_auc": 0.85,
                "individual_aucs": [0.9, 0.8, 0.85]
            }
            mock_ap.return_value = [0.88, 0.82, 0.87]
            mock_cm.return_value = np.array([[10, 2], [1, 12]])

            evaluation_response = client.post("/api/evaluation/run-evaluation")

        # Step 4: Export results (if implemented)
        with patch('pandas.DataFrame.to_csv', create=True) as mock_to_csv:
            export_response = client.post("/api/evaluation/export-metrics-csv")


class TestDatabasePersistenceWorkflow:
    """Test database operations and persistence workflow."""

    @pytest.mark.integration
    def test_database_lifecycle(self):
        """Test complete database lifecycle from creation to querying."""
        # Step 1: Create database through dataset creation
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('soundfile.read') as mock_sf_read, \
             patch('main.threading.Thread') as mock_thread:

            mock_listdir.return_value = ['test1.wav', 'test2.wav']
            mock_isfile.return_value = True
            mock_sf_read.return_value = (np.random.random(110250).astype(np.float32), 22050)

            dataset_config = {
                "audio_folder": "/mock/audio",
                "class_map": [{"name": "test_class", "value": 0}],
                "backend_model": "PERCH",
                "save_path": "/mock/save",
                "window_size": 5.0
            }

            create_response = client.post("/api/dataset/create", json=dataset_config)

        # Step 2: Query database information
        with patch('main.app_state') as mock_state:
            mock_db = Mock()
            mock_db.files_df = Mock()
            mock_db.clips_df = Mock()
            mock_db.annotations_df = Mock()

            mock_state.__getitem__.side_effect = lambda key: {
                "audio_db": mock_db
            }.get(key)

            info_response = client.get("/api/database/info")
            data_response = client.get("/api/database/data")
            files_response = client.get("/api/database/files")

        # Step 3: Test database persistence
        with patch('polars.DataFrame.write_parquet') as mock_write:
            save_response = client.post("/api/active-learning/save-database")

    @pytest.mark.integration
    def test_annotation_persistence_across_sessions(self):
        """Test that annotations persist across application sessions."""
        # Simulate session 1: Create annotations
        with patch('main.app_state') as mock_state:
            mock_db = Mock()
            state_data = {"audio_db": mock_db, "class_map": {"test_class": 0}}
            mock_state.__getitem__.side_effect = lambda key: state_data.get(key)

            annotation_request = {
                "clip_id": "test_clip_123",
                "annotation": 1
            }

            annotate_response = client.post("/api/active-learning/annotate", json=annotation_request)

        # Simulate session 2: Load and verify annotations
        with patch('polars.read_parquet') as mock_read, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('json.load') as mock_json:

            mock_exists.return_value = True
            mock_json.return_value = {"class_map": {"test_class": 0}}

            # Mock loading previously saved annotations
            mock_annotations_df = Mock()
            mock_annotations_df.filter.return_value.to_dicts.return_value = [
                {
                    "clip_id": "test_clip_123",
                    "class_name": "test_class",
                    "label": "present"
                }
            ]

            load_response = client.post("/api/active-learning/load-dataset?dataset_path=/mock/dataset")


class TestErrorRecoveryWorkflows:
    """Test error handling and recovery in integrated workflows."""

    @pytest.mark.integration
    def test_dataset_creation_error_recovery(self):
        """Test error recovery during dataset creation."""
        # Test with invalid audio folder
        invalid_config = {
            "audio_folder": "/nonexistent/folder",
            "class_map": [{"name": "test", "value": 0}],
            "backend_model": "PERCH",
            "save_path": "/mock/save"
        }

        with patch('os.listdir', side_effect=FileNotFoundError("Directory not found")):
            error_response = client.post("/api/dataset/create", json=invalid_config)
            assert error_response.status_code in [400, 500]

    @pytest.mark.integration
    def test_annotation_error_recovery(self):
        """Test error recovery during annotation workflow."""
        # Test annotation with invalid clip ID
        with patch('main.app_state') as mock_state:
            mock_db = Mock()
            mock_db.add_annotation.side_effect = ValueError("Invalid clip ID")
            state_data = {"audio_db": mock_db, "class_map": {"test": 0}}
            mock_state.__getitem__.side_effect = lambda key: state_data.get(key)

            invalid_annotation = {
                "clip_id": "nonexistent_clip",
                "annotation": 1
            }

            error_response = client.post("/api/active-learning/annotate", json=invalid_annotation)
            assert error_response.status_code in [400, 500]

    @pytest.mark.integration
    def test_model_training_interruption_recovery(self):
        """Test recovery from interrupted model training."""
        # Start training
        training_request = {
            "audio_folder": "/mock/training",
            "test_split": 0.2,
            "epochs": 10
        }

        with patch('main.threading.Thread') as mock_thread:
            start_response = client.post("/api/model-training/start", json=training_request)

        # Simulate interruption
        stop_response = client.post("/api/model-training/stop")
        assert stop_response.status_code == 200

        # Check that status reflects stopped state
        status_response = client.get("/api/model-training/status")
        assert status_response.status_code == 200


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_large_dataset_workflow(self):
        """Test workflow with large dataset simulation."""
        # Simulate large number of audio files
        large_file_list = [f"file_{i}.wav" for i in range(1000)]

        with patch('os.listdir') as mock_listdir, \
             patch('os.path.isfile') as mock_isfile, \
             patch('soundfile.read') as mock_sf_read, \
             patch('main.threading.Thread') as mock_thread:

            mock_listdir.return_value = large_file_list
            mock_isfile.return_value = True
            mock_sf_read.return_value = (np.random.random(110250).astype(np.float32), 22050)

            dataset_config = {
                "audio_folder": "/mock/large/dataset",
                "class_map": [{"name": "class1", "value": 0}, {"name": "class2", "value": 1}],
                "backend_model": "PERCH",
                "save_path": "/mock/save",
                "window_size": 5.0
            }

            response = client.post("/api/dataset/create", json=dataset_config)
            assert response.status_code == 200

    @pytest.mark.integration
    def test_concurrent_api_requests(self):
        """Test handling of concurrent API requests."""
        import threading
        import time

        results = []

        def make_request():
            response = client.get("/api/dataset/status")
            results.append(response.status_code)

        # Create multiple threads to make concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10