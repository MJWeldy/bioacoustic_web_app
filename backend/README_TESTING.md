# Backend Testing Framework

This document describes how to use the comprehensive testing framework for the bioacoustic web application backend.

## Overview

The testing framework provides comprehensive coverage of all backend components using **minimal dependencies** and extensive mocking to ensure tests run quickly and reliably without requiring external resources.

## Test Structure

```
backend/
├── tests/
│   ├── __init__.py              # Test package initialization
│   ├── conftest.py              # Shared fixtures and utilities
│   ├── test_database.py         # Database module tests (Audio_DB)
│   ├── test_utilities.py        # Audio processing and utilities tests
│   ├── test_api_endpoints.py    # FastAPI endpoint contract tests
│   └── test_integration.py      # End-to-end workflow tests
├── pytest.ini                  # Pytest configuration
├── run_tests.py                 # Convenient test runner script
└── README_TESTING.md            # This documentation
```

## Dependencies

The testing framework uses **only existing dependencies** from `environment.yml`:

- ✅ `pytest` - Test framework (already included)
- ✅ `fastapi` - TestClient for API testing (built-in)
- ✅ `numpy`, `polars` - Data manipulation (existing)
- ✅ Python standard library - `unittest.mock`, `tempfile`, `time`

**No additional packages required!**

## Running Tests

### Quick Start

```bash
# Activate the environment
conda activate bioacoustics-web-app

# Navigate to backend directory
cd backend

# Run all tests
python -m pytest tests/ -v

# Or use the convenient runner
python run_tests.py
```

### Test Categories

```bash
# Run only unit tests (fast, isolated)
python run_tests.py --unit

# Run only integration tests (slower, end-to-end)
python run_tests.py --integration

# Skip slow tests (for quick feedback)
python run_tests.py --fast

# Run specific module tests
python run_tests.py --module database
python run_tests.py --module utilities
python run_tests.py --module api
python run_tests.py --module integration
```

### Output Options

```bash
# Verbose output with test details
python run_tests.py --verbose

# Quiet output (only failures)
python run_tests.py --quiet

# Show timing information
python run_tests.py --timing

# Run failed tests first
python run_tests.py --failed-first
```

### Advanced Usage

```bash
# Run specific test file
python -m pytest tests/test_database.py -v

# Run specific test class
python -m pytest tests/test_database.py::TestAudioDBInitialization -v

# Run specific test function
python -m pytest tests/test_database.py::TestAudioDBInitialization::test_default_initialization -v

# Run tests matching pattern
python -m pytest tests/ -k "test_audio" -v

# Show test collection without running
python -m pytest tests/ --collect-only
```

## Test Categories and Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests (isolated, fast)
- `@pytest.mark.integration` - Integration tests (workflows)
- `@pytest.mark.slow` - Slow tests (skip with `--fast`)

## What Gets Tested

### Database Module (`test_database.py`)
- ✅ Audio_DB initialization and configuration
- ✅ File and clip creation workflows
- ✅ Annotation management and queries
- ✅ Three-table architecture validation
- ✅ Legacy compatibility layer
- ✅ Database persistence operations
- ✅ Error handling and edge cases

### Utilities Module (`test_utilities.py`)
- ✅ Audio loading and preprocessing
- ✅ Audio normalization algorithms
- ✅ Backend model configuration
- ✅ Embedding generation workflows
- ✅ TensorFlow integration
- ✅ Performance characteristics
- ✅ Error handling for invalid inputs

### API Endpoints (`test_api_endpoints.py`)
- ✅ All 31 FastAPI endpoints
- ✅ Request/response validation
- ✅ Error handling and status codes
- ✅ CORS configuration
- ✅ Pydantic model validation
- ✅ Authentication/authorization

### Integration Tests (`test_integration.py`)
- ✅ Complete dataset creation workflow
- ✅ Full active learning annotation workflow
- ✅ Model training and evaluation pipeline
- ✅ Database operations and persistence
- ✅ Multi-component interactions
- ✅ Error recovery scenarios

## Mocking Strategy

All tests use comprehensive mocking to avoid external dependencies:

### File Operations
```python
@patch('soundfile.read')
@patch('librosa.resample')
@patch('polars.DataFrame.write_parquet')
def test_audio_processing(mock_write, mock_resample, mock_read):
    # Test logic without real file I/O
```

### TensorFlow Operations
```python
@patch('tensorflow.keras.models.load_model')
@patch('tensorflow_hub.load')
def test_model_loading(mock_hub, mock_keras):
    # Test without loading actual models
```

### FastAPI Testing
```python
from fastapi.testclient import TestClient
client = TestClient(app)

def test_api_endpoint():
    response = client.post("/api/endpoint", json=data)
    assert response.status_code == 200
```

## Test Data Generation

Tests use synthetic data generation for consistency:

```python
# Synthetic audio data
def generate_test_audio():
    t = np.linspace(0, 5.0, 110250)  # 5 seconds at 22050 Hz
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)

# Mock embeddings
def generate_test_embeddings():
    return np.random.normal(0, 0.1, (10, 1280)).astype(np.float32)
```

## Performance

- **Fast execution**: Full test suite runs in <2 minutes
- **Parallel execution**: Tests can run concurrently
- **No external dependencies**: No network, files, or models required
- **Deterministic**: Consistent results across runs

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're in the backend directory
cd backend

# Check Python path
python -c "import modules.database; print('✓ Modules working')"
```

**Missing pytest:**
```bash
# Check if pytest is installed
python -c "import pytest; print('✓ Pytest available')"

# If missing, install via conda (should already be in environment.yml)
conda install pytest
```

**Mock not working:**
```bash
# Verify mock imports
python -c "from unittest.mock import Mock; print('✓ Mock available')"
```

### Test Failures

**File not found errors:**
- Check that mocks are properly applied
- Verify test fixtures are used correctly

**Import errors:**
- Ensure backend directory is in Python path
- Check that modules can be imported independently

**Assertion errors:**
- Review test expectations vs. actual implementation
- Check if mocked return values match expected format

## Adding New Tests

### Test File Template

```python
"""Test description."""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import module under test
from modules.your_module import YourClass


class TestYourClass:
    """Test your class functionality."""

    @pytest.mark.unit
    def test_your_function(self):
        """Test description."""
        # Arrange
        mock_data = Mock()

        # Act
        with patch('modules.your_module.external_dependency') as mock_dep:
            result = YourClass().your_function(mock_data)

        # Assert
        assert result is not None
        mock_dep.assert_called_once()
```

### Best Practices

1. **Use descriptive test names**: `test_database_creation_with_multiple_classes`
2. **Mock all external dependencies**: File I/O, network, models
3. **Test both success and failure cases**: Happy path + error conditions
4. **Use fixtures for common setup**: Shared test data and mocks
5. **Add appropriate markers**: `@pytest.mark.unit` or `@pytest.mark.integration`
6. **Keep tests focused**: One concept per test function
7. **Use assertions effectively**: Clear, specific assertions

## Continuous Integration

The testing framework is designed to work well in CI environments:

```bash
# CI test command
python -m pytest tests/ --tb=short -q

# With timing information
python -m pytest tests/ --durations=10

# Fail fast (stop on first failure)
python -m pytest tests/ -x
```

## Future Enhancements

Potential additions (all using existing dependencies):

- **Test coverage reporting**: Using `coverage` package if added
- **Performance benchmarking**: Extended timing analysis
- **Property-based testing**: Using `hypothesis` if added
- **Mutation testing**: Code quality assessment

The current framework provides a solid foundation that can be extended as needed while maintaining the minimal dependency approach.