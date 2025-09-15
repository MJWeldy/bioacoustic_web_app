# Test-Driven Development Framework Implementation Plan

## Overview
Implement a lightweight TDD framework for the bioacoustic web application backend using minimal dependencies to ensure all components function correctly while maintaining simplicity.

## Phase 1: Minimal Test Infrastructure Setup

### 1.1 Lightweight Test Environment
- **Use existing pytest**: Already included in `environment.yml` - no additional dependencies
- **Built-in FastAPI testing**: Use FastAPI's built-in TestClient (no extra packages needed)
- **Python standard library**: Leverage `unittest.mock`, `tempfile`, and `os` for mocking and test isolation
- **Simple test data**: Use small embedded test data instead of external dependencies

### 1.2 Zero-Dependency Test Utilities
- **Mock audio generation**: Use numpy to create simple sine waves for testing
- **In-memory databases**: Use temporary Polars DataFrames for testing (no external database needed)
- **File system mocking**: Use `tempfile` and `unittest.mock` for file operations
- **Environment variables**: Use `os.environ` manipulation for configuration testing

## Phase 2: Core Module Testing (Minimal Approach)

### 2.1 Database Module (`modules/database.py`) - HIGH PRIORITY
**Minimal Unit Tests**:
- Test `Audio_DB` initialization with mock data
- Test core methods with in-memory operations only
- Use `unittest.mock.patch` for file system operations
- Validate three-table structure without requiring real files

**Test Strategy**:
```python
# Example: No external dependencies
import tempfile
import unittest.mock as mock
from modules.database import Audio_DB

def test_database_creation():
    # Uses only standard library + existing packages
    db = Audio_DB(num_classes=2)
    assert len(db.files_df) == 0
    assert db.num_classes == 2
```

### 2.2 Utilities Module (`modules/utilities.py`) - HIGH PRIORITY
**Minimal Audio Testing**:
- Mock `soundfile.read()` and `librosa.resample()` calls
- Test audio processing logic without requiring real audio files
- Use numpy arrays to simulate audio data
- Test backend model configuration without loading actual models

**Test Strategy**:
```python
# Mock audio processing without real files
@mock.patch('soundfile.read')
@mock.patch('librosa.resample')
def test_audio_loading(mock_resample, mock_read):
    # Test logic without file I/O dependencies
    mock_read.return_value = (np.zeros(1000), 22050)
    # Test the actual processing logic
```

### 2.3 API Endpoints (`main.py`) - HIGH PRIORITY
**FastAPI Built-in Testing**:
- Use `FastAPI.TestClient` (already available, no extra dependencies)
- Mock all external operations (file system, model loading)
- Test request/response validation and error handling
- Focus on API contract testing rather than full integration

**Test Strategy**:
```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_dataset_status():
    # Test API without requiring real datasets
    response = client.get("/api/dataset/status")
    assert response.status_code == 200
```

## Phase 3: Simplified Testing Categories

### 3.1 Core Functionality Tests (Essential)
**Database Operations**:
- Table creation and schema validation
- Basic CRUD operations with mock data
- Legacy compatibility layer

**API Contract Tests**:
- Request/response validation
- Error handling and status codes
- Authentication and authorization

**Audio Processing Logic**:
- Normalization algorithms
- Sample rate conversion logic
- Backend model configuration

### 3.2 Integration Tests (Simplified)
**Mock-Based Integration**:
- End-to-end workflows using mocked components
- State persistence across API calls
- Error propagation through the system

**No External Dependencies**:
- All file operations mocked
- All model loading mocked
- All network calls mocked

## Phase 4: Lightweight Test Execution

### 4.1 Simple Test Runner
**Standard pytest execution**:
```bash
# No additional test infrastructure needed
cd backend
python -m pytest tests/ -v
```

**Minimal configuration** (`pytest.ini`):
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

### 4.2 Manual Performance Validation
**Simple timing tests**:
- Use `time.time()` for basic performance measurement
- Memory usage with `tracemalloc` (standard library)
- No complex performance testing frameworks

## Phase 5: Test Organization (Minimal Structure)

### 5.1 Simple Directory Structure
```
backend/
├── tests/
│   ├── __init__.py
│   ├── test_database.py          # Database module tests
│   ├── test_utilities.py         # Audio processing tests
│   ├── test_api_endpoints.py     # FastAPI endpoint tests
│   ├── test_integration.py       # Mock-based integration tests
│   └── conftest.py               # Shared fixtures
├── modules/
└── main.py
```

### 5.2 Shared Test Utilities (`conftest.py`)
**Minimal fixtures using standard library**:
```python
import pytest
import tempfile
from unittest.mock import Mock, patch

@pytest.fixture
def mock_audio_db():
    """Create a mock database for testing"""
    with patch('modules.database.Audio_DB') as mock:
        yield mock

@pytest.fixture
def temp_dir():
    """Temporary directory for file tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
```

## Phase 6: Dependencies Analysis

### 6.1 Required Dependencies (Already Available)
**From existing `environment.yml`**:
- `pytest>=7.0.0` ✅ Already included
- `numpy` ✅ Already included
- `polars` ✅ Already included
- `fastapi` ✅ Already included

**Python Standard Library** (No additional install):
- `unittest.mock` - Mocking framework
- `tempfile` - Temporary file/directory management
- `time` - Performance measurement
- `tracemalloc` - Memory profiling
- `os` - Environment variable management

### 6.2 Explicitly Avoiding These Dependencies
**Heavy testing frameworks** (NOT NEEDED):
- ❌ `pytest-asyncio` - Use basic async/await testing
- ❌ `pytest-mock` - Use standard library `unittest.mock`
- ❌ `httpx` - Use FastAPI's built-in `TestClient`
- ❌ `factory-boy` - Create simple test data manually
- ❌ `pytest-cov` - Use basic test execution without coverage
- ❌ `pytest-benchmark` - Use simple timing measurements

## Implementation Priority

### Week 1: Core Infrastructure
1. Set up basic test directory structure
2. Create minimal `conftest.py` with essential fixtures
3. Write database module tests using mocks

### Week 2: API Testing
1. Test critical API endpoints with `TestClient`
2. Mock all external dependencies
3. Validate request/response contracts

### Week 3: Integration & Documentation
1. Create mock-based integration tests
2. Document test execution procedures
3. Create simple test running scripts

## Success Metrics (Simplified)

**Functionality Coverage**:
- All critical database operations tested
- All API endpoints have basic contract tests
- Core audio processing logic validated

**Maintainability**:
- Tests run in <2 minutes total
- No external test dependencies beyond existing packages
- Clear, simple test structure

**Reliability**:
- Tests are deterministic (no flaky tests due to external dependencies)
- Easy to run locally and in CI
- Fast feedback loop for developers

## Running Tests

### Basic Execution
```bash
# Activate environment
conda activate bioacoustics-web-app

# Run all tests
cd backend
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_database.py -v

# Run with simple timing
python -m pytest tests/ -v --durations=10
```

### No Additional Setup Required
- Uses existing conda environment
- No database setup needed (all mocked)
- No external services required
- No additional configuration files

This minimal approach ensures comprehensive testing while maintaining simplicity and avoiding dependency bloat.