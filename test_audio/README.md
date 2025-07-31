# Test Suite for Bioacoustics Web Application

This directory contains a comprehensive test suite designed to validate the functionality, performance, and reliability of the bioacoustics web application.

## Test Audio Files

The test suite uses two sample audio files from real bioacoustic recordings:

### Audio Files Specifications
- **2024-07-10_07_31.wav**: 61.22 seconds, 32kHz, mono, 3.92MB
- **2024-07-10_07_45.wav**: 56.44 seconds, 32kHz, mono, 3.61MB

These files represent typical morning recordings that may contain bird vocalizations mixed with background sounds, making them ideal for testing classification and annotation workflows.

## Test Components

### 1. Automated Integration Tests (`integration_test.sh`)

**Purpose**: Comprehensive automated testing of the complete application workflow

**Features**:
- Tests all major API endpoints
- Validates dataset creation workflow
- Tests active learning annotation process
- Validates model training pipeline
- Tests audio streaming functionality
- Performance benchmarking
- Automatic pass/fail reporting

**Usage**:
```bash
# Ensure application is running first
./run_dev.sh

# Run integration tests
cd test_audio
chmod +x integration_test.sh
./integration_test.sh
```

**Expected Duration**: 15-30 minutes (depending on backend model loading)

### 2. Manual Test Checklist (`manual_test_checklist.md`)

**Purpose**: Detailed UI/UX validation checklist for human testers

**Coverage**:
- All 5 application tabs (Dataset Builder, Active Learning, Model Training, Evaluation, Database Viewer)
- User interface interactions
- Error handling and edge cases
- Cross-browser compatibility
- Accessibility features
- Performance validation

**Usage**: Open the checklist and systematically verify each item while using the web application.

### 3. Stress Testing (`stress_test.sh`)

**Purpose**: High-load testing to validate application stability and performance under stress

**Tests**:
- Concurrent spectrogram generation
- Rapid annotation operations
- Memory usage monitoring
- API response times under load
- Concurrent database operations

**Usage**:
```bash
# Ensure application is running
./run_dev.sh

# Run stress tests
cd test_audio
chmod +x stress_test.sh
./stress_test.sh
```

**Expected Duration**: 10-15 minutes

### 4. Test Configuration (`test_config.json`)

**Purpose**: Centralized configuration for all test scenarios

**Contains**:
- Audio file specifications
- Multiple class map examples (binary, multiclass, detailed)
- Backend model configurations
- Test scenario definitions
- Performance benchmarks
- Expected output specifications

## Test Scenarios

### Quick Validation Test
- **Duration**: 5 minutes
- **Purpose**: Fast verification of core functionality
- **Class Map**: Simple binary (bird_song, background)
- **Backend Model**: BirdNET_2.4

### Full Workflow Test
- **Duration**: 30 minutes
- **Purpose**: Complete end-to-end testing
- **Class Map**: Multiclass (NOWA_song, CEWA, WIWA, background)
- **Includes**: Dataset creation → Active learning → Model training → Evaluation

### Stress Test
- **Duration**: 10 minutes
- **Purpose**: High-load performance validation
- **Concurrent Requests**: 10
- **Class Map**: Detailed 8-class classification

## Running the Complete Test Suite

### Prerequisites
```bash
# 1. Ensure application is installed and configured
./setup.sh

# 2. Start the application
./run_dev.sh

# 3. Verify services are running
./health_check.sh
```

### Automated Testing
```bash
cd test_audio

# Make scripts executable
chmod +x *.sh

# Run integration tests
./integration_test.sh

# Run stress tests  
./stress_test.sh
```

### Manual Testing
1. Open `manual_test_checklist.md`
2. Navigate to http://localhost:3000
3. Systematically test each checklist item
4. Document any issues found

## Expected Outputs

### Successful Test Run Should Produce:
```
test_audio/outputs/
├── test_dataset.parquet           # Created dataset
├── test_dataset_embeddings.npy    # Generated embeddings
├── test_dataset_metadata.json     # Dataset metadata
├── training_clips/                # Exported annotated clips
│   ├── metadata.json              # Export metadata
│   └── *.wav                      # Individual audio clips
├── trained_model.keras            # Trained classifier model
├── training_history.json          # Training metrics
└── evaluation_metrics.json        # Model performance metrics
```

### Test Reports:
- Integration test results (console output)
- Stress test report (`stress_test_report.md`)
- Resource usage logs (`resource_usage.log`)
- API response time analysis (`response_times.txt`)

## Performance Benchmarks

### Expected Performance Metrics:
- **Dataset Creation**: < 5 minutes for 2 audio files
- **Spectrogram Generation**: < 3 seconds per clip
- **Audio Streaming Latency**: < 500ms
- **Annotation Response Time**: < 200ms
- **Concurrent Requests**: Handle 5+ simultaneous spectrogram requests
- **Memory Usage**: < 4GB total during normal operation

## Extending the Test Suite

### Adding More Audio Files
1. Place additional WAV files in `test_audio/audio/`
2. Update `test_config.json` with file specifications
3. Modify test scripts to include new files

### Creating Custom Test Scenarios
1. Edit `test_config.json` to add new scenarios
2. Define custom class maps for your use case
3. Adjust performance benchmarks as needed

### Adding New Test Categories
1. Create new test scripts following existing patterns
2. Use the logging framework for consistent output
3. Update this README with new test descriptions

## Troubleshooting Test Failures

### Common Issues:

#### Backend Not Starting
```bash
# Check backend status
curl http://localhost:8000/docs

# If not running, check logs
tail -f logs/backend.log

# Restart if needed
./reset.sh
./run_dev.sh
```

#### Test Timeouts
- Increase timeout values in test scripts
- Check system resources (CPU, memory)
- Ensure no other heavy processes are running

#### Audio File Issues
- Verify audio files are not corrupted: `file test_audio/audio/*.wav`
- Check file permissions: `ls -la test_audio/audio/`
- Test audio playback: `aplay test_audio/audio/2024-07-10_07_31.wav`

#### Memory Issues
- Monitor system memory: `free -h`
- Check for memory leaks in backend
- Restart application between test runs

### Test Output Analysis

#### Integration Test Success Indicators:
- All API endpoints return 200 status codes
- Database files are created successfully
- Spectrograms generate valid base64 images
- Audio streaming returns proper content types
- No error messages in test output

#### Stress Test Success Indicators:
- >80% success rate for concurrent operations
- Memory usage increases <500MB during tests
- API response times <2 seconds average
- No backend crashes or timeouts

## Contributing to the Test Suite

### When adding new features to the application:
1. Add corresponding tests to `integration_test.sh`
2. Update the manual checklist with new UI elements
3. Add stress tests for any resource-intensive operations
4. Update `test_config.json` with new configuration options
5. Document expected behaviors and performance metrics

### Test Maintenance:
- Review and update benchmarks quarterly
- Add new test scenarios for reported bugs
- Validate tests after major application updates
- Keep test audio files representative of real-world usage

---

## Quick Start Guide

**For new users wanting to validate their installation:**

```bash
# 1. Start the application
./run_dev.sh

# 2. Run quick integration test
cd test_audio
chmod +x integration_test.sh
./integration_test.sh

# 3. If tests pass, your installation is working correctly!
```

**For developers wanting comprehensive validation:**

```bash
# 1. Run all automated tests
cd test_audio
./integration_test.sh && ./stress_test.sh

# 2. Perform manual UI testing using the checklist
# 3. Review all generated reports and logs
# 4. Fix any issues found and re-test
```

This test suite ensures the bioacoustics web application functions correctly across all use cases and performance scenarios. Regular testing helps maintain application quality and reliability.

## To Make the Test Suite Even More Comprehensive:

  1. More Diverse Audio Files:
    - Different durations (short clips <10s, long recordings >2min)
    - Various sample rates (22kHz, 44.1kHz, 48kHz)
    - Multi-channel recordings (stereo)
    - Different file formats (MP3, FLAC)
    - Files with known species for validation
  2. Annotated Reference Data:
    - Pre-annotated clips with ground truth labels
    - Confidence scores for validation
    - Metadata files with species information
  3. Edge Case Audio Files:
    - Very quiet recordings (low amplitude)
    - Noisy recordings (high background noise)
    - Clipped/distorted audio
    - Silent recordings
    - Very short clips (<1 second)
  4. Model Files:
    - Pre-trained classifier models for testing
    - Different model architectures
    - Models with known performance metrics
  5. Large Dataset Simulation:
    - Scripts to generate many test files
    - Synthetic audio with known characteristics
    - Large-scale database files for performance testing

