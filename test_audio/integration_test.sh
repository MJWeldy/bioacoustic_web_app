#!/bin/bash

# Comprehensive Integration Test Suite for Bioacoustics Web App
# Tests the complete workflow from dataset creation to model training

echo "=========================================="
echo "Bioacoustics Web App Integration Test"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_AUDIO_DIR="/mnt/storage_10TB/bioacoustic_web_app-main/test_audio/audio"
TEST_OUTPUT_DIR="/mnt/storage_10TB/bioacoustic_web_app-main/test_audio/outputs"
BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Create output directory
mkdir -p "$TEST_OUTPUT_DIR"

log_test() {
    local test_name="$1"
    local status="$2"
    local message="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if [ "$status" = "PASS" ]; then
        echo -e "[${GREEN}PASS${NC}] $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        [ -n "$message" ] && echo "      $message"
    elif [ "$status" = "FAIL" ]; then
        echo -e "[${RED}FAIL${NC}] $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        [ -n "$message" ] && echo "      Error: $message"
    elif [ "$status" = "INFO" ]; then
        echo -e "[${BLUE}INFO${NC}] $test_name"
        [ -n "$message" ] && echo "      $message"
    fi
}

# Test if services are running
test_services_running() {
    log_test "Testing service availability" "INFO"
    
    # Test backend
    if curl -s "$BACKEND_URL/docs" > /dev/null; then
        log_test "Backend API accessible" "PASS" "$BACKEND_URL/docs"
    else
        log_test "Backend API accessible" "FAIL" "Cannot reach $BACKEND_URL"
        return 1
    fi
    
    # Test frontend
    if curl -s "$FRONTEND_URL" > /dev/null; then
        log_test "Frontend accessible" "PASS" "$FRONTEND_URL"
    else
        log_test "Frontend accessible" "FAIL" "Cannot reach $FRONTEND_URL"
        return 1
    fi
    
    return 0
}

# Test dataset creation
test_dataset_creation() {
    log_test "Testing Dataset Builder workflow" "INFO"
    
    # Create test class map
    local class_map='{"bird_song": 0, "background": 1}'
    local payload=$(cat <<EOF
{
    "audio_folder": "$TEST_AUDIO_DIR",
    "class_map": $class_map,
    "backend_model": "BirdNET_2.4",
    "save_path": "$TEST_OUTPUT_DIR/test_dataset",
    "pretrained_classifier_path": null
}
EOF
)
    
    # Send dataset creation request
    local response=$(curl -s -X POST "$BACKEND_URL/api/dataset/create" \
        -H "Content-Type: application/json" \
        -d "$payload")
    
    if echo "$response" | grep -q '"status":"success"'; then
        log_test "Dataset creation API call" "PASS" "Dataset creation initiated"
    else
        log_test "Dataset creation API call" "FAIL" "API response: $response"
        return 1
    fi
    
    # Wait for dataset creation to complete (with timeout)
    local timeout=300  # 5 minutes
    local elapsed=0
    local interval=10
    
    while [ $elapsed -lt $timeout ]; do
        local status_response=$(curl -s "$BACKEND_URL/api/dataset/status")
        
        if echo "$status_response" | grep -q '"processing":false'; then
            if echo "$status_response" | grep -q '"database_loaded":true'; then
                log_test "Dataset creation completion" "PASS" "Database loaded successfully"
                break
            else
                log_test "Dataset creation completion" "FAIL" "Database not loaded"
                return 1
            fi
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        echo "      Waiting for dataset creation... (${elapsed}s/${timeout}s)"
    done
    
    if [ $elapsed -ge $timeout ]; then
        log_test "Dataset creation timeout" "FAIL" "Timed out after ${timeout}s"
        return 1
    fi
    
    # Verify output files exist
    if [ -f "$TEST_OUTPUT_DIR/test_dataset.parquet" ]; then
        log_test "Database file created" "PASS" "test_dataset.parquet exists"
    else
        log_test "Database file created" "FAIL" "test_dataset.parquet not found"
        return 1
    fi
    
    return 0
}

# Test active learning workflow
test_active_learning() {
    log_test "Testing Active Learning workflow" "INFO"
    
    # Load dataset
    local load_payload='{"dataset_path": "'$TEST_OUTPUT_DIR'/test_dataset.parquet"}'
    local load_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/load-dataset" \
        -H "Content-Type: application/json" \
        -d "$load_payload")
    
    if echo "$load_response" | grep -q '"status":"success"'; then
        log_test "Dataset loading" "PASS" "Dataset loaded for active learning"
    else
        log_test "Dataset loading" "FAIL" "Load response: $load_response"
        return 1
    fi
    
    # Get clips for annotation
    local clips_payload='{"score_min": 0.0, "score_max": 1.0, "limit": 10}'
    local clips_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/get-clips" \
        -H "Content-Type: application/json" \
        -d "$clips_payload")
    
    if echo "$clips_response" | grep -q '"clips":\['; then
        log_test "Clip retrieval" "PASS" "Retrieved clips for annotation"
    else
        log_test "Clip retrieval" "FAIL" "Clips response: $clips_response"
        return 1
    fi
    
    # Test annotation (annotate first clip)
    local clip_id=$(echo "$clips_response" | grep -o '"clip_id":[0-9]*' | head -1 | cut -d: -f2)
    if [ -n "$clip_id" ]; then
        local annotate_payload='{"clip_id": '$clip_id', "class_name": "bird_song", "annotation": 1}'
        local annotate_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/annotate" \
            -H "Content-Type: application/json" \
            -d "$annotate_payload")
        
        if echo "$annotate_response" | grep -q '"status":"success"'; then
            log_test "Clip annotation" "PASS" "Successfully annotated clip $clip_id"
        else
            log_test "Clip annotation" "FAIL" "Annotation response: $annotate_response"
            return 1
        fi
    else
        log_test "Clip annotation" "FAIL" "No clip_id found for annotation"
        return 1
    fi
    
    # Test spectrogram generation
    local spec_payload='{"audio_path": "'$TEST_AUDIO_DIR'/2024-07-10_07_31.wav", "start_time": 0, "duration": 5, "colormap": "viridis"}'
    local spec_response=$(curl -s -X POST "$BACKEND_URL/api/spectrogram" \
        -H "Content-Type: application/json" \
        -d "$spec_payload")
    
    if echo "$spec_response" | grep -q 'data:image/png;base64'; then
        log_test "Spectrogram generation" "PASS" "Generated base64 spectrogram"
    else
        log_test "Spectrogram generation" "FAIL" "Spectrogram response: $spec_response"
        return 1
    fi
    
    # Test database saving
    local save_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/save-database" \
        -H "Content-Type: application/json" \
        -d '{}')
    
    if echo "$save_response" | grep -q '"status":"success"'; then
        log_test "Database saving" "PASS" "Database saved successfully"
    else
        log_test "Database saving" "FAIL" "Save response: $save_response"
        return 1
    fi
    
    return 0
}

# Test model training workflow
test_model_training() {
    log_test "Testing Model Training workflow" "INFO"
    
    # Create some annotated clips for training
    mkdir -p "$TEST_OUTPUT_DIR/training_clips"
    
    # Export clips first
    local export_payload='{"output_folder": "'$TEST_OUTPUT_DIR'/training_clips", "export_uncertain": true}'
    local export_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/export-clips" \
        -H "Content-Type: application/json" \
        -d "$export_payload")
    
    if echo "$export_response" | grep -q '"status":"success"'; then
        log_test "Clip export for training" "PASS" "Exported annotated clips"
    else
        log_test "Clip export for training" "FAIL" "Export response: $export_response"
        return 1
    fi
    
    # Load training data
    local load_training_payload='{"folder_path": "'$TEST_OUTPUT_DIR'/training_clips"}'
    local load_training_response=$(curl -s -X POST "$BACKEND_URL/api/training/load-data" \
        -H "Content-Type: application/json" \
        -d "$load_training_payload")
    
    if echo "$load_training_response" | grep -q '"status":"success"'; then
        log_test "Training data loading" "PASS" "Loaded training data"
    else
        log_test "Training data loading" "FAIL" "Load training response: $load_training_response"
        return 1
    fi
    
    # Create data partitions
    local partition_payload='{"test_size": 0.2, "random_state": 42}'
    local partition_response=$(curl -s -X POST "$BACKEND_URL/api/training/create-partitions" \
        -H "Content-Type: application/json" \
        -d "$partition_payload")
    
    if echo "$partition_response" | grep -q '"status":"success"'; then
        log_test "Data partitioning" "PASS" "Created train/test splits"
    else
        log_test "Data partitioning" "FAIL" "Partition response: $partition_response"
        return 1
    fi
    
    return 0
}

# Test audio streaming
test_audio_streaming() {
    log_test "Testing audio streaming" "INFO"
    
    local audio_path="$TEST_AUDIO_DIR/2024-07-10_07_31.wav"
    local encoded_path=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$audio_path', safe=''))")
    
    local audio_response=$(curl -s -I "$BACKEND_URL/api/audio/$encoded_path")
    
    if echo "$audio_response" | grep -q "200 OK"; then
        log_test "Audio streaming endpoint" "PASS" "Audio file accessible via API"
    else
        log_test "Audio streaming endpoint" "FAIL" "Cannot stream audio file"
        return 1
    fi
    
    return 0
}

# Run performance tests
test_performance() {
    log_test "Testing performance characteristics" "INFO"
    
    # Test concurrent API calls
    local start_time=$(date +%s)
    
    # Make 5 concurrent spectrogram requests
    for i in {1..5}; do
        (curl -s -X POST "$BACKEND_URL/api/spectrogram" \
            -H "Content-Type: application/json" \
            -d '{"audio_path": "'$TEST_AUDIO_DIR'/2024-07-10_07_31.wav", "start_time": '$i', "duration": 5, "colormap": "viridis"}' \
            > "$TEST_OUTPUT_DIR/spec_$i.json") &
    done
    
    wait  # Wait for all background jobs to complete
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $duration -lt 30 ]; then
        log_test "Concurrent spectrogram generation" "PASS" "5 spectrograms generated in ${duration}s"
    else
        log_test "Concurrent spectrogram generation" "FAIL" "Took ${duration}s (expected <30s)"
        return 1
    fi
    
    return 0
}

# Main test execution
main() {
    echo "Test configuration:"
    echo "  Audio files: $(ls -1 $TEST_AUDIO_DIR/*.wav | wc -l) files"
    echo "  Output directory: $TEST_OUTPUT_DIR"
    echo "  Backend URL: $BACKEND_URL"
    echo "  Frontend URL: $FRONTEND_URL"
    echo ""
    
    # Run test suite
    test_services_running || exit 1
    test_dataset_creation || exit 1
    test_active_learning || exit 1
    test_model_training || exit 1
    test_audio_streaming || exit 1
    test_performance || exit 1
    
    # Final results
    echo ""
    echo "=========================================="
    echo "Test Results Summary"
    echo "=========================================="
    echo "Total tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All tests passed! The web application is working correctly.${NC}"
        exit 0
    else
        echo -e "\n${RED}❌ Some tests failed. Check the output above for details.${NC}"
        exit 1
    fi
}

# Check if services are running before starting tests
if ! curl -s "$BACKEND_URL/docs" > /dev/null; then
    echo -e "${RED}Error: Backend service not running at $BACKEND_URL${NC}"
    echo "Please start the application with: ./run_dev.sh"
    exit 1
fi

if ! curl -s "$FRONTEND_URL" > /dev/null; then
    echo -e "${YELLOW}Warning: Frontend service not running at $FRONTEND_URL${NC}"
    echo "Frontend tests will be skipped, but backend tests will continue."
fi

# Run tests
main