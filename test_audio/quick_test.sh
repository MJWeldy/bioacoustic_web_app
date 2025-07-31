#!/bin/bash

# Quick Test Script for Bioacoustics Web App
# Fast validation of core functionality (5-minute test)

echo "=========================================="
echo "Bioacoustics Web App Quick Test"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
TEST_AUDIO_DIR="/mnt/storage_10TB/bioacoustic_web_app-main/test_audio/audio"
TEST_OUTPUT_DIR="/mnt/storage_10TB/bioacoustic_web_app-main/test_audio/quick_test_outputs"
BACKEND_URL="http://localhost:8000"
FRONTEND_URL="http://localhost:3000"

# Create output directory
mkdir -p "$TEST_OUTPUT_DIR"

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0

log_test() {
    local test_name="$1"
    local status="$2" 
    local message="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    printf "%-50s" "$test_name"
    
    if [ "$status" = "PASS" ]; then
        echo -e "[${GREEN}PASS${NC}]"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        [ -n "$message" ] && echo "    $message"
    elif [ "$status" = "FAIL" ]; then
        echo -e "[${RED}FAIL${NC}]"
        [ -n "$message" ] && echo "    Error: $message"
    elif [ "$status" = "INFO" ]; then
        echo -e "[${BLUE}INFO${NC}] $test_name"
        [ -n "$message" ] && echo "    $message"
    fi
}

# Quick service check
log_test "Service Check" "INFO"

if curl -s "$BACKEND_URL/docs" > /dev/null; then
    log_test "Backend API accessible" "PASS"
else
    log_test "Backend API accessible" "FAIL"
    echo -e "${RED}Backend not running. Start with: ./run_dev.sh${NC}"
    exit 1
fi

if curl -s "$FRONTEND_URL" > /dev/null; then
    log_test "Frontend accessible" "PASS"
else
    log_test "Frontend accessible" "FAIL"
fi

# Quick dataset creation test
log_test "Quick dataset creation" "INFO"

class_map='{"bird": 0, "background": 1}'
payload=$(cat <<EOF
{
    "audio_folder": "$TEST_AUDIO_DIR",
    "class_map": $class_map,
    "backend_model": "BirdNET_2.4",
    "save_path": "$TEST_OUTPUT_DIR/quick_dataset",
    "pretrained_classifier_path": null
}
EOF
)

response=$(curl -s -X POST "$BACKEND_URL/api/dataset/create" \
    -H "Content-Type: application/json" \
    -d "$payload")

if echo "$response" | grep -q '"status":"success"'; then
    log_test "Dataset creation initiated" "PASS"
    
    # Wait for completion (max 3 minutes)
    for i in {1..18}; do
        status_response=$(curl -s "$BACKEND_URL/api/dataset/status")
        if echo "$status_response" | grep -q '"processing":false'; then
            if echo "$status_response" | grep -q '"database_loaded":true'; then
                log_test "Dataset creation completed" "PASS"
                break
            else
                log_test "Dataset creation completed" "FAIL"
                break
            fi
        fi
        sleep 10
        echo -n "."
    done
    echo ""
else
    log_test "Dataset creation initiated" "FAIL"
fi

# Quick spectrogram test
spec_payload='{"audio_path": "'$TEST_AUDIO_DIR'/2024-07-10_07_31.wav", "start_time": 0, "duration": 5, "colormap": "viridis"}'
spec_response=$(curl -s -X POST "$BACKEND_URL/api/spectrogram" \
    -H "Content-Type: application/json" \
    -d "$spec_payload")

if echo "$spec_response" | grep -q 'data:image/png;base64'; then
    log_test "Spectrogram generation" "PASS"
else
    log_test "Spectrogram generation" "FAIL"
fi

# Quick audio streaming test
audio_path="$TEST_AUDIO_DIR/2024-07-10_07_31.wav"
encoded_path=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$audio_path', safe=''))")
if curl -s -I "$BACKEND_URL/api/audio/$encoded_path" | grep -q "200 OK"; then
    log_test "Audio streaming" "PASS"
else
    log_test "Audio streaming" "FAIL"
fi

# Quick active learning test
if [ -f "$TEST_OUTPUT_DIR/quick_dataset.parquet" ]; then
    load_payload='{"dataset_path": "'$TEST_OUTPUT_DIR'/quick_dataset.parquet"}'
    load_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/load-dataset" \
        -H "Content-Type: application/json" \
        -d "$load_payload")
    
    if echo "$load_response" | grep -q '"status":"success"'; then
        log_test "Active learning dataset load" "PASS"
        
        # Quick annotation test
        clips_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/get-clips" \
            -H "Content-Type: application/json" \
            -d '{"score_min": 0.0, "score_max": 1.0, "limit": 5}')
        
        if echo "$clips_response" | grep -q '"clips":\['; then
            log_test "Clip retrieval" "PASS"
        else
            log_test "Clip retrieval" "FAIL"
        fi
    else
        log_test "Active learning dataset load" "FAIL"
    fi
else
    log_test "Active learning dataset load" "FAIL" "Dataset file not found"
fi

# Results summary
echo ""
echo "=========================================="
echo "Quick Test Results"
echo "=========================================="
echo "Tests completed: $TOTAL_TESTS"
echo -e "Tests passed: ${GREEN}$PASSED_TESTS${NC}"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo -e "\n${GREEN}🎉 Quick test PASSED! Core functionality is working.${NC}"
    echo ""
    echo "Next steps:"
    echo "  - Run full integration test: ./integration_test.sh"
    echo "  - Perform manual UI testing with checklist"
    echo "  - Run stress tests: ./stress_test.sh"
    exit 0
elif [ $PASSED_TESTS -gt $((TOTAL_TESTS / 2)) ]; then
    echo -e "\n${YELLOW}⚠️  Quick test PARTIAL. Some issues found.${NC}"
    echo "Run './health_check.sh' to diagnose issues."
    exit 1
else
    echo -e "\n${RED}❌ Quick test FAILED. Major issues detected.${NC}"
    echo "Check logs and run './health_check.sh' for diagnosis."
    exit 1
fi