#!/bin/bash

# Stress Test Suite for Bioacoustics Web App
# Tests application under high load and concurrent operations

echo "=========================================="
echo "Bioacoustics Web App Stress Test Suite"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_AUDIO_DIR="/mnt/storage_10TB/bioacoustic_web_app-main/test_audio/audio"
TEST_OUTPUT_DIR="/mnt/storage_10TB/bioacoustic_web_app-main/test_audio/stress_outputs"
BACKEND_URL="http://localhost:8000"
CONCURRENT_REQUESTS=10
STRESS_DURATION=300  # 5 minutes

# Create output directory
mkdir -p "$TEST_OUTPUT_DIR"

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

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

# Monitor system resources during tests
monitor_resources() {
    local duration=$1
    local output_file="$TEST_OUTPUT_DIR/resource_usage.log"
    
    echo "timestamp,cpu_percent,memory_mb,disk_io" > "$output_file"
    
    for ((i=0; i<duration; i+=5)); do
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        local cpu_percent=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
        local memory_mb=$(free -m | awk 'NR==2{printf "%.1f", $3}')
        local disk_io=$(iostat -d 1 1 | tail -n +4 | awk '{print $4}' | tail -1)
        
        echo "$timestamp,$cpu_percent,$memory_mb,$disk_io" >> "$output_file"
        sleep 5
    done
}

# Test concurrent spectrogram generation
test_concurrent_spectrograms() {
    log_test "Testing concurrent spectrogram generation" "INFO"
    
    local start_time=$(date +%s)
    local success_count=0
    local error_count=0
    
    # Start resource monitoring in background
    monitor_resources 60 &
    local monitor_pid=$!
    
    # Generate spectrograms concurrently
    for i in $(seq 1 $CONCURRENT_REQUESTS); do
        {
            local audio_file=$(ls "$TEST_AUDIO_DIR"/*.wav | shuf -n 1)
            local start_offset=$((i % 30))  # Vary start times
            
            local response=$(curl -s -X POST "$BACKEND_URL/api/spectrogram" \
                -H "Content-Type: application/json" \
                -d '{
                    "audio_path": "'$audio_file'",
                    "start_time": '$start_offset',
                    "duration": 5,
                    "colormap": "viridis"
                }')
            
            if echo "$response" | grep -q 'data:image/png;base64'; then
                echo "SUCCESS:$i" >> "$TEST_OUTPUT_DIR/spectrogram_results.txt"
            else
                echo "ERROR:$i:$response" >> "$TEST_OUTPUT_DIR/spectrogram_results.txt"
            fi
        } &
    done
    
    # Wait for all background jobs
    wait
    
    # Stop resource monitoring
    kill $monitor_pid 2>/dev/null
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Count results
    if [ -f "$TEST_OUTPUT_DIR/spectrogram_results.txt" ]; then
        success_count=$(grep -c "SUCCESS" "$TEST_OUTPUT_DIR/spectrogram_results.txt")
        error_count=$(grep -c "ERROR" "$TEST_OUTPUT_DIR/spectrogram_results.txt")
    fi
    
    if [ $success_count -ge $((CONCURRENT_REQUESTS * 8 / 10)) ]; then
        log_test "Concurrent spectrogram stress test" "PASS" "$success_count/$CONCURRENT_REQUESTS succeeded in ${duration}s"
    else
        log_test "Concurrent spectrogram stress test" "FAIL" "Only $success_count/$CONCURRENT_REQUESTS succeeded"
    fi
    
    rm -f "$TEST_OUTPUT_DIR/spectrogram_results.txt"
}

# Test rapid annotation operations
test_rapid_annotations() {
    log_test "Testing rapid annotation operations" "INFO"
    
    # First create a dataset for annotations
    local class_map='{"bird_song": 0, "background": 1}'
    local dataset_payload=$(cat <<EOF
{
    "audio_folder": "$TEST_AUDIO_DIR",
    "class_map": $class_map,
    "backend_model": "BirdNET_2.4",
    "save_path": "$TEST_OUTPUT_DIR/stress_dataset",
    "pretrained_classifier_path": null
}
EOF
)
    
    # Create dataset
    curl -s -X POST "$BACKEND_URL/api/dataset/create" \
        -H "Content-Type: application/json" \
        -d "$dataset_payload" > /dev/null
    
    # Wait for dataset creation
    sleep 30
    
    # Load dataset for active learning
    local load_payload='{"dataset_path": "'$TEST_OUTPUT_DIR'/stress_dataset.parquet"}'
    curl -s -X POST "$BACKEND_URL/api/active-learning/load-dataset" \
        -H "Content-Type: application/json" \
        -d "$load_payload" > /dev/null
    
    # Get clips
    local clips_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/get-clips" \
        -H "Content-Type: application/json" \
        -d '{"score_min": 0.0, "score_max": 1.0, "limit": 50}')
    
    # Extract clip IDs
    local clip_ids=($(echo "$clips_response" | grep -o '"clip_id":[0-9]*' | cut -d: -f2))
    
    if [ ${#clip_ids[@]} -eq 0 ]; then
        log_test "Rapid annotation stress test" "FAIL" "No clips available for annotation"
        return 1
    fi
    
    local start_time=$(date +%s)
    local annotation_count=0
    local error_count=0
    
    # Perform rapid annotations
    for clip_id in "${clip_ids[@]:0:20}"; do  # Limit to first 20 clips
        {
            local class_name=$([ $((RANDOM % 2)) -eq 0 ] && echo "bird_song" || echo "background")
            local annotation=$((RANDOM % 2))
            
            local annotate_payload='{"clip_id": '$clip_id', "class_name": "'$class_name'", "annotation": '$annotation'}'
            local response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/annotate" \
                -H "Content-Type: application/json" \
                -d "$annotate_payload")
            
            if echo "$response" | grep -q '"status":"success"'; then
                echo "SUCCESS:$clip_id" >> "$TEST_OUTPUT_DIR/annotation_results.txt"
            else
                echo "ERROR:$clip_id:$response" >> "$TEST_OUTPUT_DIR/annotation_results.txt"
            fi
        } &
        
        # Limit concurrent annotations to prevent overwhelming the system
        if [ $(jobs -r | wc -l) -ge 5 ]; then
            wait
        fi
    done
    
    wait  # Wait for all remaining jobs
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Count results
    if [ -f "$TEST_OUTPUT_DIR/annotation_results.txt" ]; then
        annotation_count=$(grep -c "SUCCESS" "$TEST_OUTPUT_DIR/annotation_results.txt")
        error_count=$(grep -c "ERROR" "$TEST_OUTPUT_DIR/annotation_results.txt")
    fi
    
    if [ $annotation_count -ge 15 ]; then  # Expect at least 75% success rate
        log_test "Rapid annotation stress test" "PASS" "$annotation_count annotations completed in ${duration}s"
    else
        log_test "Rapid annotation stress test" "FAIL" "Only $annotation_count annotations succeeded"
    fi
    
    rm -f "$TEST_OUTPUT_DIR/annotation_results.txt"
}

# Test memory usage with large datasets
test_memory_usage() {
    log_test "Testing memory usage under load" "INFO"
    
    # Get initial memory usage
    local initial_memory=$(ps -o pid,vsz,rss,comm -p $(pgrep -f "python main.py") | tail -1 | awk '{print $3}')
    
    if [ -z "$initial_memory" ]; then
        log_test "Memory usage test" "FAIL" "Cannot find backend process"
        return 1
    fi
    
    # Perform memory-intensive operations
    for i in {1..5}; do
        # Generate multiple spectrograms
        curl -s -X POST "$BACKEND_URL/api/spectrogram" \
            -H "Content-Type: application/json" \
            -d '{
                "audio_path": "'$TEST_AUDIO_DIR'/2024-07-10_07_31.wav",
                "start_time": '$i',
                "duration": 10,
                "colormap": "viridis"
            }' > /dev/null &
    done
    
    wait
    sleep 5  # Allow for garbage collection
    
    # Get final memory usage
    local final_memory=$(ps -o pid,vsz,rss,comm -p $(pgrep -f "python main.py") | tail -1 | awk '{print $3}')
    
    if [ -z "$final_memory" ]; then
        log_test "Memory usage test" "FAIL" "Backend process disappeared during test"
        return 1
    fi
    
    local memory_increase=$((final_memory - initial_memory))
    local memory_increase_mb=$((memory_increase / 1024))
    
    if [ $memory_increase_mb -lt 500 ]; then  # Less than 500MB increase
        log_test "Memory usage test" "PASS" "Memory increased by ${memory_increase_mb}MB (within limits)"
    else
        log_test "Memory usage test" "FAIL" "Memory increased by ${memory_increase_mb}MB (excessive)"
    fi
}

# Test API response times under load
test_api_response_times() {
    log_test "Testing API response times under load" "INFO"
    
    local endpoint_tests=(
        "GET /docs"
        "GET /api/dataset/status"
        "POST /api/spectrogram"
    )
    
    local results_file="$TEST_OUTPUT_DIR/response_times.txt"
    echo "endpoint,response_time_ms,status" > "$results_file"
    
    for endpoint in "${endpoint_tests[@]}"; do
        local method=$(echo "$endpoint" | cut -d' ' -f1)
        local path=$(echo "$endpoint" | cut -d' ' -f2)
        local url="$BACKEND_URL$path"
        
        # Test response times with concurrent requests
        for i in {1..10}; do
            {
                local start_time=$(date +%s%3N)
                
                if [ "$method" = "GET" ]; then
                    local response=$(curl -s -w "%{http_code}" "$url")
                    local status_code="${response: -3}"
                elif [ "$method" = "POST" ] && [ "$path" = "/api/spectrogram" ]; then
                    local response=$(curl -s -w "%{http_code}" -X POST "$url" \
                        -H "Content-Type: application/json" \
                        -d '{
                            "audio_path": "'$TEST_AUDIO_DIR'/2024-07-10_07_31.wav",
                            "start_time": 0,
                            "duration": 5,
                            "colormap": "viridis"
                        }')
                    local status_code="${response: -3}"
                fi
                
                local end_time=$(date +%s%3N)
                local response_time=$((end_time - start_time))
                
                echo "$endpoint,$response_time,$status_code" >> "$results_file"
            } &
        done
        
        wait  # Wait for all concurrent requests to complete
    done
    
    # Analyze response times
    local avg_response_time=$(awk -F',' 'NR>1 {sum+=$2; count++} END {print int(sum/count)}' "$results_file")
    local max_response_time=$(awk -F',' 'NR>1 {if($2>max) max=$2} END {print max}' "$results_file")
    local error_count=$(awk -F',' 'NR>1 && $3 != "200" {count++} END {print count+0}' "$results_file")
    
    if [ $avg_response_time -lt 2000 ] && [ $max_response_time -lt 5000 ] && [ $error_count -eq 0 ]; then
        log_test "API response time test" "PASS" "Avg: ${avg_response_time}ms, Max: ${max_response_time}ms, Errors: $error_count"
    else
        log_test "API response time test" "FAIL" "Avg: ${avg_response_time}ms, Max: ${max_response_time}ms, Errors: $error_count"
    fi
}

# Test database operations under concurrent load
test_concurrent_database_operations() {
    log_test "Testing concurrent database operations" "INFO"
    
    # Create a dataset first
    local class_map='{"test_class": 0, "background": 1}'
    local dataset_payload=$(cat <<EOF
{
    "audio_folder": "$TEST_AUDIO_DIR",
    "class_map": $class_map,
    "backend_model": "BirdNET_2.4",
    "save_path": "$TEST_OUTPUT_DIR/concurrent_test_dataset",
    "pretrained_classifier_path": null
}
EOF
)
    
    curl -s -X POST "$BACKEND_URL/api/dataset/create" \
        -H "Content-Type: application/json" \
        -d "$dataset_payload" > /dev/null
    
    # Wait for dataset creation
    sleep 30
    
    local success_count=0
    local error_count=0
    
    # Perform concurrent database operations
    for i in {1..10}; do
        {
            # Load dataset
            local load_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/load-dataset" \
                -H "Content-Type: application/json" \
                -d '{"dataset_path": "'$TEST_OUTPUT_DIR'/concurrent_test_dataset.parquet"}')
            
            # Get clips
            local clips_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/get-clips" \
                -H "Content-Type: application/json" \
                -d '{"score_min": 0.0, "score_max": 1.0, "limit": 5}')
            
            # Save database
            local save_response=$(curl -s -X POST "$BACKEND_URL/api/active-learning/save-database" \
                -H "Content-Type: application/json" \
                -d '{}')
            
            if echo "$load_response$clips_response$save_response" | grep -q '"status":"success"'; then
                echo "SUCCESS:$i" >> "$TEST_OUTPUT_DIR/db_results.txt"
            else
                echo "ERROR:$i" >> "$TEST_OUTPUT_DIR/db_results.txt"
            fi
        } &
    done
    
    wait
    
    # Count results
    if [ -f "$TEST_OUTPUT_DIR/db_results.txt" ]; then
        success_count=$(grep -c "SUCCESS" "$TEST_OUTPUT_DIR/db_results.txt")
        error_count=$(grep -c "ERROR" "$TEST_OUTPUT_DIR/db_results.txt")
    fi
    
    if [ $success_count -ge 8 ]; then  # 80% success rate
        log_test "Concurrent database operations" "PASS" "$success_count/10 operations succeeded"
    else
        log_test "Concurrent database operations" "FAIL" "Only $success_count/10 operations succeeded"
    fi
    
    rm -f "$TEST_OUTPUT_DIR/db_results.txt"
}

# Generate stress test report
generate_report() {
    local report_file="$TEST_OUTPUT_DIR/stress_test_report.md"
    
    cat > "$report_file" << EOF
# Stress Test Report

**Date**: $(date)
**Duration**: Various test durations
**Concurrent Requests**: $CONCURRENT_REQUESTS
**Test Audio Files**: $(ls -1 $TEST_AUDIO_DIR/*.wav | wc -l)

## Test Results Summary

- **Total Tests**: $TOTAL_TESTS
- **Passed**: $PASSED_TESTS
- **Failed**: $FAILED_TESTS
- **Success Rate**: $((PASSED_TESTS * 100 / TOTAL_TESTS))%

## Resource Usage

EOF
    
    if [ -f "$TEST_OUTPUT_DIR/resource_usage.log" ]; then
        echo "### CPU and Memory Usage" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        tail -10 "$TEST_OUTPUT_DIR/resource_usage.log" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
    fi
    
    if [ -f "$TEST_OUTPUT_DIR/response_times.txt" ]; then
        echo "### API Response Times" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
        head -20 "$TEST_OUTPUT_DIR/response_times.txt" >> "$report_file"
        echo "\`\`\`" >> "$report_file"
    fi
    
    echo "Report generated: $report_file"
}

# Main test execution
main() {
    echo "Stress test configuration:"
    echo "  Concurrent requests: $CONCURRENT_REQUESTS"
    echo "  Test duration: Various"
    echo "  Audio files: $(ls -1 $TEST_AUDIO_DIR/*.wav | wc -l)"
    echo "  Output directory: $TEST_OUTPUT_DIR"
    echo ""
    
    # Check if backend is running
    if ! curl -s "$BACKEND_URL/docs" > /dev/null; then
        echo -e "${RED}Error: Backend service not running at $BACKEND_URL${NC}"
        echo "Please start the application with: ./run_dev.sh"
        exit 1
    fi
    
    # Run stress tests
    test_concurrent_spectrograms
    test_rapid_annotations
    test_memory_usage
    test_api_response_times
    test_concurrent_database_operations
    
    # Generate report
    generate_report
    
    # Final results
    echo ""
    echo "=========================================="
    echo "Stress Test Results Summary"
    echo "=========================================="
    echo "Total tests: $TOTAL_TESTS"
    echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All stress tests passed! The application handles load well.${NC}"
        exit 0
    else
        echo -e "\n${YELLOW}⚠️  Some stress tests failed. Check the report for details.${NC}"
        exit 1
    fi
}

# Run stress tests
main