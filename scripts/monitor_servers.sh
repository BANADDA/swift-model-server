#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Default values
CONFIG_DIR="config/model_configs"
WATCH_MODE=false
REFRESH_RATE=5

# Print header
print_header() {
    clear
    echo -e "${BLUE}============================================${NC}"
    echo -e "${GREEN}      Swift Model Server Monitor      ${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# Print usage
print_usage() {
    echo -e "Usage: $0 [options]"
    echo -e "Options:"
    echo -e "  --watch           Watch mode with automatic refresh"
    echo -e "  --refresh SECS    Refresh rate in seconds (default: 5)"
    echo -e "  --help            Show this help message and exit"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --watch)
                WATCH_MODE=true
                shift
                ;;
            --refresh)
                REFRESH_RATE="$2"
                shift 2
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done
}

# Get model status by checking the process and API endpoint
check_model_status() {
    local model_config="$1"
    local model_id=$(jq -r '.model_id' "$model_config")
    local port=$(jq -r '.port' "$model_config")
    local pid=$(jq -r '.pid // 0' "$model_config")
    
    local status="unknown"
    local details=""
    
    # Check if process is running
    if [[ "$pid" != "null" && "$pid" != "0" ]]; then
        if kill -0 "$pid" 2>/dev/null; then
            # Process is running, now check API
            if curl -s "http://localhost:${port}/v1/models" &>/dev/null; then
                status="running"
            else
                status="starting"
                details="Process running but API not responding"
            fi
        else
            status="stopped"
            details="Process not running"
        fi
    else
        status="not started"
        details="No PID information"
    fi
    
    echo "$status|$details"
}

# Get GPU utilization for a specific GPU
get_gpu_util() {
    local gpu_id="$1"
    
    if ! command -v nvidia-smi &>/dev/null; then
        echo "N/A (nvidia-smi not found)"
        return
    fi
    
    local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)
    local gpu_mem=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null)
    
    if [[ -n "$gpu_util" && -n "$gpu_mem" ]]; then
        echo "${gpu_util}% GPU, ${gpu_mem}% MEM"
    else
        echo "N/A"
    fi
}

# Display model information
display_model_info() {
    if [[ ! -d "$CONFIG_DIR" ]]; then
        echo -e "${YELLOW}No model configurations found in $CONFIG_DIR${NC}"
        return
    fi
    
    echo -e "${BOLD}ACTIVE MODELS:${NC}"
    echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"
    printf "${BOLD}%-30s %-12s %-8s %-8s %-15s %-25s${NC}\n" "MODEL" "STATUS" "PORT" "GPU" "UTILIZATION" "DETAILS"
    echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"
    
    local found_models=false
    
    for config_file in "$CONFIG_DIR"/*.json; do
        # Skip if no files match the pattern
        [[ -e "$config_file" ]] || continue
        
        found_models=true
        
        local model_name=$(jq -r '.display_name // basename(.model_id)' "$config_file")
        local port=$(jq -r '.port // "N/A"' "$config_file")
        local gpu_id=$(jq -r '.gpu_id // "N/A"' "$config_file")
        local quant_method=$(jq -r '.quantization.method // "none"' "$config_file")
        local quant_bits=$(jq -r '.quantization.bits // 0' "$config_file")
        
        # Get status and details
        local status_info=$(check_model_status "$config_file")
        local status=$(echo "$status_info" | cut -d'|' -f1)
        local details=$(echo "$status_info" | cut -d'|' -f2)
        
        # Get GPU utilization
        local gpu_util=$(get_gpu_util "$gpu_id")
        
        # Colorize status
        local status_colored
        case "$status" in
            running)
                status_colored="${GREEN}running${NC}"
                ;;
            starting)
                status_colored="${YELLOW}starting${NC}"
                ;;
            stopped)
                status_colored="${RED}stopped${NC}"
                ;;
            *)
                status_colored="${CYAN}$status${NC}"
                ;;
        esac
        
        # Add quantization info to details if present
        if [[ "$quant_method" != "none" && "$quant_method" != "null" ]]; then
            if [[ -n "$details" ]]; then
                details="$details, "
            fi
            details="${details}${quant_method}-${quant_bits}bit"
        fi
        
        printf "%-30s %-12b %-8s %-8s %-15s %-25s\n" \
               "$model_name" "$status_colored" "$port" "$gpu_id" "$gpu_util" "$details"
    done
    
    if [[ "$found_models" == false ]]; then
        echo -e "${YELLOW}No model configurations found in $CONFIG_DIR${NC}"
    fi
    
    echo ""
}

# Display server information
display_server_info() {
    # Check if server is running by checking for the process
    local server_pid=$(pgrep -f "uvicorn.*app:app" | head -n 1)
    local server_status
    local server_uptime=""
    
    if [[ -n "$server_pid" ]]; then
        server_status="${GREEN}running${NC}"
        
        # Get process start time
        if command -v ps &>/dev/null; then
            local start_time=$(ps -o lstart= -p "$server_pid")
            if [[ -n "$start_time" ]]; then
                server_uptime="since $start_time"
            fi
        fi
    else
        server_status="${RED}stopped${NC}"
    fi
    
    echo -e "${BOLD}SERVER STATUS:${NC}"
    echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"
    printf "${BOLD}%-15s %-12s %-60s${NC}\n" "COMPONENT" "STATUS" "DETAILS"
    echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"
    
    printf "%-15s %-12b %-60s\n" "Main Server" "$server_status" "$server_uptime"
    
    # Check if API is responding
    local api_port=8888
    if curl -s "http://localhost:${api_port}/" &>/dev/null; then
        local api_info=$(curl -s "http://localhost:${api_port}/")
        local model_count=$(echo "$api_info" | jq -r '.models_loaded // "unknown"')
        printf "%-15s %-12b %-60s\n" "API Endpoint" "${GREEN}available${NC}" "Models loaded: $model_count"
    else
        printf "%-15s %-12b %-60s\n" "API Endpoint" "${RED}unavailable${NC}" "API not responding"
    fi
    
    echo ""
}

# Display system information
display_system_info() {
    echo -e "${BOLD}SYSTEM INFORMATION:${NC}"
    echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"
    
    # Get RAM information
    if command -v free &>/dev/null; then
        local mem_info=$(free -h | grep "Mem:")
        local mem_total=$(echo "$mem_info" | awk '{print $2}')
        local mem_used=$(echo "$mem_info" | awk '{print $3}')
        local mem_usage_pct=$(free | grep Mem | awk '{print $3/$2 * 100}' | cut -d. -f1)
        
        printf "%-15s %-60s\n" "Memory" "${mem_used} / ${mem_total} (${mem_usage_pct}%)"
    else
        printf "%-15s %-60s\n" "Memory" "Information not available"
    fi
    
    # Get disk information
    if command -v df &>/dev/null; then
        local disk_info=$(df -h . | tail -n 1)
        local disk_total=$(echo "$disk_info" | awk '{print $2}')
        local disk_used=$(echo "$disk_info" | awk '{print $3}')
        local disk_usage_pct=$(echo "$disk_info" | awk '{print $5}')
        
        printf "%-15s %-60s\n" "Disk" "${disk_used} / ${disk_total} (${disk_usage_pct})"
    else
        printf "%-15s %-60s\n" "Disk" "Information not available"
    fi
    
    # Get GPU information
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        echo -e "${BOLD}GPU INFORMATION:${NC}"
        echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"
        printf "${BOLD}%-5s %-20s %-15s %-15s %-15s${NC}\n" "ID" "NAME" "MEMORY USED" "UTILIZATION" "TEMPERATURE"
        echo -e "${BOLD}--------------------------------------------------------------------------------------------------------${NC}"
        
        # Get list of GPUs
        local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
        
        for ((i=0; i<gpu_count; i++)); do
            local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$i")
            local gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$i")
            local gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i "$i")
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$i")
            local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -i "$i")
            
            printf "%-5s %-20s %-15s %-15s %-15s\n" \
                   "$i" "$gpu_name" "${gpu_mem_used} / ${gpu_mem_total}" "${gpu_util}" "${gpu_temp}°C"
        done
    else
        printf "%-15s %-60s\n" "GPU" "Information not available (nvidia-smi not found)"
    fi
    
    echo ""
}

# Main display function
display_info() {
    print_header
    display_server_info
    display_model_info
    display_system_info
    
    if [[ "$WATCH_MODE" == true ]]; then
        echo -e "${BLUE}Auto-refresh every ${REFRESH_RATE} seconds. Press Ctrl+C to exit.${NC}"
    fi
}

# Main function
main() {
    parse_args "$@"
    
    if [[ "$WATCH_MODE" == true ]]; then
        while true; do
            display_info
            sleep "$REFRESH_RATE"
        done
    else
        display_info
    fi
}

# Execute main function with all arguments
main "$@"