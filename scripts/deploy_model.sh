#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
MODEL=""
PORT=8000
GPU_ID=0
QUANTIZE="none"
MAX_MODEL_LEN=8192
MEMORY_UTIL=0.9
LOG_FILE="logs/model.log"
CONFIG_FILE=""

# Print header
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${GREEN}      Swift Model Deployment Script       ${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# Print usage
print_usage() {
    echo -e "Usage: $0 [options]"
    echo -e "Options:"
    echo -e "  --model MODEL      Model ID or path (required)"
    echo -e "  --port PORT        Port number [default: 8000]"
    echo -e "  --gpu GPU_ID       GPU ID to use [default: 0]"
    echo -e "  --quantize METHOD  Quantization method (none, awq, gptq, bnb) [default: none]"
    echo -e "  --bits BITS        Quantization bits (4 or 8) [default: 4]"
    echo -e "  --max-len LEN      Maximum model length [default: 8192]"
    echo -e "  --memory UTIL      GPU memory utilization (0.0-1.0) [default: 0.9]"
    echo -e "  --log FILE         Log file [default: logs/model.log]"
    echo -e "  --config FILE      Config file to use instead of command line options"
    echo -e "  --help             Show this help message and exit"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                MODEL="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --gpu)
                GPU_ID="$2"
                shift 2
                ;;
            --quantize)
                QUANTIZE="$2"
                shift 2
                ;;
            --bits)
                BITS="$2"
                shift 2
                ;;
            --max-len)
                MAX_MODEL_LEN="$2"
                shift 2
                ;;
            --memory)
                MEMORY_UTIL="$2"
                shift 2
                ;;
            --log)
                LOG_FILE="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
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

# Load configuration from file
load_config() {
    if [[ -n "$CONFIG_FILE" ]]; then
        if [[ ! -f "$CONFIG_FILE" ]]; then
            echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
            exit 1
        fi
        
        echo -e "${BLUE}Loading configuration from $CONFIG_FILE${NC}"
        
        # Parse JSON config file
        MODEL=$(jq -r '.model_id // ""' "$CONFIG_FILE")
        PORT=$(jq -r '.port // 8000' "$CONFIG_FILE")
        GPU_ID=$(jq -r '.gpu_id // 0' "$CONFIG_FILE")
        
        QUANT_ENABLED=$(jq -r '.quantization.enabled // false' "$CONFIG_FILE")
        if [[ "$QUANT_ENABLED" == "true" ]]; then
            QUANTIZE=$(jq -r '.quantization.method // "none"' "$CONFIG_FILE")
            BITS=$(jq -r '.quantization.bits // 4' "$CONFIG_FILE")
        else
            QUANTIZE="none"
        fi
        
        MAX_MODEL_LEN=$(jq -r '.parameters.max_model_len // 8192' "$CONFIG_FILE")
        MEMORY_UTIL=$(jq -r '.parameters.gpu_memory_utilization // 0.9' "$CONFIG_FILE")
        
        # Create log file name based on model name
        MODEL_NAME=$(basename "$MODEL")
        LOG_FILE="logs/${MODEL_NAME}.log"
    fi
}

# Validate arguments
validate_args() {
    if [[ -z "$MODEL" ]]; then
        echo -e "${RED}Error: Model ID or path is required${NC}"
        print_usage
        exit 1
    fi
    
    if [[ "$QUANTIZE" != "none" && "$QUANTIZE" != "awq" && "$QUANTIZE" != "gptq" && "$QUANTIZE" != "bnb" ]]; then
        echo -e "${RED}Error: Invalid quantization method: $QUANTIZE${NC}"
        echo -e "Valid methods: none, awq, gptq, bnb"
        exit 1
    fi
    
    if [[ "$QUANTIZE" != "none" && -z "$BITS" ]]; then
        # Default to 4 bits for AWQ/GPTQ and 8 bits for BNB
        if [[ "$QUANTIZE" == "bnb" ]]; then
            BITS=8
        else
            BITS=4
        fi
    fi
}

# Apply quantization to the model
apply_quantization() {
    if [[ "$QUANTIZE" == "none" ]]; then
        return
    fi
    
    echo -e "${YELLOW}Applying $QUANTIZE quantization (${BITS}-bit) to model $MODEL...${NC}"
    
    # Extract model name for output directory
    MODEL_NAME=$(basename "$MODEL")
    OUTPUT_DIR="models/quantized/${MODEL_NAME}-${QUANTIZE}-${BITS}b"
    
    # Check if already quantized
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo -e "${YELLOW}Model already quantized at $OUTPUT_DIR. Skipping quantization.${NC}"
        QUANTIZED_MODEL="$OUTPUT_DIR"
        return
    fi
    
    # Create calibration data if needed
    if [[ ! -f "data/calibration/samples.jsonl" ]]; then
        echo -e "${YELLOW}Creating calibration data...${NC}"
        mkdir -p data/calibration
        echo '{"text": "This is a sample text for calibration."}' > data/calibration/samples.jsonl
    fi
    
    # Run quantization
    echo -e "${BLUE}Running quantization with Swift...${NC}"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    swift export \
        --model "$MODEL" \
        --quant_bits $BITS \
        --quant_method $QUANTIZE \
        --dataset data/calibration \
        --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
    
    # Check if quantization succeeded
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Quantization failed. See log for details: $LOG_FILE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Quantization complete. Model saved to $OUTPUT_DIR${NC}"
    QUANTIZED_MODEL="$OUTPUT_DIR"
}

# Deploy the model as API server
deploy_model() {
    # Determine which model to deploy (original or quantized)
    if [[ "$QUANTIZE" != "none" && -n "$QUANTIZED_MODEL" ]]; then
        DEPLOY_MODEL="$QUANTIZED_MODEL"
    else
        DEPLOY_MODEL="$MODEL"
    fi
    
    echo -e "${YELLOW}Deploying model $DEPLOY_MODEL on port $PORT...${NC}"
    
    # Choose inference backend
    BACKEND="vllm"
    
    # Create directory for logs
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Deploy the model
    echo -e "${BLUE}Starting Swift deploy...${NC}"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    nohup swift deploy \
        --model "$DEPLOY_MODEL" \
        --infer_backend $BACKEND \
        --port $PORT \
        --host 0.0.0.0 \
        --max_model_len $MAX_MODEL_LEN \
        --gpu_memory_utilization $MEMORY_UTIL \
        --max_batch_size 32 > "$LOG_FILE" 2>&1 &
    
    # Store PID for monitoring
    PID=$!
    echo $PID > "logs/$(basename "$DEPLOY_MODEL")-$PORT.pid"
    
    echo -e "${GREEN}Model deployed with PID $PID${NC}"
    echo -e "Logs are being written to $LOG_FILE"
    echo -e "API server is running at ${BLUE}http://localhost:$PORT${NC}"
    
    # Wait a moment and check if process is still running
    sleep 5
    if kill -0 $PID 2>/dev/null; then
        echo -e "${GREEN}Server started successfully!${NC}"
    else
        echo -e "${RED}Server failed to start. Check the logs: $LOG_FILE${NC}"
    fi
}

# Register model with main API server
register_model() {
    if [[ "$QUANTIZE" != "none" && -n "$QUANTIZED_MODEL" ]]; then
        MODEL_PATH="$QUANTIZED_MODEL"
    else
        MODEL_PATH="$MODEL"
    fi
    
    # Extract model name
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    # Create model config file
    CONFIG_DIR="config/model_configs"
    mkdir -p "$CONFIG_DIR"
    
    CONFIG_JSON="$CONFIG_DIR/${MODEL_NAME}.json"
    
    echo -e "${BLUE}Creating model configuration: $CONFIG_JSON${NC}"
    
    # Create JSON configuration
    cat > "$CONFIG_JSON" << EOF
{
    "model_id": "$MODEL_PATH",
    "display_name": "$MODEL_NAME",
    "port": $PORT,
    "gpu_id": $GPU_ID,
    "quantization": {
        "enabled": $([ "$QUANTIZE" != "none" ] && echo "true" || echo "false"),
        "method": "$QUANTIZE",
        "bits": $BITS
    },
    "parameters": {
        "max_model_len": $MAX_MODEL_LEN,
        "gpu_memory_utilization": $MEMORY_UTIL
    },
    "pid": $PID
}
EOF

    echo -e "${GREEN}Model registered in configuration.${NC}"
}

# Main function
main() {
    print_header
    parse_args "$@"
    load_config
    validate_args
    
    # Set default log file if not specified
    if [[ "$LOG_FILE" == "logs/model.log" ]]; then
        MODEL_NAME=$(basename "$MODEL")
        LOG_FILE="logs/${MODEL_NAME}.log"
    fi
    
    # Apply quantization if requested
    if [[ "$QUANTIZE" != "none" ]]; then
        apply_quantization
    fi
    
    # Deploy the model
    deploy_model
    
    # Register the model
    register_model
    
    echo -e "${GREEN}Deployment complete!${NC}"
    echo -e "The model is now serving requests at ${BLUE}http://localhost:$PORT${NC}"
}

# Execute main function with all arguments
main "$@"