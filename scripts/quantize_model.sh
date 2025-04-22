#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
MODEL=""
GPU_ID=0
QUANT_METHOD="awq"
BITS=4
OUTPUT_DIR=""
LOG_FILE="logs/quantize.log"

# Print header
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${GREEN}      Swift Model Quantization Script      ${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

# Print usage
print_usage() {
    echo -e "Usage: $0 [options]"
    echo -e "Options:"
    echo -e "  --model MODEL      Model ID or path (required)"
    echo -e "  --method METHOD    Quantization method (awq, gptq, bnb) [default: awq]"
    echo -e "  --bits BITS        Quantization bits (4 or 8) [default: 4]"
    echo -e "  --gpu GPU_ID       GPU ID to use [default: 0]"
    echo -e "  --output DIR       Output directory [default: models/quantized/<model-name>-<method>]"
    echo -e "  --log FILE         Log file [default: logs/quantize.log]"
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
            --method)
                QUANT_METHOD="$2"
                shift 2
                ;;
            --bits)
                BITS="$2"
                shift 2
                ;;
            --gpu)
                GPU_ID="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --log)
                LOG_FILE="$2"
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

# Validate arguments
validate_args() {
    if [[ -z "$MODEL" ]]; then
        echo -e "${RED}Error: Model ID or path is required${NC}"
        print_usage
        exit 1
    fi
    
    if [[ "$QUANT_METHOD" != "awq" && "$QUANT_METHOD" != "gptq" && "$QUANT_METHOD" != "bnb" ]]; then
        echo -e "${RED}Error: Invalid quantization method: $QUANT_METHOD${NC}"
        echo -e "Valid methods: awq, gptq, bnb"
        exit 1
    fi
    
    if [[ "$BITS" != "4" && "$BITS" != "8" ]]; then
        echo -e "${RED}Error: Invalid bits value: $BITS${NC}"
        echo -e "Valid values: 4, 8"
        exit 1
    fi
    
    # If quantization method is bnb and bits is 4, warn the user
    if [[ "$QUANT_METHOD" == "bnb" && "$BITS" == "4" ]]; then
        echo -e "${YELLOW}Warning: BitsAndBytes (bnb) works best with 8-bit quantization.${NC}"
        echo -e "${YELLOW}Consider using --bits 8 for better results.${NC}"
    fi
    
    # If output directory is not specified, create a default one
    if [[ -z "$OUTPUT_DIR" ]]; then
        MODEL_NAME=$(basename "$MODEL")
        OUTPUT_DIR="models/quantized/${MODEL_NAME}-${QUANT_METHOD}-${BITS}b"
    fi
}

# Create calibration data for quantization
create_calibration_data() {
    CALIB_DIR="data/calibration"
    
    if [[ ! -f "${CALIB_DIR}/samples.jsonl" ]]; then
        echo -e "${YELLOW}Creating calibration data...${NC}"
        mkdir -p "$CALIB_DIR"
        
        cat > "${CALIB_DIR}/samples.jsonl" << EOF
{"text": "This is a sample text for calibration. It helps the quantization process maintain accuracy."}
{"text": "Machine learning models can be compressed through quantization while preserving most of their capabilities."}
{"text": "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the English alphabet."}
{"text": "Artificial intelligence has made remarkable progress in natural language processing and understanding."}
{"text": "Quantization reduces model size by representing weights with fewer bits, trading some precision for efficiency."}
{"text": "Large language models can generate coherent and contextually relevant text based on input prompts."}
{"text": "The process of fine-tuning adapts pre-trained models to specific tasks with minimal additional training."}
{"text": "Data scientists and machine learning engineers work together to develop and deploy AI systems."}
EOF
        
        echo -e "${GREEN}Created calibration data at ${CALIB_DIR}/samples.jsonl${NC}"
    else
        echo -e "${BLUE}Using existing calibration data at ${CALIB_DIR}/samples.jsonl${NC}"
    fi
}

# Run quantization
run_quantization() {
    echo -e "${YELLOW}Quantizing model $MODEL with $QUANT_METHOD (${BITS}-bit)...${NC}"
    
    # Create output directory if it doesn't exist
    mkdir -p "$(dirname "$OUTPUT_DIR")"
    
    # Create directory for logs
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Set environment variables for GPU
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    echo -e "${BLUE}Starting quantization...${NC}"
    echo -e "${BLUE}This may take a while depending on the model size.${NC}"
    
    # Run Swift export command
    swift export \
        --model "$MODEL" \
        --quant_bits $BITS \
        --quant_method $QUANT_METHOD \
        --dataset data/calibration \
        --output_dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"
    
    # Check if quantization succeeded
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Quantization failed. See log for details: $LOG_FILE${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Quantization complete!${NC}"
    echo -e "${GREEN}Quantized model saved to: $OUTPUT_DIR${NC}"
    echo -e "${BLUE}Log file: $LOG_FILE${NC}"
}

# Main function
main() {
    print_header
    parse_args "$@"
    validate_args
    create_calibration_data
    run_quantization
    
    echo -e "${GREEN}You can now use the quantized model with:${NC}"
    echo -e "${BLUE}swift deploy --model $OUTPUT_DIR --infer_backend vllm --port 8000${NC}"
}

# Execute main function with all arguments
main "$@"