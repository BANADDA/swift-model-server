#!/bin/bash

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}  Swift Model Server - Installation Script  ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if running with sudo/root permissions
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}Warning: This script might need sudo privileges for some operations.${NC}"
  echo "Consider running with sudo if you encounter permission errors."
  echo ""
fi

# Create the directory structure
echo -e "${GREEN}Creating directory structure...${NC}"
mkdir -p config/model_configs config/templates
mkdir -p scripts
mkdir -p server
mkdir -p models/quantized models/cache
mkdir -p logs
mkdir -p data/calibration

# Create sample calibration data
echo -e "${GREEN}Creating sample calibration data...${NC}"
echo '{"text": "This is a sample text for calibration."}' > data/calibration/samples.jsonl
echo '{"text": "Another example sentence to help with model calibration."}' >> data/calibration/samples.jsonl
echo '{"text": "Quantization needs diverse texts to maintain quality and accuracy."}' >> data/calibration/samples.jsonl
echo '{"text": "Machine learning models benefit from representative data samples."}' >> data/calibration/samples.jsonl
echo '{"text": "Natural language processing requires good examples of human language."}' >> data/calibration/samples.jsonl

# Check for Python
echo -e "${GREEN}Checking for Python installation...${NC}"
if command -v python3 &> /dev/null; then
    echo "Python 3 is installed."
    PYTHON="python3"
elif command -v python &> /dev/null; then
    echo "Python is installed."
    PYTHON="python"
else
    echo -e "${RED}Error: Python is not installed. Please install Python 3.9 or higher.${NC}"
    exit 1
fi

# Create requirements.txt
echo -e "${GREEN}Creating requirements file...${NC}"
cat > requirements.txt << EOF
ms-swift>=1.0.0
fastapi>=0.95.0
uvicorn>=0.22.0
httpx>=0.24.0
pydantic>=2.0.0
python-dotenv>=1.0.0
loguru>=0.7.0
typer>=0.9.0
rich>=13.4.0
EOF

# Create main server configuration
echo -e "${GREEN}Creating server configuration...${NC}"
cat > config/server_config.json << EOF
{
    "host": "0.0.0.0",
    "port": 8888,
    "log_level": "info",
    "infer_backend": "vllm",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.9,
    "enable_cors": true,
    "allowed_origins": ["*"],
    "default_model_configs": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.95
    }
}
EOF

# Create example model configurations
echo -e "${GREEN}Creating example model configurations...${NC}"
cat > config/model_configs/qwen1.5.json << EOF
{
    "model_id": "Qwen/Qwen1.5-7B-Chat",
    "display_name": "Qwen 1.5 7B",
    "port": 8000,
    "gpu_id": 0,
    "quantization": {
        "enabled": false,
        "method": "awq",
        "bits": 4
    },
    "parameters": {
        "max_model_len": 8192,
        "tensor_parallel_size": 1
    }
}
EOF

cat > config/model_configs/deepseek.json << EOF
{
    "model_id": "deepseek-ai/deepseek-coder-7b-instruct",
    "display_name": "DeepSeek Coder",
    "port": 8001,
    "gpu_id": 0,
    "quantization": {
        "enabled": true,
        "method": "awq",
        "bits": 4
    },
    "parameters": {
        "max_model_len": 8192,
        "tensor_parallel_size": 1
    }
}
EOF

# Copy scripts from the artifact repository
echo -e "${GREEN}Setting up scripts...${NC}"
cp deploy_model.sh scripts/
cp quantize_model.sh scripts/
cp monitor_servers.sh scripts/
cp server/app.py server/
cp server/model_manager.py server/
cp server/proxy.py server/
cp server/utils.py server/

# Make scripts executable
chmod +x scripts/*.sh

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
$PYTHON -m pip install -r requirements.txt

# Check for CUDA
echo -e "${GREEN}Checking for CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available."
    nvidia-smi
else
    echo -e "${YELLOW}Warning: CUDA is not detected. GPU acceleration may not be available.${NC}"
fi

# Confirm installation
echo -e "${GREEN}Installation complete!${NC}"
echo -e "To start the server, run: ${BLUE}python3 server/app.py${NC}"
echo -e "To deploy a model, run: ${BLUE}scripts/deploy_model.sh${NC}"
echo ""
echo -e "Documentation is available in the README.md file."