# Swift Model Server

A comprehensive system for deploying and serving multiple large language models with quantization options using the Swift framework.

## Features

- **Multi-Model Support**: Deploy and serve multiple models simultaneously
- **Quantization**: Apply AWQ, GPTQ, or BNB quantization to reduce model size and improve inference speed
- **OpenAI-Compatible API**: Use the same API format as OpenAI for easy integration
- **Model Management**: Add, remove, and monitor models through a unified interface
- **Performance Optimization**: Leverages vLLM or LMDeploy backends for high-performance inference

## Getting Started

### Prerequisites

- CUDA-compatible GPU (recommended)
- Python 3.9 or higher
- CUDA drivers and libraries (for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://BANADDA/swift-model-server.git
cd swift-model-server

# Run the installation script
./install.sh
```

### Quick Start

1. Deploy a model (e.g., Qwen 1.5)

```bash
./scripts/deploy_model.sh --model Qwen/Qwen1.5-7B-Chat --port 8000 --gpu 0
```

2. Deploy another model with quantization (e.g., DeepSeek)

```bash
./scripts/deploy_model.sh --model deepseek-ai/deepseek-coder-7b-instruct --port 8001 --gpu 0 --quantize awq
```

3. Start the API server

```bash
python server/app.py
```

4. Interact with the models

```python
import requests

# Chat with Qwen
response = requests.post(
    "http://localhost:8888/v1/chat/completions",
    json={
        "model": "Qwen 1.5 7B",
        "messages": [{"role": "user", "content": "Hello, who are you?"}],
        "temperature": 0.7,
        "max_tokens": 1024
    }
)

print(response.json())
```

## Configuration

### Server Configuration

The main server configuration is located in `config/server_config.json`:

```json
{
    "host": "0.0.0.0",
    "port": 8888,
    "log_level": "info",
    "infer_backend": "vllm",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.9
}
```

### Model Configuration

Each model has its own configuration file in `config/model_configs/`:

```json
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
```

## API Endpoints

### Server Management

- `GET /`: Server health check
- `GET /v1/models`: List all available models
- `POST /v1/models/add`: Add a new model
- `DELETE /v1/models/{model_id}`: Remove a model

### Model Inference

- `POST /v1/chat/completions`: Chat completions (ChatGPT-like interface)
- `POST /v1/completions`: Text completions (traditional interface)

## Quantization

The server supports several quantization methods:

- **AWQ (Activation-aware Weight Quantization)**: 4-bit quantization with minimal accuracy loss
- **GPTQ (Generative Pre-trained Transformer Quantization)**: 4-bit quantization with good performance
- **BNB (Bitsandbytes)**: 8-bit quantization for a good balance of speed and quality

To quantize a model:

```bash
./scripts/quantize_model.sh --model Qwen/Qwen1.5-7B-Chat --method awq --bits 4
```

## Performance Optimization

For optimal performance:

- Use vLLM as the inference backend
- Apply appropriate quantization for your hardware
- Consider using tensor parallelism on multi-GPU systems
- Adjust `max_model_len` based on your use case and available memory

## Troubleshooting

- **Out of Memory Errors**: Reduce the model size with quantization or adjust `gpu_memory_utilization`
- **Slow Inference**: Consider using vLLM backend and enabling batching
- **Failed Model Loading**: Check GPU availability and CUDA installation

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
