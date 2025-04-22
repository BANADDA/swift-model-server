# proxy.py - Request proxying utilities

import json
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("swift-model-server")

async def proxy_request(request: Request, target_url: str, endpoint: str) -> Any:
    """
    Proxy a request to a model server
    
    Args:
        request: The original FastAPI request
        target_url: The base URL of the target model server
        endpoint: The API endpoint to forward to
    
    Returns:
        The response from the model server
    """
    # Get request body
    body = await request.json()
    
    # Get request headers (include relevant ones)
    headers = {}
    for header_name in ["Content-Type", "Authorization"]:
        if header_name in request.headers:
            headers[header_name] = request.headers[header_name]
    
    # Check if this is a streaming request
    is_streaming = body.get("stream", False)
    
    # Get full target URL
    full_url = f"{target_url}/v1/{endpoint}"
    
    logger.debug(f"Proxying request to {full_url}")
    logger.debug(f"Request body: {json.dumps(body)}")
    
    try:
        # Handle streaming and non-streaming differently
        if is_streaming:
            return await proxy_streaming_request(full_url, body, headers)
        else:
            return await proxy_standard_request(full_url, body, headers)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error from model server: {e.response.status_code} - {e.response.text}")
        return JSONResponse(
            status_code=e.response.status_code,
            content=e.response.json() if e.response.headers.get("content-type") == "application/json" else {"error": e.response.text}
        )
    except httpx.RequestError as e:
        logger.error(f"Request error to model server: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"error": f"Model server unavailable: {str(e)}"}
        )
    except Exception as e:
        logger.error(f"Error proxying request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

async def proxy_standard_request(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> JSONResponse:
    """Handle a standard (non-streaming) request"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=body,
            headers=headers,
            timeout=90.0  # Long timeout for large responses
        )
        response.raise_for_status()
        
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code
        )

async def proxy_streaming_request(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> StreamingResponse:
    """Handle a streaming request"""
    async def stream_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, json=body, headers=headers, timeout=600.0) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    yield chunk
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream"
    )

# utils.py - Utility functions

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("swift-model-server")

def load_config(config_path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a JSON configuration file
    
    Args:
        config_path: Path to the configuration file
        default: Default configuration to use if the file doesn't exist
    
    Returns:
        The loaded configuration
    """
    if default is None:
        default = {}
    
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found, using defaults")
        return default
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        return default

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save a configuration to a JSON file
    
    Args:
        config: The configuration to save
        config_path: Path to save the configuration to
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with the override taking precedence
    
    Args:
        base: Base configuration
        override: Configuration to override with
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge nested dictionaries
            result[key] = merge_configs(result[key], value)
        else:
            # Override or add the value
            result[key] = value
    
    return result

def get_gpu_info() -> Dict[str, Any]:
    """
    Get information about available GPUs
    
    Returns:
        Dictionary with GPU information
    """
    try:
        import re
        import subprocess

        # Run nvidia-smi to get GPU information
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Error running nvidia-smi: {result.stderr}")
            return {"available": False, "error": result.stderr}
        
        # Parse the output
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split(", ")
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total": parts[2],
                    "memory_used": parts[3],
                    "memory_free": parts[4]
                })
        
        return {
            "available": True,
            "count": len(gpus),
            "gpus": gpus
        }
    
    except Exception as e:
        logger.error(f"Error getting GPU information: {str(e)}")
        return {"available": False, "error": str(e)}