#!/usr/bin/env python3
"""
Swift Model Server - Main Application
A FastAPI server that provides a unified interface to multiple language models using Swift.
"""

import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Union, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/server.log")
    ]
)
logger = logging.getLogger("swift-model-server")

# Import local modules
from model_manager import ModelManager
from proxy import proxy_request
from utils import load_config

# Define API models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False

class ModelAddRequest(BaseModel):
    model_id: str
    port: Optional[int] = None
    gpu_id: Optional[int] = 0
    quantize: Optional[bool] = False
    quant_method: Optional[str] = "awq"
    quant_bits: Optional[int] = 4
    max_model_len: Optional[int] = 8192
    memory_util: Optional[float] = 0.9

# Initialize FastAPI app
app = FastAPI(
    title="Swift Model Server",
    description="A unified API server for multiple language models using Swift",
    version="1.0.0"
)

# Load server configuration
server_config = load_config("config/server_config.json")

# Enable CORS if configured
if server_config.get("enable_cors", True):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.get("allowed_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize model manager
model_manager = ModelManager(server_config)

# Load all configured models on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Swift Model Server")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/quantized", exist_ok=True)
    os.makedirs("config/model_configs", exist_ok=True)
    
    # Load and start all configured models
    model_config_dir = Path("config/model_configs")
    if model_config_dir.exists():
        for config_file in model_config_dir.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    model_config = json.load(f)
                
                logger.info(f"Loading model from config: {config_file}")
                await model_manager.add_model_from_config(model_config)
            except Exception as e:
                logger.error(f"Failed to load model from {config_file}: {str(e)}")

# Shutdown event to clean up resources
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Swift Model Server")
    await model_manager.shutdown_all()

# API endpoints
@app.get("/")
async def root():
    """Server health check endpoint"""
    return {
        "status": "ok",
        "server": "Swift Model Server",
        "version": "1.0.0",
        "models_loaded": len(model_manager.models)
    }

@app.get("/v1/models")
async def list_models():
    """List all available models"""
    return {"models": model_manager.list_models()}

@app.post("/v1/models/add")
async def add_model(model_req: ModelAddRequest, background_tasks: BackgroundTasks):
    """Add a new model to the server"""
    try:
        model_info = await model_manager.add_model(
            model_id=model_req.model_id,
            port=model_req.port,
            gpu_id=model_req.gpu_id,
            quantize=model_req.quantize,
            quant_method=model_req.quant_method,
            quant_bits=model_req.quant_bits,
            max_model_len=model_req.max_model_len,
            memory_util=model_req.memory_util,
            background_tasks=background_tasks
        )
        return {
            "status": "started",
            "message": f"Model {model_req.model_id} is being loaded",
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Error adding model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add model: {str(e)}")

@app.delete("/v1/models/{model_id}")
async def remove_model(model_id: str):
    """Remove a model from the server"""
    try:
        result = await model_manager.remove_model(model_id)
        return {
            "status": "success",
            "message": f"Model {model_id} removed",
            "details": result
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.error(f"Error removing model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to remove model: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completions by proxying to the appropriate model server"""
    try:
        body = await request.json()
        model_id = body.get("model")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        
        # Find the model by ID or name
        model_info = model_manager.get_model(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Check if model is ready
        if model_info.status != "ready":
            raise HTTPException(
                status_code=503, 
                detail=f"Model {model_id} is not ready (status: {model_info.status})"
            )
        
        # Forward the request to the model's server
        return await proxy_request(request, model_info.url, "chat/completions")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/v1/completions")
async def completions(request: Request):
    """Handle text completions by proxying to the appropriate model server"""
    try:
        body = await request.json()
        model_id = body.get("model")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="Model ID is required")
        
        # Find the model by ID or name
        model_info = model_manager.get_model(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        # Check if model is ready
        if model_info.status != "ready":
            raise HTTPException(
                status_code=503, 
                detail=f"Model {model_id} is not ready (status: {model_info.status})"
            )
        
        # Forward the request to the model's server
        return await proxy_request(request, model_info.url, "completions")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in completions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/v1/models/{model_id}/status")
async def model_status(model_id: str):
    """Get the status of a specific model"""
    model_info = model_manager.get_model(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "model_id": model_info.id,
        "status": model_info.status,
        "details": {
            "name": model_info.name,
            "url": model_info.url,
            "gpu_id": model_info.gpu_id,
            "quantization": model_info.quantization,
            "uptime": model_info.get_uptime()
        }
    }

def main():
    """Main entry point for running the server"""
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8888)
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()