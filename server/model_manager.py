#!/usr/bin/env python3
"""
Swift Model Server - Model Manager
Handles the management of model servers, including starting, stopping, and monitoring.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("swift-model-server")

class ModelInfo(BaseModel):
    """Information about a deployed model"""
    id: str
    name: str
    port: int
    url: str
    gpu_id: int
    status: str = "loading"
    quantization: Dict[str, Any]
    pid: Optional[int] = None
    start_time: Optional[float] = None
    
    def get_uptime(self) -> str:
        """Get the uptime of the model server"""
        if not self.start_time:
            return "Not started"
        
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

class ModelManager:
    """Manages multiple model servers"""
    
    def __init__(self, server_config: Dict[str, Any]):
        self.server_config = server_config
        self.models: Dict[str, ModelInfo] = {}
        self.next_port = server_config.get("starting_port", 8000)
    
    def _get_next_port(self) -> int:
        """Get the next available port"""
        port = self.next_port
        self.next_port += 1
        return port
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        return [
            {
                "id": model.id,
                "name": model.name,
                "status": model.status,
                "quantization": model.quantization,
                "uptime": model.get_uptime()
            }
            for model in self.models.values()
        ]
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get a model by ID or name"""
        # Try direct ID lookup
        if model_id in self.models:
            return self.models[model_id]
        
        # Try name lookup
        for model in self.models.values():
            if model.name == model_id:
                return model
        
        return None
    
    async def add_model_from_config(self, model_config: Dict[str, Any]) -> ModelInfo:
        """Add a model from a configuration file"""
        model_id = model_config.get("model_id")
        if not model_id:
            raise ValueError("Missing model_id in configuration")
        
        # Extract model name
        model_name = os.path.basename(model_id)
        
        # Create ModelInfo
        model_info = ModelInfo(
            id=model_id,
            name=model_name,
            port=model_config.get("port", self._get_next_port()),
            url=f"http://localhost:{model_config.get('port', self._get_next_port())}",
            gpu_id=model_config.get("gpu_id", 0),
            status="loading",
            quantization={
                "enabled": model_config.get("quantization", {}).get("enabled", False),
                "method": model_config.get("quantization", {}).get("method", "none"),
                "bits": model_config.get("quantization", {}).get("bits", 4)
            }
        )
        
        # Store the model info
        self.models[model_id] = model_info
        
        # Start the model server as a background process
        await self._start_model_server(
            model_info=model_info,
            max_model_len=model_config.get("parameters", {}).get("max_model_len", 8192),
            memory_util=model_config.get("parameters", {}).get("gpu_memory_utilization", 0.9)
        )
        
        return model_info
    
    async def add_model(
        self,
        model_id: str,
        port: Optional[int] = None,
        gpu_id: int = 0,
        quantize: bool = False,
        quant_method: str = "awq",
        quant_bits: int = 4,
        max_model_len: int = 8192,
        memory_util: float = 0.9,
        background_tasks: Optional[BackgroundTasks] = None
    ) -> ModelInfo:
        """Add a new model to the server"""
        # Check if model already exists
        if model_id in self.models:
            raise ValueError(f"Model {model_id} already exists")
        
        # Assign port if not provided
        if port is None:
            port = self._get_next_port()
        
        # Extract model name
        model_name = os.path.basename(model_id)
        
        # Apply quantization if requested
        deployed_model_id = model_id
        if quantize:
            deployed_model_id = await self._quantize_model(
                model_id=model_id,
                quant_method=quant_method,
                quant_bits=quant_bits,
                gpu_id=gpu_id
            )
        
        # Create ModelInfo
        model_info = ModelInfo(
            id=model_id,
            name=model_name,
            port=port,
            url=f"http://localhost:{port}",
            gpu_id=gpu_id,
            status="loading",
            quantization={
                "enabled": quantize,
                "method": quant_method if quantize else "none",
                "bits": quant_bits if quantize else 0
            }
        )
        
        # Store the model info
        self.models[model_id] = model_info
        
        # Start the model server in background
        if background_tasks:
            background_tasks.add_task(
                self._start_model_server,
                model_info=model_info,
                deployed_model_id=deployed_model_id,
                max_model_len=max_model_len,
                memory_util=memory_util
            )
        else:
            await self._start_model_server(
                model_info=model_info,
                deployed_model_id=deployed_model_id,
                max_model_len=max_model_len,
                memory_util=memory_util
            )
        
        # Save configuration
        self._save_model_config(model_info, deployed_model_id, max_model_len, memory_util)
        
        return model_info
    
    async def _quantize_model(
        self,
        model_id: str,
        quant_method: str,
        quant_bits: int,
        gpu_id: int
    ) -> str:
        """Quantize a model using Swift"""
        logger.info(f"Quantizing model {model_id} with {quant_method} ({quant_bits}-bit)")
        
        # Extract model name for output directory
        model_name = os.path.basename(model_id)
        output_dir = f"models/quantized/{model_name}-{quant_method}-{quant_bits}b"
        
        # Check if already quantized
        if os.path.exists(output_dir):
            logger.info(f"Model already quantized at {output_dir}. Skipping quantization.")
            return output_dir
        
        # Create calibration data if needed
        calib_dir = "data/calibration"
        if not os.path.exists(f"{calib_dir}/samples.jsonl"):
            logger.info("Creating calibration data...")
            os.makedirs(calib_dir, exist_ok=True)
            with open(f"{calib_dir}/samples.jsonl", "w") as f:
                f.write('{"text": "This is a sample text for calibration."}\n')
                f.write('{"text": "Another example for model calibration."}\n')
        
        # Run quantization
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = [
            "swift", "export",
            "--model", model_id,
            "--quant_bits", str(quant_bits),
            "--quant_method", quant_method,
            "--dataset", calib_dir,
            "--output_dir", output_dir
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        process = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            logger.error(f"Quantization failed: {process.stderr}")
            raise RuntimeError(f"Failed to quantize model: {process.stderr}")
        
        logger.info(f"Quantization complete. Model saved to {output_dir}")
        return output_dir
    
    async def _start_model_server(
        self,
        model_info: ModelInfo,
        deployed_model_id: Optional[str] = None,
        max_model_len: int = 8192,
        memory_util: float = 0.9
    ) -> None:
        """Start a model server as a background process"""
        if deployed_model_id is None:
            deployed_model_id = model_info.id
            
        logger.info(f"Starting model server for {model_info.name} on port {model_info.port}")
        
        # Create log directory
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/{model_info.name.replace('/', '_')}.log"
        
        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(model_info.gpu_id)
        
        # Choose inference backend
        backend = self.server_config.get("infer_backend", "vllm")
        
        # Prepare command
        cmd = [
            "swift", "deploy",
            "--model", deployed_model_id,
            "--infer_backend", backend,
            "--port", str(model_info.port),
            "--host", "0.0.0.0",
            "--max_model_len", str(max_model_len),
            "--gpu_memory_utilization", str(memory_util),
            "--max_batch_size", "32"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            # Open log file
            with open(log_file, "w") as log:
                # Start process
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            # Update model info
            model_info.pid = process.pid
            model_info.start_time = time.time()
            
            # Wait for server to initialize
            for _ in range(30):  # Wait for up to 30 seconds
                await asyncio.sleep(1)
                
                # Check if process is still running
                if process.poll() is not None:
                    # Process terminated
                    returncode = process.poll()
                    with open(log_file, "r") as log:
                        error_log = log.read()
                    logger.error(f"Model server process exited with code {returncode}")
                    logger.error(f"Log: {error_log}")
                    model_info.status = "failed"
                    raise RuntimeError(f"Model server process exited with code {returncode}")
                
                # Try to connect to the server
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        response = await client.get(
                            f"http://localhost:{model_info.port}/v1/models",
                            timeout=2.0
                        )
                        if response.status_code == 200:
                            logger.info(f"Model server for {model_info.name} started successfully")
                            model_info.status = "ready"
                            return
                except Exception:
                    # Continue waiting
                    pass
            
            # If we got here, server didn't start in time
            logger.warning(f"Model server for {model_info.name} is taking longer than expected to start")
            model_info.status = "starting"
        
        except Exception as e:
            logger.error(f"Failed to start model server: {str(e)}")
            model_info.status = "failed"
            raise
    
    def _save_model_config(
        self,
        model_info: ModelInfo,
        deployed_model_id: str,
        max_model_len: int,
        memory_util: float
    ) -> None:
        """Save model configuration to file"""
        config_dir = "config/model_configs"
        os.makedirs(config_dir, exist_ok=True)
        
        model_name = model_info.name.replace('/', '_')
        config_file = f"{config_dir}/{model_name}.json"
        
        config = {
            "model_id": model_info.id,
            "deployed_model_id": deployed_model_id,
            "display_name": model_info.name,
            "port": model_info.port,
            "gpu_id": model_info.gpu_id,
            "quantization": model_info.quantization,
            "parameters": {
                "max_model_len": max_model_len,
                "gpu_memory_utilization": memory_util
            },
            "pid": model_info.pid,
            "created_at": datetime.now().isoformat()
        }
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved model configuration to {config_file}")
    
    async def remove_model(self, model_id: str) -> Dict[str, Any]:
        """Remove a model from the server"""
        # Find the model
        model_info = self.get_model(model_id)
        if not model_info:
            raise KeyError(f"Model {model_id} not found")
        
        result = {
            "id": model_info.id,
            "name": model_info.name,
            "port": model_info.port
        }
        
        # Stop the process if running
        if model_info.pid:
            try:
                # Try to terminate the process gracefully
                os.kill(model_info.pid, 15)  # SIGTERM
                
                # Wait for process to exit
                for _ in range(10):  # Wait for up to 10 seconds
                    await asyncio.sleep(1)
                    try:
                        # Check if process still exists
                        os.kill(model_info.pid, 0)
                    except OSError:
                        # Process doesn't exist anymore
                        break
                else:
                    # Process didn't terminate, try SIGKILL
                    os.kill(model_info.pid, 9)  # SIGKILL
                
                logger.info(f"Stopped model server for {model_info.name} (PID: {model_info.pid})")
            except ProcessLookupError:
                logger.warning(f"Process for model {model_info.name} (PID: {model_info.pid}) not found")
            except Exception as e:
                logger.error(f"Error stopping model server: {str(e)}")
        
        # Remove configuration file
        model_name = model_info.name.replace('/', '_')
        config_file = f"config/model_configs/{model_name}.json"
        if os.path.exists(config_file):
            os.remove(config_file)
        
        # Remove from models dictionary
        self.models.pop(model_info.id)
        
        return result
    
    async def shutdown_all(self) -> None:
        """Shutdown all model servers"""
        for model_id in list(self.models.keys()):
            try:
                await self.remove_model(model_id)
            except Exception as e:
                logger.error(f"Error shutting down model {model_id}: {str(e)}")