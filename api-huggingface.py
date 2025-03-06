from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import os
import time
import json
from model import XrayVQAModel
from utils import save_upload, enhance_xray_for_display, read_dicom_tags, create_directory_if_not_exists
import tempfile
from PIL import Image
import pydicom
import numpy as np
from typing import List, Optional
import shutil
import logging
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Ollama URL from environment variable or use default
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
HF_API_URL = os.getenv("HF_API_URL", None)  # For alternative API usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("xray-vqa-api")

# Check for CUDA or other accelerators
logger.info("="*50)
logger.info("DEVICE CONFIGURATION")
logger.info("="*50)

if torch.cuda.is_available():
    # Get CUDA device details
    cuda_device_count = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0)
    logger.info(f"CUDA available: {cuda_device_count} device(s)")
    logger.info(f"Using CUDA device: {cuda_device_name}")
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
# Check for MPS (Metal Performance Shaders for Mac)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.backends.mps.is_built = True
    logger.info("MPS (Metal Performance Shaders) is available and enabled")
    device = "mps"
else:
    logger.info("No GPU acceleration available, using CPU")
    device = "cpu"

logger.info(f"Selected device: {device}")
logger.info("="*50)

# Initialize directories
create_directory_if_not_exists("uploads")
create_directory_if_not_exists("temp")

# Initialize model with Ollama URL - with error handling for Hugging Face environment
try:
    # Initialize the model
    model = XrayVQAModel(ollama_url=OLLAMA_URL)
    USING_OLLAMA = True
    logger.info(f"Model initialized successfully using device: {model.device}")
    logger.info(f"Using Ollama API at: {OLLAMA_URL}")
except Exception as e:
    logger.error(f"Error initializing Ollama model: {str(e)}")
    logger.info("Please verify that Ollama is properly set up and running.")
    logger.info("For Hugging Face Spaces, you may need to configure an external Ollama server.")
    USING_OLLAMA = False
    logger.info("Continuing without active model...")

# Create FastAPI app
app = FastAPI(
    title="X-Ray VQA API",
    description="API for X-Ray Visual Question Answering using LLaVA via Ollama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "X-Ray VQA API is running",
        "docs": "/docs",
        "status": "active",
        "ollama_status": "connected" if USING_OLLAMA else "disconnected",
        "ollama_url": OLLAMA_URL,
        "device": device,
    }

@app.post("/upload-xray/")
async def upload_xray(file: UploadFile = File(...)):
    """
    Upload an X-ray image for analysis.
    
    Args:
        file: The X-ray image file to upload
        
    Returns:
        dict: Information about the uploaded file
    """
    try:
        # Save the uploaded file
        file_path = save_upload(file.file, file.filename)
        
        # Get file info
        file_info = {
            "filename": os.path.basename(file_path),
            "path": file_path,
            "size": os.path.getsize(file_path),
            "type": file.content_type or "unknown"
        }
        
        # For DICOM files, extract metadata
        if file_path.lower().endswith('.dcm'):
            try:
                tags = read_dicom_tags(file_path)
                file_info["dicom_metadata"] = tags
            except Exception as e:
                logger.error(f"Error reading DICOM tags: {str(e)}")
        
        logger.info(f"Successfully uploaded file: {file_path}")
        return {
            "message": "File uploaded successfully",
            "file_info": file_info,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/ask-question/")
async def ask_question(image_path: str = Form(...), question: str = Form(...)):
    """
    Ask a question about the uploaded X-ray image.
    
    Args:
        image_path: Path to the X-ray image to analyze
        question: Question about the X-ray image
        
    Returns:
        dict: Answer to the question
    """
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
            
        # Check if question is provided
        if not question or question.strip() == "":
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Processing question: \"{question}\" for image: {image_path}")
        
        # Generate answer
        if USING_OLLAMA:
            start_time = time.time()
            answer = model.answer_question(image_path, question)
            end_time = time.time()
            
            logger.info(f"Generated answer in {end_time - start_time:.2f} seconds")
            
            return {
                "question": question,
                "answer": answer,
                "image_path": image_path,
                "processing_time": round(end_time - start_time, 2),
                "status": "success"
            }
        elif HF_API_URL:
            # Implement alternative API call here if needed
            return {
                "question": question,
                "answer": "Using alternative API for inference (Ollama not available)",
                "image_path": image_path,
                "status": "success"
            }
        else:
            return {
                "question": question,
                "answer": "⚠️ Ollama service is not available. Please make sure Ollama is running with LLaVA model installed.",
                "image_path": image_path,
                "status": "error",
                "error": "ollama_unavailable"
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/enhance-image/")
async def enhance_image(image_path: str = Form(...)):
    """
    Enhance an X-ray image for better visibility.
    
    Args:
        image_path: Path to the X-ray image to enhance
        
    Returns:
        dict: Path to the enhanced image
    """
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
        
        # Enhance the image
        enhanced_path = enhance_xray_for_display(image_path)
        
        logger.info(f"Enhanced image saved to: {enhanced_path}")
        
        return {
            "original_image": image_path,
            "enhanced_image": enhanced_path,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enhancing image: {str(e)}")

@app.get("/get-image/{image_path:path}")
async def get_image(image_path: str):
    """
    Get an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        FileResponse: The image file
    """
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
        
        # Return the file
        return FileResponse(
            image_path,
            media_type="image/jpeg" if image_path.endswith(('.jpg', '.jpeg')) else "image/png"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting image: {str(e)}")

@app.get("/list-images/")
async def list_images(directory: Optional[str] = "uploads"):
    """
    List all images in a directory.
    
    Args:
        directory: Directory to list images from
        
    Returns:
        dict: List of images
    """
    try:
        # Validate directory to prevent directory traversal
        if directory not in ["uploads", "temp"]:
            raise HTTPException(status_code=400, detail="Invalid directory specified")
        
        # Get list of files
        files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm')):
                file_info = {
                    "filename": filename,
                    "path": file_path,
                    "size": os.path.getsize(file_path),
                    "last_modified": os.path.getmtime(file_path),
                    "type": "dicom" if filename.lower().endswith('.dcm') else "image"
                }
                files.append(file_info)
        
        return {
            "directory": directory,
            "files": files,
            "count": len(files),
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing images: {str(e)}")

@app.delete("/delete-image/{filename}")
async def delete_image(filename: str, directory: Optional[str] = "uploads"):
    """
    Delete an image file.
    
    Args:
        filename: Name of the file to delete
        directory: Directory the file is in
        
    Returns:
        dict: Information about the deleted file
    """
    try:
        # Validate directory to prevent directory traversal
        if directory not in ["uploads", "temp"]:
            raise HTTPException(status_code=400, detail="Invalid directory specified")
        
        # Check if the file exists
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        
        # Delete the file
        os.remove(file_path)
        logger.info(f"File deleted: {file_path}")
        
        return {
            "filename": filename,
            "directory": directory,
            "deleted": True,
            "message": f"File {filename} successfully deleted",
            "status": "success"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/status")
async def get_status():
    """Get API status information"""
    return {
        "status": "active",
        "ollama_status": "connected" if USING_OLLAMA else "disconnected",
        "ollama_url": OLLAMA_URL,
        "device": device,
        "time": time.time()
    }

# Launch the FastAPI app with Uvicorn
if __name__ == "__main__":
    # Run with settings suitable for Hugging Face Spaces
    uvicorn.run(
        "api-huggingface:app", 
        host="0.0.0.0", 
        port=7860,  # Standard port for Hugging Face Spaces
        log_level="info"
    ) 