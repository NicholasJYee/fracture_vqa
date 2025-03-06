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
from dotenv import load_dotenv

# Load environment variables
<<<<<<< HEAD
<<<<<<< HEAD
print("Loading environment variables from .env file...")
load_dotenv()

# Check for Hugging Face token
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    print(f"Hugging Face token loaded: {hf_token[:4]}...{hf_token[-4:]}")
else:
    print("No Hugging Face token found in .env file")
=======
=======
>>>>>>> 451b279b8e5e35715ab9a11e4a3eb284180992c1
load_dotenv()

# Get Ollama URL from environment variable or use default
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
<<<<<<< HEAD
>>>>>>> efc448571e4c844db60714480a6fba315f800236
=======
>>>>>>> 451b279b8e5e35715ab9a11e4a3eb284180992c1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("xray-vqa-api")

# Initialize directories
create_directory_if_not_exists("uploads")
create_directory_if_not_exists("examples")
create_directory_if_not_exists("temp")

# Initialize model with Ollama URL
model = XrayVQAModel(ollama_url=OLLAMA_URL)
device_type = "CUDA" if str(model.device) == "cuda" else "MPS" if str(model.device) == "mps" else "CPU"
logger.info(f"Model initialized successfully using device: {device_type} ({model.device})")
logger.info(f"Using Ollama API at: {OLLAMA_URL}")

# Create FastAPI app
app = FastAPI(
    title="X-Ray VQA API",
    description="API for X-Ray Visual Question Answering",
    version="1.0.0"
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
    """Root endpoint to check if API is running."""
    return {"message": "X-Ray VQA API is running", "status": "ok"}

@app.post("/upload-xray/")
async def upload_xray(file: UploadFile = File(...)):
    """
    Upload an X-ray image file
    
    Args:
        file: The uploaded X-ray image
        
    Returns:
        dict: Information about the uploaded file
    """
    try:
        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract file info
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1]
        
        # Check if it's a DICOM file
        is_dicom = file_ext.lower() == '.dcm'
        
        # Get image metadata
        if is_dicom:
            metadata = read_dicom_tags(file_path)
        else:
            try:
                img = Image.open(file_path)
                metadata = {
                    "Format": str(img.format),
                    "Mode": str(img.mode),
                    "Size": f"{img.width} x {img.height}"
                }
            except Exception as e:
                metadata = {"Error": str(e)}
        
        logger.info(f"File uploaded successfully: {file.filename}")
        
        return {
            "filename": file.filename,
            "stored_path": file_path,
            "file_size": file_size,
            "content_type": file.content_type,
            "is_dicom": is_dicom,
            "metadata": metadata,
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/ask-question/")
async def ask_question(image_path: str = Form(...), question: str = Form(...)):
    """
    Ask a question about an X-ray image
    
    Args:
        image_path: Path to the uploaded X-ray image
        question: Question about the X-ray
        
    Returns:
        dict: Answer to the question
    """
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
        
        # Process the question
        start_time = time.time()
        answer = model.answer_question(image_path, question)
        processing_time = time.time() - start_time
        
        logger.info(f"Question answered: '{question}' for image: {image_path}")
        
        return {
            "question": question,
            "answer": answer,
            "image_path": image_path,
            "processing_time_seconds": round(processing_time, 2)
        }
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/enhance-image/")
async def enhance_image(image_path: str = Form(...)):
    """
    Enhance an X-ray image for better visibility
    
    Args:
        image_path: Path to the X-ray image
        
    Returns:
        dict: Information about the enhanced image
    """
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"File not found: {image_path}")
        
        # Enhance the image
        enhanced_path = enhance_xray_for_display(image_path)
        
        logger.info(f"Image enhanced: {image_path} -> {enhanced_path}")
        
        return {
            "original_image": image_path,
            "enhanced_image": enhanced_path,
            "success": True
        }
    
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error enhancing image: {str(e)}")

@app.get("/get-image/{image_path:path}")
async def get_image(image_path: str):
    """
    Retrieve an image file
    
    Args:
        image_path: Path to the image
        
    Returns:
        FileResponse: The image file
    """
    try:
        # Check if the file exists
        full_path = os.path.join("uploads", image_path)
        
        if not os.path.exists(full_path):
            # Check if it's in temp directory
            full_path = os.path.join("temp", image_path)
            if not os.path.exists(full_path):
                raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        
        # Handle DICOM files
        if full_path.lower().endswith('.dcm'):
            try:
                # Convert DICOM to PNG for display
                dicom = pydicom.dcmread(full_path)
                img_array = dicom.pixel_array
                
                # Normalize and convert to 8-bit
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                
                # Create a temporary PNG file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                    temp_path = temp.name
                
                # Save as PNG
                Image.fromarray(img_array).save(temp_path)
                return FileResponse(temp_path, media_type="image/png")
            
            except Exception as e:
                logger.error(f"Error converting DICOM to viewable format: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing DICOM: {str(e)}")
        
        # Determine media type based on file extension
        _, ext = os.path.splitext(full_path)
        if ext.lower() in ['.jpg', '.jpeg']:
            media_type = "image/jpeg"
        elif ext.lower() == '.png':
            media_type = "image/png"
        else:
            media_type = "application/octet-stream"
        
        return FileResponse(full_path, media_type=media_type)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")

@app.get("/list-images/")
async def list_images(directory: Optional[str] = "uploads"):
    """
    List available X-ray images
    
    Args:
        directory: Directory to list images from
        
    Returns:
        dict: List of available images
    """
    try:
        # Validate directory to prevent directory traversal
        if directory not in ["uploads", "examples", "temp"]:
            raise HTTPException(status_code=400, detail="Invalid directory specified")
        
        # Get list of images
        image_files = []
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                file_size = os.path.getsize(file_path)
                file_time = os.path.getmtime(file_path)
                
                image_files.append({
                    "filename": filename,
                    "path": file_path,
                    "size_bytes": file_size,
                    "last_modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_time))
                })
        
        return {
            "directory": directory,
            "image_count": len(image_files),
            "images": image_files
        }
    
    except Exception as e:
        logger.error(f"Error listing images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing images: {str(e)}")

@app.delete("/delete-image/{filename}")
async def delete_image(filename: str, directory: Optional[str] = "uploads"):
    """
    Delete an X-ray image
    
    Args:
        filename: Name of the file to delete
        directory: Directory containing the file
        
    Returns:
        dict: Status of the deletion
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
            "message": f"File {filename} successfully deleted"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 