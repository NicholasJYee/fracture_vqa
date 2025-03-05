import os
import random
import string
import pydicom
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import re
import shutil
import traceback

def create_directory_if_not_exists(directory_path):
    """
    Create a directory if it doesn't exist
    
    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def generate_random_id(length=8):
    """
    Generate a random ID string
    
    Args:
        length (int): Length of the ID string
        
    Returns:
        str: Random ID string
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def sanitize_filename(filename):
    """
    Sanitize a filename by removing invalid characters
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace spaces with underscores and remove invalid characters
    sanitized = re.sub(r'[^\w\-_.]', '_', filename.replace(' ', '_'))
    return sanitized

def save_upload(uploaded_file, directory="uploads"):
    """
    Save an uploaded file to disk
    
    Args:
        uploaded_file: The uploaded file object or path string
        directory (str): Directory to save the file in
        
    Returns:
        str: Path to the saved file
    """
    try:
        print(f"save_upload received: {type(uploaded_file)}")
        create_directory_if_not_exists(directory)
        
        # Handle empty inputs
        if uploaded_file is None:
            raise ValueError("No file provided")
            
        # Special case for Gradio file input
        if isinstance(uploaded_file, list) and len(uploaded_file) > 0:
            print("Converting list to first element")
            uploaded_file = uploaded_file[0]
            
        # Handle string paths (possibly from example files)
        if isinstance(uploaded_file, str):
            if os.path.exists(uploaded_file):
                # Check if it's a directory
                if os.path.isdir(uploaded_file):
                    raise ValueError(f"Provided path is a directory: {uploaded_file}")
                    
                # If it's already a valid path, we can just return it or copy it
                print(f"Using existing file path: {uploaded_file}")
                
                # Optionally copy to uploads directory
                filename = os.path.basename(uploaded_file)
                dest_path = os.path.join(directory, filename)
                
                # Only copy if not already in the target directory
                if os.path.abspath(os.path.dirname(uploaded_file)) != os.path.abspath(directory):
                    shutil.copy2(uploaded_file, dest_path)
                    print(f"Copied to: {dest_path}")
                    return dest_path
                return uploaded_file
            else:
                raise ValueError(f"File not found at path: {uploaded_file}")
        
        # Special handling for Gradio file component which might return a dict
        if isinstance(uploaded_file, dict) and "name" in uploaded_file:
            print("Processing Gradio file dict")
            original_filename = uploaded_file["name"]
            
            # If there's a 'path' key, use that for the source file
            if "path" in uploaded_file:
                source_path = uploaded_file["path"]
                if os.path.exists(source_path) and os.path.isfile(source_path):
                    filename = sanitize_filename(original_filename)
                    file_id = generate_random_id()
                    dest_path = os.path.join(directory, f"{file_id}_{filename}")
                    shutil.copy2(source_path, dest_path)
                    return dest_path
        
        # Handle file-like objects from Gradio
        # Generate a unique filename
        if hasattr(uploaded_file, 'name'):
            original_filename = uploaded_file.name
        else:
            # If no name attribute, generate a random name with the correct extension
            if hasattr(uploaded_file, 'orig_name'):
                # Some Gradio versions use orig_name
                original_filename = uploaded_file.orig_name
            else:
                # Last resort - make something up
                original_filename = f"upload_{generate_random_id()}.jpg"
        
        filename = sanitize_filename(original_filename)
        file_id = generate_random_id()
        file_path = os.path.join(directory, f"{file_id}_{filename}")
        
        print(f"Saving uploaded file to: {file_path}")
        
        # Write the file - handle different types of file objects
        if hasattr(uploaded_file, 'read'):
            # File-like object with read method
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
        elif hasattr(uploaded_file, 'file'):
            # Gradio UploadFile with file attribute
            with open(file_path, "wb") as f:
                shutil.copyfileobj(uploaded_file.file, f)
        elif hasattr(uploaded_file, 'path') and not os.path.isdir(uploaded_file.path):
            # Object with a path attribute (that isn't a directory)
            shutil.copy2(uploaded_file.path, file_path)
        else:
            # Try direct file writing as last resort
            try:
                with open(file_path, "wb") as f:
                    f.write(uploaded_file)
            except:
                raise TypeError(f"Unsupported file object type: {type(uploaded_file)}")
        
        # Verify file was written correctly
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            raise IOError(f"Failed to write file or file is empty: {file_path}")
            
        print(f"Successfully saved file to: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"Error in save_upload: {str(e)}")
        print(traceback.format_exc())
        raise

def read_dicom_tags(dicom_path):
    """
    Read and return important DICOM metadata tags
    
    Args:
        dicom_path (str): Path to the DICOM file
        
    Returns:
        dict: Dictionary of DICOM tags
    """
    try:
        dicom = pydicom.dcmread(dicom_path)
        
        # Extract commonly useful tags
        tags = {
            "Patient ID": getattr(dicom, "PatientID", "Unknown"),
            "Patient Name": str(getattr(dicom, "PatientName", "Unknown")),
            "Study Date": getattr(dicom, "StudyDate", "Unknown"),
            "Modality": getattr(dicom, "Modality", "Unknown"),
            "Body Part": getattr(dicom, "BodyPartExamined", "Unknown"),
            "Image Type": getattr(dicom, "ImageType", ["Unknown"])[0] if hasattr(dicom, "ImageType") else "Unknown",
            "Slice Thickness": getattr(dicom, "SliceThickness", "Unknown"),
        }
        
        return tags
    except Exception as e:
        return {"Error": f"Failed to read DICOM tags: {str(e)}"}

def image_to_base64(image_path):
    """
    Convert an image to base64 string for embedding in HTML
    
    Args:
        image_path (str): Path to the image
        
    Returns:
        str: Base64 encoded image string
    """
    img_format = image_path.split('.')[-1].lower()
    if img_format not in ['jpg', 'jpeg', 'png']:
        img_format = 'png'
    
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    
    return f"data:image/{img_format};base64,{encoded}"

def enhance_xray_for_display(image_path):
    """
    Enhance an X-ray image for better display
    
    Args:
        image_path (str): Path to the X-ray image
        
    Returns:
        str: Path to the enhanced image
    """
    # Load image
    if image_path.lower().endswith('.dcm'):
        try:
            dicom = pydicom.dcmread(image_path)
            image = dicom.pixel_array.astype(np.uint8)
        except Exception as e:
            print(f"Error reading DICOM file: {e}")
            return image_path
    else:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to load image from {image_path}")
                return image_path
        except Exception as e:
            print(f"Error reading image file: {e}")
            return image_path
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Save the enhanced image
    output_path = f"{os.path.splitext(image_path)[0]}_enhanced.png"
    cv2.imwrite(output_path, enhanced)
    
    return output_path 