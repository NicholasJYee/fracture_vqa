import os
import random
import string
import pydicom
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re

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
        uploaded_file: The uploaded file object
        directory (str): Directory to save the file in
        
    Returns:
        str: Path to the saved file
    """
    create_directory_if_not_exists(directory)
    
    # Generate a unique filename
    filename = sanitize_filename(uploaded_file.name)
    file_id = generate_random_id()
    file_path = os.path.join(directory, f"{file_id}_{filename}")
    
    # Write the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    return file_path

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
    # Handle DICOM format
    if image_path.lower().endswith('.dcm'):
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array.astype(float)
        
        # Normalize to 0-1 range
        image = (image - image.min()) / (image.max() - image.min())
    else:
        # Regular image format
        image = np.array(Image.open(image_path).convert('L')).astype(float) / 255.0
    
    # Apply CLAHE for contrast enhancement
    image = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    # Save enhanced image
    output_path = f"{os.path.splitext(image_path)[0]}_enhanced.png"
    cv2.imwrite(output_path, enhanced)
    
    return output_path

def generate_heatmap(image_path, regions_of_interest=None):
    """
    Generate a heatmap overlay for regions of interest in an X-ray
    
    Args:
        image_path (str): Path to the X-ray image
        regions_of_interest (list): List of (x, y, w, h) tuples for ROIs
        
    Returns:
        str: Path to the heatmap image
    """
    # If no ROIs provided, return the original image
    if not regions_of_interest:
        return image_path
    
    # Load image
    if image_path.lower().endswith('.dcm'):
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array.astype(np.uint8)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Create a heatmap overlay
    heatmap = np.zeros_like(image)
    
    # Add heat to regions of interest
    for x, y, w, h in regions_of_interest:
        cv2.rectangle(heatmap, (x, y), (x+w, y+h), 255, -1)
    
    # Blur the heatmap for a smoother look
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    # Normalize heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert grayscale image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Blend images
    alpha = 0.7
    overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap_colored, alpha, 0)
    
    # Save result
    output_path = f"{os.path.splitext(image_path)[0]}_heatmap.png"
    cv2.imwrite(output_path, overlay)
    
    return output_path 