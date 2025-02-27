import os
import torch
from torch import nn
from transformers import BlipProcessor, BlipForQuestionAnswering
import requests
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import pydicom
from skimage import exposure

class XrayVQAModel:
    def __init__(self, model_name="Salesforce/blip-vqa-base"):
        """
        Initialize the X-ray Visual Question Answering model
        
        Args:
            model_name (str): The name of the pre-trained model to use
        """
        # Choose device: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        
    def preprocess_image(self, image_path):
        """
        Preprocess X-ray image for model input
        
        Args:
            image_path (str): Path to the X-ray image file
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Handle DICOM format
        if image_path.lower().endswith('.dcm'):
            dicom = pydicom.dcmread(image_path)
            image = dicom.pixel_array.astype(float)
            
            # Normalize to 0-1 range
            image = (image - image.min()) / (image.max() - image.min())
            
            # Apply contrast limited adaptive histogram equalization (CLAHE)
            image = exposure.equalize_adapthist(image)
            
            # Convert to uint8 and to RGB (DICOM is typically grayscale)
            image = (image * 255).astype(np.uint8)
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
            
            return Image.fromarray(image)
        
        # Handle regular image formats (JPEG, PNG, etc.)
        else:
            image = Image.open(image_path).convert('RGB')
            
            # Convert to numpy for preprocessing
            img_array = np.array(image)
            
            # Check if image is grayscale (X-ray) and convert to RGB if needed
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Enhance contrast for better feature visibility
            if img_array.mean() < 128:  # Dark image typical of X-rays
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                
                # Convert to LAB color space for CLAHE
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                
                # Split the LAB image into channels - cv2.split returns a tuple, convert to list
                lab_planes = list(cv2.split(lab))
                
                # Apply CLAHE to L-channel
                lab_planes[0] = clahe.apply(lab_planes[0])
                
                # Merge the enhanced L-channel back
                lab = cv2.merge(lab_planes)
                
                # Convert back to RGB
                img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(img_array)
    
    def answer_question(self, image_path, question):
        """
        Answer a question about the provided X-ray image
        
        Args:
            image_path (str): Path to the X-ray image
            question (str): Question about the X-ray
            
        Returns:
            str: Answer to the question
        """
        # Preprocess the image
        image = self.preprocess_image(image_path)
        
        # Add medical context to the question if needed
        if not any(term in question.lower() for term in ['x-ray', 'xray', 'radiograph', 'image', 'scan']):
            question = f"In this X-ray image, {question}"
        
        # Prepare inputs for the model
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return answer
        
    def load_from_url(self, image_url, question):
        """
        Load an image from a URL and answer a question about it
        
        Args:
            image_url (str): URL of the X-ray image
            question (str): Question about the X-ray
            
        Returns:
            str: Answer to the question
        """
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Preprocess the image (convert from PIL to numpy and back)
        img_array = np.array(image)
        
        # Apply the same preprocessing as for local images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
            
        if img_array.mean() < 128:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Split the LAB image into channels - convert tuple to list
            lab_planes = list(cv2.split(lab))
            
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        image = Image.fromarray(img_array)
        
        # Add medical context to the question if needed
        if not any(term in question.lower() for term in ['x-ray', 'xray', 'radiograph', 'image', 'scan']):
            question = f"In this X-ray image, {question}"
        
        # Prepare inputs for the model
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return answer


# Example usage
if __name__ == "__main__":
    model = XrayVQAModel()
    
    # Example with a local file
    result = model.answer_question("/Users/Y/Dropbox/1University_of_Toronto/0Graduate_School/2FARIL_internship/fracture-vqa-v6/example/calcaneal-fracture.jpeg", "Describe the x-ray")
    print(f"Answer: {result}")
    
    # Example with a URL
    xray_url = "https://upload.orthobullets.com/topic/1051/images/tongue%20type%20lateral.jpg"
    result = model.load_from_url(xray_url, "Describe the x-ray")
    print(f"Answer: {result}") 