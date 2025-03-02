import os
import torch
from torch import nn
import requests
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import pydicom
from skimage import exposure
import tempfile
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForQuestionAnswering, ViTImageProcessor, ViTModel, BertTokenizer, BertModel

# Load environment variables at module level
load_dotenv()

class MedQFormer3D(nn.Module):
    """
    MedQFormer module adapted from the MedBLIP paper (https://arxiv.org/abs/2305.10799)
    This module bridges the visual features from a vision encoder with a language model
    for medical image understanding with pseudo-3D features.
    """
    def __init__(self, vision_hidden_size=768, text_hidden_size=768, num_query_tokens=8, num_slices=3):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        self.num_slices = num_slices
        
        # Learnable query tokens that will be enhanced with visual information
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, text_hidden_size))
        torch.nn.init.normal_(self.query_tokens, std=0.02)
        
        # Projection from vision encoder space to text encoder space
        self.vision_proj = nn.Linear(vision_hidden_size, text_hidden_size)
        
        # 3D spatial fusion module - convolves across slices to create 3D context
        self.spatial_fusion = nn.Sequential(
            nn.Conv1d(num_slices, 1, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Cross-attention layers to blend visual and query token features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_hidden_size,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # LayerNorm and FFN for query tokens
        self.layer_norm1 = nn.LayerNorm(text_hidden_size)
        self.layer_norm2 = nn.LayerNorm(text_hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(text_hidden_size, text_hidden_size * 4),
            nn.GELU(),
            nn.Linear(text_hidden_size * 4, text_hidden_size)
        )
    
    def forward(self, vision_features_3d):
        # vision_features_3d shape: [batch_size, num_slices, seq_length, hidden_size]
        batch_size, num_slices, seq_length, hidden_size = vision_features_3d.shape
        
        # Reshape for processing
        vision_features_flat = vision_features_3d.view(batch_size * num_slices, seq_length, hidden_size)
        
        # Project all slice features to text space
        vision_features_proj = self.vision_proj(vision_features_flat)
        vision_features_proj = vision_features_proj.view(batch_size, num_slices, seq_length, -1)
        
        # Fuse 3D information across slices using 1D convolution
        # Transpose for conv1d: [batch, seq_length, slices, hidden_size]
        vision_features_proj = vision_features_proj.permute(0, 2, 1, 3)
        vision_features_proj = vision_features_proj.reshape(batch_size * seq_length, num_slices, -1)
        vision_features_fused = self.spatial_fusion(vision_features_proj)
        vision_features_fused = vision_features_fused.squeeze(1)
        vision_features_fused = vision_features_fused.reshape(batch_size, seq_length, -1)
        
        # Expand query tokens to match batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Cross-attention: query tokens attend to 3D-fused vision features
        query_tokens = self.layer_norm1(query_tokens)
        vision_features_fused = self.layer_norm1(vision_features_fused)
        
        # Self-attention mechanism
        query_tokens_attn, _ = self.cross_attention(
            query=query_tokens,
            key=vision_features_fused,
            value=vision_features_fused
        )
        
        # Residual connection and FFN
        query_tokens = query_tokens + query_tokens_attn
        query_tokens = query_tokens + self.ffn(self.layer_norm2(query_tokens))
        
        return query_tokens

class MedBLIPModel(nn.Module):
    """
    Medical BLIP implementation for X-ray analysis using the original BLIP architecture.
    This model leverages BLIP's integrated vision-language approach for more efficient processing.
    """
    def __init__(self, device, num_slices=3):
        super().__init__()
        self.device = device
        self.num_slices = num_slices
        
        # Check for Hugging Face token in environment variables
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Use BLIP model for VQA
        model_id = "Salesforce/blip-vqa-base"
        print(f"Loading BLIP model: {model_id}")
        
        # Use auth token if available when loading models
        auth_token = self.hf_token if self.hf_token else None
        if auth_token:
            print(f"Using Hugging Face authentication token: {auth_token[:4]}...{auth_token[-4:]}")
            self.processor = BlipProcessor.from_pretrained(model_id, use_auth_token=auth_token)
            self.model = BlipForQuestionAnswering.from_pretrained(model_id, use_auth_token=auth_token)
        else:
            print("No Hugging Face token found, loading models without authentication")
            self.processor = BlipProcessor.from_pretrained(model_id)
            self.model = BlipForQuestionAnswering.from_pretrained(model_id)
        
        # Move model to device
        self.model.to(device)
        
        # Cache for answers to improve response time for repeated questions
        self.answer_cache = {}
        
        # List of medical terms for better medical context
        self.medical_terms = [
            "fracture", "break", "pneumonia", "opacity", "normal", 
            "abnormal", "effusion", "pneumothorax", "dislocation"
        ]
    
    def preprocess_image_for_model(self, image):
        """Simple preprocessing for images"""
        # Check if it's a PIL image, if so convert to numpy
        if isinstance(image, Image.Image):
            return image
        else:
            # Convert numpy array to PIL
            if len(image.shape) == 2:  # If grayscale
                image = np.stack([image] * 3, axis=2)
            return Image.fromarray(image.astype(np.uint8))
        
    def create_3d_from_2d(self, image):
        """
        Create a simplified representation from a 2D image with minimal processing.
        For BLIP, we only need the original image.
        """
        # Return the original image since BLIP processes a single image
        return self.preprocess_image_for_model(image)
    
    def generate_answer(self, image, question, max_length=50):
        """
        Generate an answer to a question about a medical image using BLIP.
        
        Args:
            image: PIL Image of a medical scan
            question: String question about the image
            max_length: Maximum length of the generated answer
            
        Returns:
            String answer to the question
        """
        # Check cache first for faster response
        cached_answer = self._check_cache(image, question)
        if cached_answer:
            return cached_answer
            
        # Ensure image is properly formatted
        image = self.create_3d_from_2d(image)
        
        # Enhanced question with basic prefix for medical context
        enhanced_question = f"As a radiologist, {question}"
        
        # Process with BLIP model
        print("Generating response with BLIP...")
        inputs = self.processor(image, enhanced_question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Ensure proper sentence structure
        if answer and answer[-1] not in ['.', '!', '?']:
            answer += '.'
            
        # Update cache with the generated answer
        self._update_cache(image, question, answer)
            
        return answer

    # Add a simple cache check method
    def _check_cache(self, image, question):
        """Check if the answer is in cache"""
        # Create a simple hash for the image using its size and mean pixel value
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            img_hash = f"{img_array.shape}_{np.mean(img_array):.2f}"
        else:
            img_hash = "unknown_image"
            
        # Create cache key
        cache_key = f"{img_hash}_{question}"
        
        # Return cached answer if available
        if cache_key in self.answer_cache:
            print("Using cached answer")
            return self.answer_cache[cache_key]
        
        return None
        
    # Update cache method
    def _update_cache(self, image, question, answer):
        """Update the answer cache"""
        # Create a simple hash for the image
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            img_hash = f"{img_array.shape}_{np.mean(img_array):.2f}"
        else:
            img_hash = "unknown_image"
            
        # Create cache key
        cache_key = f"{img_hash}_{question}"
        
        # Store in cache (limit cache size to 100 entries)
        if len(self.answer_cache) >= 100:
            # Remove oldest entry
            oldest_key = next(iter(self.answer_cache))
            del self.answer_cache[oldest_key]
            
        self.answer_cache[cache_key] = answer

class XrayVQAModel:
    def __init__(self):
        """Initialize the model and processor"""
        print("\n" + "="*80)
        print("X-RAY VISUAL QUESTION ANSWERING MODEL INITIALIZATION")
        print("="*80)
        
        # Choose device: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Initialize lightweight model for medical VQA
        try:
            print("\nLoading lightweight model for medical VQA...")
            print("MODEL SELECTION: Lightweight VQA model")
            print("Using smaller models for faster predictions")
            
            # Initialize our model
            self.medblip_model = MedBLIPModel(device=self.device, num_slices=3)
            
            self.model_type = "lightweight-radiological"
            print(f"✅ Successfully loaded lightweight model with basic radiological processing")
        except Exception as e:
            error_msg = f"❌ Failed to initialize model: {str(e)}"
            print(error_msg)
            raise RuntimeError(f"Cannot initialize model: {str(e)}")
        
        print("\nACTIVE MODEL INFORMATION:")
        print(f"• Model Type: {self.model_type}")
        print(f"• Running on: {self.device}")
        print("="*80 + "\n")
        
        # Set up the cache directory
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def preprocess_image(self, image_path):
        """
        Ultra-fast preprocessing for X-ray imaging with minimal operations
        
        Args:
            image_path (str): Path to the X-ray image file
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Handle DICOM format
        if image_path.lower().endswith('.dcm'):
            try:
                dicom = pydicom.dcmread(image_path)
                image = dicom.pixel_array.astype(float)
                
                # Simple normalization
                image = (image - image.min()) / (image.max() - image.min())
                
                # Convert to uint8 and to RGB
                image = (image * 255).astype(np.uint8)
                image = np.stack([image] * 3, axis=-1)  # Convert to RGB
                
                return Image.fromarray(image)
            except Exception as e:
                print(f"DICOM processing error: {e}. Falling back to basic processing.")
                # Fall back to basic processing if DICOM fails
                try:
                    return Image.open(image_path).convert('RGB')
                except:
                    # Last resort - create a blank image with error text
                    blank = Image.new('RGB', (224, 224), color=(0, 0, 0))
                    return blank
        
        # Handle regular image formats (JPEG, PNG, etc.)
        else:
            try:
                # Fast direct loading
                return Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Image processing error: {e}. Creating blank image.")
                # Last resort - create a blank image
                blank = Image.new('RGB', (224, 224), color=(0, 0, 0))
                return blank
    
    def _enhance_question_with_radiological_context(self, question):
        """Simple question enhancement for faster processing"""
        # Just add a basic prefix for all questions
        return f"As a radiologist, {question}"
    
    def answer_question(self, image_path, question):
        """
        Answer a question about the provided X-ray image with faster processing
        
        Args:
            image_path (str): Path to the X-ray image
            question (str): Question about the X-ray
            
        Returns:
            str: Radiological assessment answering the question
        """
        print(f"\n📋 Fast analysis of X-ray")
        print(f"📝 Question: \"{question}\"")
        
        # Handle empty or None question
        if not question or question is None:
            return "Please ask a specific question about this radiographic study."
            
        # Simple question enhancement
        enhanced_question = self._enhance_question_with_radiological_context(question)
        
        # Preprocess the image with simplified pipeline
        print(f"🖼️  Fast preprocessing of radiographic image")
        image = self.preprocess_image(image_path)
        
        # Check if we have a cached answer
        cached_answer = self.medblip_model._check_cache(image, enhanced_question)
        if cached_answer:
            print("🔄 Using cached answer for faster response")
            return cached_answer
        
        # Generate answer
        print("🔬 Fast analysis in progress...")
        answer = self.medblip_model.generate_answer(image, enhanced_question)
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
        print(f"\n🌐 Loading X-ray from URL")
        print(f"📝 Question: \"{question}\"")
        
        # Handle empty or None question
        if not question or question is None:
            return "Please ask a specific question about this X-ray image."
            
        # Download the image
        print(f"⬇️ Downloading image from URL")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Preprocess the image (convert from PIL to numpy and back)
        img_array = np.array(image)
        
        # Apply the same preprocessing as for local images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
            
        if img_array.mean() < 128:
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        image = Image.fromarray(img_array)
        
        # Use our PeFoMed model to generate an answer
        print("🔬 Analyzing the X-ray...")
        answer = self.medblip_model.generate_answer(image, question)
        
        return answer


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("X-RAY VISUAL QUESTION ANSWERING DEMO")
    print("="*80)
    
    # Initialize the model
    print("\n📌 INITIALIZING MODEL...")
    model = XrayVQAModel()
    
    # Example with a local file
    print("\n" + "="*80)
    print("DEMO 1: LOCAL X-RAY ANALYSIS")
    print("="*80)
    image_path = "/Users/Y/Dropbox/1University_of_Toronto/0Graduate_School/2FARIL_internship/fracture-vqa-v6/example/calcaneal-fracture.jpeg"
    question = "Is there a fracture in this X-ray?"
    print(f"\n📁 Analyzing local image: {os.path.basename(image_path)}")
    print(f"❓ Question: \"{question}\"")
    result = model.answer_question(image_path, question)
    print(f"\n🔎 RESULT:")
    print(f"{result}")
    
    # Example with a URL
    print("\n" + "="*80)
    print("DEMO 2: REMOTE X-RAY ANALYSIS")
    print("="*80)
    xray_url = "https://upload.orthobullets.com/topic/1051/images/tongue%20type%20lateral.jpg"
    question = "Describe the x-ray and identify any abnormalities"
    print(f"\n🌐 Analyzing image from URL: {xray_url}")
    print(f"❓ Question: \"{question}\"")
    result = model.load_from_url(xray_url, question)
    print(f"\n🔎 RESULT:")
    print(f"{result}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETED")
    print("="*80) 