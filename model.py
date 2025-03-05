import os
import torch
import requests
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import pydicom
import gc  # For explicit garbage collection
import json
import base64
import io

class OllamaLLaVAModel:
    """
    Implementation of an X-ray VQA model using Ollama API with LLaVA model.
    This uses the Ollama API to serve the LLaVA model for visual question answering.
    """
    def __init__(self, device="cpu", ollama_url="http://localhost:11434"):
        super().__init__()
        self.device = device
        self.ollama_url = ollama_url
        self.model_name = "llava"  # Using LLaVA as the model
        
        # Cache for answers to improve response time for repeated questions
        self.answer_cache = {}
        
        print(f"Initialized Ollama LLaVA Model with API endpoint: {ollama_url}")
        print(f"Using model: {self.model_name}")
        
        # Test the Ollama connection
        try:
            self._test_ollama_connection()
            print("✅ Successfully connected to Ollama API")
        except Exception as e:
            print(f"❌ Failed to connect to Ollama API: {str(e)}")
            print("Please make sure Ollama is running with LLaVA model downloaded")
            print("You can install it with: 'ollama pull llava'")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama API"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                llava_available = any(model.get('name', '').startswith('llava') for model in models)
                
                if not llava_available:
                    print("LLaVA model not found in Ollama. You may need to pull it with 'ollama pull llava'")
            else:
                print(f"Error connecting to Ollama API: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to connect to Ollama API: {str(e)}")
    
    def _encode_image_to_base64(self, image):
        """Convert PIL image to base64 string for Ollama API"""
        # Ensure image is a PIL Image
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            else:
                raise ValueError("Image must be a PIL Image or numpy array")
                
        # Convert to RGB if not already
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def generate_answer(self, image, question, max_length=500):
        """
        Generate an answer to a question about a medical image using Ollama's LLaVA.
        
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
            
        # Enhanced question with medical context
        enhanced_question = f"As a radiologist, please analyze this X-ray image and {question}"
        
        # Encode image to base64
        image_base64 = self._encode_image_to_base64(image)
        
        # Prepare the request to Ollama API
        prompt_data = {
            "model": self.model_name,
            "prompt": enhanced_question,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.2,  # Lower temperature for more deterministic medical answers
                "num_predict": max_length
            }
        }
        
        print("Sending request to Ollama API...")
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate", 
                json=prompt_data,
                timeout=30
            )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "")
                
                # Ensure proper sentence structure
                if answer and answer[-1] not in ['.', '!', '?']:
                    answer += '.'
                    
                # Update cache with the generated answer
                self._update_cache(image, question, answer)
                
                return answer
            else:
                error_msg = f"Error from Ollama API: {response.status_code} - {response.text}"
                print(error_msg)
                return f"Error generating answer: {error_msg}"
                
        except Exception as e:
            error_msg = f"Exception while calling Ollama API: {str(e)}"
            print(error_msg)
            return f"Error generating answer: {error_msg}"
    
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
    def __init__(self, ollama_url="http://localhost:11434"):
        """Initialize the model and processor"""
        print("\n" + "="*80)
        print("X-RAY VISUAL QUESTION ANSWERING MODEL INITIALIZATION (OLLAMA + LLaVA)")
        print("="*80)
        
        # Enhanced device selection with better CUDA support
        if torch.cuda.is_available():
            # Get CUDA device info
            cuda_id = 0  # Default to first GPU
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(cuda_id)
            
            print(f"CUDA available: {cuda_device_count} device(s)")
            print(f"Using CUDA device {cuda_id}: {cuda_device_name}")
            
            # Set device with properties
            self.device = torch.device(f"cuda:{cuda_id}")
            
            # Set CUDA optimizations
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 precision mode
            
            # Set max memory usage (adjust based on your GPU)
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)  # Use up to 90% of GPU memory
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (No CUDA/MPS available)")
        
        print(f"Device: {self.device}")
        
        # Initialize Ollama with LLaVA model
        try:
            print("\nInitializing Ollama with LLaVA model...")
            print("MODEL SELECTION: Ollama LLaVA for medical VQA")
            
            # Initialize Ollama LLaVA model
            self.ollama_model = OllamaLLaVAModel(device=self.device, ollama_url=ollama_url)
            
            self.model_type = "ollama-llava"
            print(f"✅ Successfully initialized Ollama LLaVA model for medical VQA")
        except Exception as e:
            error_msg = f"❌ Failed to initialize model: {str(e)}"
            print(error_msg)
            raise RuntimeError(f"Cannot initialize model: {str(e)}")
        
        print("\nACTIVE MODEL INFORMATION:")
        print(f"• Model Type: {self.model_type}")
        print(f"• Running on: {self.device}")
        print(f"• Ollama API: {ollama_url}")
        print(f"• Model: {self.ollama_model.model_name}")
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
        print(f"\n📋 Analysis of X-ray using Ollama LLaVA")
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
        cached_answer = self.ollama_model._check_cache(image, enhanced_question)
        if cached_answer:
            print("🔄 Using cached answer for faster response")
            return cached_answer
        
        # Generate answer
        print("🔬 Analyzing with Ollama LLaVA...")
        answer = self.ollama_model.generate_answer(image, enhanced_question)
        
        # Clear any residual CUDA memory
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()
            
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
        
        # Use our model to generate an answer
        print("🔬 Analyzing the X-ray with Ollama LLaVA...")
        answer = self.ollama_model.generate_answer(image, question)
        
        # Clear any residual CUDA memory
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
            gc.collect()
            
        return answer


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("X-RAY VISUAL QUESTION ANSWERING DEMO (OLLAMA + LLaVA)")
    print("="*80)
    
    # Initialize the model
    print("\n📌 INITIALIZING MODEL...")
    model = XrayVQAModel()
    
    # Example with a local file
    print("\n" + "="*80)
    print("DEMO 1: LOCAL X-RAY ANALYSIS")
    print("="*80)
    image_path = "./example/calcaneal-fracture.jpeg"
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