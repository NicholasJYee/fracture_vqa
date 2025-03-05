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
import time

class OllamaLLaVAModel:
    """
    Implementation of an X-ray VQA model using Ollama API with LLaVA model.
    This uses the Ollama API to serve the LLaVA model for visual question answering.
    """
    def __init__(self, device="cpu", ollama_url="http://localhost:11434", cache_dir=None):
        super().__init__()
        self.device = device
        self.ollama_url = ollama_url
        self.model_name = "llava:latest"  # Using LLaVA as the model
        
        # Timeouts for API requests (in seconds)
        self.connection_timeout = 10  # Timeout for establishing connection
        self.read_timeout = 120       # Longer timeout for model inference
        
        # Cache for answers to improve response time for repeated questions
        self.answer_cache = {}
        
        # Set cache directory
        self.cache_dir = cache_dir
        if self.cache_dir:
            print(f"OllamaLLaVAModel using cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Initialized Ollama LLaVA Model with API endpoint: {ollama_url}")
        print(f"Using model: {self.model_name}")
        print(f"API timeouts: connection={self.connection_timeout}s, read={self.read_timeout}s")
        
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
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=(self.connection_timeout, 10)  # Use shorter read timeout for simple API check
            )
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
        print(f"Using timeouts: connection={self.connection_timeout}s, read={self.read_timeout}s")
        
        try:
            # Call Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate", 
                json=prompt_data,
                timeout=(self.connection_timeout, self.read_timeout)
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
    
    def _check_cache(self, image, question):
        """
        Check if we already have a cached answer for this image+question
        """
        if not self.cache_dir:
            return None
            
        try:
            # Create a hash of the image and question to use as a key
            # We use a simple hash here - a production system would use a more robust approach
            img_bytes = self._encode_image_to_base64(image)
            cache_key = f"{hash(img_bytes)}-{hash(question)}"
            cache_file = os.path.join(self.cache_dir, f"response_{cache_key}.json")
            
            if os.path.exists(cache_file):
                print(f"Cache hit: Using cached answer for question")
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    return cached_data['answer']
            return None
        except Exception as e:
            print(f"Cache check error: {e}")
            return None
            
    def _update_cache(self, image, question, answer):
        """
        Update the cache with a new answer
        """
        if not self.cache_dir:
            return
            
        try:
            # Create a hash of the image and question to use as a key
            img_bytes = self._encode_image_to_base64(image)
            cache_key = f"{hash(img_bytes)}-{hash(question)}"
            cache_file = os.path.join(self.cache_dir, f"response_{cache_key}.json")
            
            # Save to cache file
            with open(cache_file, 'w') as f:
                json.dump({
                    'question': question,
                    'answer': answer,
                    'timestamp': time.time()
                }, f)
                
            print(f"Saved answer to cache: {cache_file}")
        except Exception as e:
            print(f"Cache update error: {e}")
            # Continue execution even if caching fails

class XrayVQAModel:
    def __init__(self, ollama_url="http://localhost:11434", cache_dir=None):
        """Initialize the model and processor"""
        print("\n" + "="*80)
        print("X-RAY VISUAL QUESTION ANSWERING MODEL INITIALIZATION (OLLAMA + LLaVA)")
        print("="*80)
        
        # Set cache directory if provided
        self.cache_dir = cache_dir
        if self.cache_dir:
            print(f"Using cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
        
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
            print(f"Initializing LLaVA model via Ollama API at {ollama_url}")
            if self.device == "cuda":
                print("Using CUDA device with Ollama")
            self.ollama_model = OllamaLLaVAModel(device=self.device, ollama_url=ollama_url, cache_dir=self.cache_dir)
            print("Successfully initialized LLaVA model via Ollama")
        except Exception as e:
            print(f"Error initializing Ollama model: {str(e)}")
            raise
        
        print("\nACTIVE MODEL INFORMATION:")
        print(f"• Model Type: {self.model_type}")
        print(f"• Running on: {self.device}")
        print(f"• Ollama API: {ollama_url}")
        print(f"• Model: {self.ollama_model.model_name}")
        print("="*80 + "\n")
    
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
        # Use a shorter timeout for image download since it's just fetching data, not processing
        response = requests.get(image_url, timeout=(self.ollama_model.connection_timeout, 30))
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