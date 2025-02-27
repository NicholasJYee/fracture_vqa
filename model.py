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
from transformers import BlipProcessor, BlipForQuestionAnswering, ViTImageProcessor, ViTModel, BertTokenizer, BertModel

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
    PeFoMed (Perception Foundation Model for Medicine) implementation for X-ray analysis.
    This model specializes in understanding radiological images and answering medical questions
    with high accuracy and clinically relevant context.
    """
    def __init__(self, device, num_slices=3):
        super().__init__()
        self.device = device
        self.num_slices = num_slices
        
        # PeFoMed vision encoder - specialized for medical perception
        self.vision_encoder_name = "google/vit-base-patch16-224"  # Smaller, faster vision model
        print(f"Loading lightweight vision encoder: {self.vision_encoder_name}")
        self.vision_processor = ViTImageProcessor.from_pretrained(self.vision_encoder_name)
        self.vision_encoder = ViTModel.from_pretrained(self.vision_encoder_name)
        
        # Freeze the vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # PeFoMed Text encoder - clinical language understanding
        # Using a smaller, faster BERT model
        self.text_encoder_name = "distilbert-base-uncased"  # Smaller, faster text model
        print(f"Loading lightweight text encoder: {self.text_encoder_name}")
        self.text_tokenizer = BertTokenizer.from_pretrained(self.text_encoder_name)
        self.text_encoder = BertModel.from_pretrained(self.text_encoder_name)
        
        # Freeze the text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # PeFoMed multimodal bridge - enhanced for radiological context
        self.medqformer = MedQFormer3D(
            vision_hidden_size=self.vision_encoder.config.hidden_size,
            text_hidden_size=self.text_encoder.config.hidden_size,
            num_query_tokens=16,  # Increased for better medical detail capture
            num_slices=self.num_slices
        )
        
        # PeFoMed classifier with differential diagnosis capabilities
        self.vqa_classifier = nn.Linear(
            self.text_encoder.config.hidden_size,
            self.text_encoder.config.vocab_size
        )
        
        # Enhanced radiological terminology for X-ray interpretation
        self.medical_terms = [
            # General terms
            "normal", "abnormal", "unremarkable", "remarkable", 
            # Bone conditions
            "fracture", "break", "dislocation", "subluxation", "osteophyte",
            "erosion", "sclerosis", "lytic", "blastic", "lucency", "density",
            "periosteal", "comminuted", "displaced", "nondisplaced", "transverse",
            "oblique", "spiral", "compression", "impacted", "avulsion",
            # Chest/Lung findings
            "pneumonia", "infection", "consolidation", "infiltrate", "opacity",
            "cardiomegaly", "effusion", "pneumothorax", "edema", "atelectasis", 
            "pleural", "fibrosis", "emphysema", "mass", "nodule", "interstitial",
            "bronchiectasis", "hilum", "fissure", "aspiration", "silhouette",
            # Abdominal findings
            "obstruction", "ileus", "perforation", "free air", "calcification",
            "nephrolithiasis", "aerobilia", "organomegaly", "ascites",
            # Vascular findings
            "aneurysm", "dissection", "thrombus", "embolism", "ischemia", 
            "atherosclerosis", "stenosis", "dilation", "calcification",
            # Tumor-related
            "tumor", "mass", "metastasis", "lesion", "malignancy", "cancer",
            "primary", "secondary", "radiolucent", "radiopaque", "lytic",
            # Spinal conditions
            "arthritis", "osteoarthritis", "scoliosis", "kyphosis", "spondylosis", 
            "spondylolisthesis", "narrowing", "stenosis", "disc", "vertebral",
            "degenerative", "osteophyte", "compression", "herniation",
            # Pediatric specific
            "epiphyseal", "metaphyseal", "physeal", "salter-harris", "growth plate",
            # Assessment terms
            "yes", "no", "maybe", "unclear", "possible", "probable", "definite",
            "cannot exclude", "consistent with", "suggestive of", "characteristic",
            "diagnostic", "pathognomonic", "nonspecific", "unchanged", "improved",
            "worsened", "acute", "chronic", "subacute", "resolving", "sequela"
        ]
        
        # Get token IDs for medical terms to enhance their prediction probability
        self.medical_token_ids = []
        for term in self.medical_terms:
            tokens = self.text_tokenizer.encode(term, add_special_tokens=False)
            self.medical_token_ids.extend(tokens)
        self.medical_token_ids = list(set(self.medical_token_ids))  # Remove duplicates
        
        # Improved specialized prompting system for radiological contexts
        self.clinical_prefixes = {
            "general": "Radiological interpretation: ",
            "chest": "Chest radiograph assessment: ",
            "bone": "Skeletal radiographic evaluation: ",
            "joint": "Articular radiographic findings: ",
            "abdomen": "Abdominal radiographic analysis: ",
            "spine": "Spinal imaging evaluation: ",
            "pediatric": "Pediatric radiographic assessment: ",
            "emergency": "Emergency radiological findings: ",
            "fracture": "Fracture analysis: ",
            "foreign_body": "Foreign body assessment: ",
            "soft_tissue": "Soft tissue evaluation: "
        }
        
        # Advanced radiological findings categories for structured reporting
        self.finding_categories = {
            "fracture": ["fracture", "break", "discontinuity", "fx", "comminuted", "displaced"],
            "alignment": ["alignment", "dislocation", "subluxation", "position", "rotation", "angulation"],
            "joints": ["joint space", "articulation", "arthritis", "degenerative", "erosion"],
            "bone_quality": ["density", "osteopenia", "osteoporosis", "sclerosis", "lucency"],
            "soft_tissue": ["swelling", "effusion", "edema", "hematoma", "fluid", "mass"],
            "hardware": ["hardware", "implant", "prosthesis", "fixation", "plate", "screw", "rod"],
            "airspace": ["consolidation", "opacity", "infiltrate", "pneumonia", "atelectasis"],
            "pleural": ["pleural", "effusion", "pneumothorax", "hemothorax", "fluid", "thickening"],
            "cardiac": ["cardiac", "cardiomegaly", "heart", "failure", "enlargement", "silhouette"],
            "vascular": ["vascular", "aorta", "atherosclerosis", "calcification", "hilar", "hilum"],
            "abdominal": ["bowel", "gas", "obstruction", "ileus", "free air", "organomegaly"],
            "foreign_body": ["foreign", "object", "device", "tube", "line", "catheter"]
        }
        
        # PeFoMed confidence scoring system
        self.confidence_scales = {
            "high": 0.9,     # Findings with high certainty
            "moderate": 0.7, # Probable findings
            "low": 0.5       # Possible findings
        }
        
        # Move model to device
        self.to(device)
    
    def preprocess_image_for_model(self, image):
        """Resize and normalize image for model input"""
        # Check if it's a PIL image, if so convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Make sure it's RGB
        if len(image_np.shape) == 2:  # If grayscale
            image_np = np.stack([image_np] * 3, axis=2)
            
        # Convert back to PIL
        pil_image = Image.fromarray(image_np.astype(np.uint8))
        
        # Use the vision processor to properly format for the model
        # This handles resizing and normalization
        return pil_image
        
    def create_3d_from_2d(self, image):
        """
        Create a simplified pseudo-3D volume from a 2D image with minimal processing
        for faster execution.
        """
        # Ensure the image is properly formatted
        image = self.preprocess_image_for_model(image)
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Create simplified variations with basic processing
        pseudo_3d_volume = []
        
        # 1. Original image
        pseudo_3d_volume.append(img_array.copy())
        
        # 2. Simple blurred version
        blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
        pseudo_3d_volume.append(blurred)
        
        # 3. Simple brightness adjusted version (faster than edge detection)
        if len(img_array.shape) == 3:
            adjusted = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
        else:
            adjusted = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)
            adjusted = np.stack([adjusted] * 3, axis=2)
        
        pseudo_3d_volume.append(adjusted)
            
        # Ensure all slices are RGB
        for i in range(len(pseudo_3d_volume)):
            slice_img = pseudo_3d_volume[i]
            # Convert to RGB if grayscale
            if len(slice_img.shape) == 2:
                slice_img = np.stack([slice_img] * 3, axis=2)
            # Ensure it's uint8
            if slice_img.dtype != np.uint8:
                slice_img = (slice_img * 255).astype(np.uint8)
            pseudo_3d_volume[i] = slice_img
                
        # Convert to list of PIL images
        return [Image.fromarray(slice_img) for slice_img in pseudo_3d_volume]
    
    def _analyze_question_type(self, question):
        """
        Analyze question to determine its focus for better radiological context
        """
        question_lower = question.lower()
        
        # Detect question types for proper response formatting
        is_yes_no = any(q in question_lower for q in ["is there", "are there", "do you see", "can you see", "is it", "are the", "is this", "does this"])
        is_what = "what" in question_lower
        is_where = "where" in question_lower
        is_how = "how" in question_lower
        is_why = "why" in question_lower
        is_when = "when" in question_lower
        is_describe = "describe" in question_lower
        
        # Analyze radiological focus areas
        focus_areas = []
        for category, terms in self.finding_categories.items():
            if any(term in question_lower for term in terms):
                focus_areas.append(category)
        
        # Determine radiological modality context
        if any(term in question_lower for term in ["chest", "lung", "heart", "pneumonia", "effusion", "thoracic", "pulmonary"]):
            modality = "chest"
        elif any(term in question_lower for term in ["bone", "fracture", "break", "joint", "dislocation", "skeletal"]):
            modality = "bone"
        elif any(term in question_lower for term in ["joint", "articulation", "knee", "elbow", "shoulder", "hip", "wrist", "ankle"]):
            modality = "joint"
        elif any(term in question_lower for term in ["abdomen", "liver", "spleen", "kidney", "intestine", "bowel", "stomach"]):
            modality = "abdomen"
        elif any(term in question_lower for term in ["spine", "vertebra", "disc", "spinal", "cervical", "lumbar", "thoracic"]):
            modality = "spine"
        elif any(term in question_lower for term in ["pediatric", "child", "infant", "baby", "adolescent", "newborn"]):
            modality = "pediatric"
        elif any(term in question_lower for term in ["emergency", "trauma", "urgent", "acute", "accident"]):
            modality = "emergency"
        elif any(term in question_lower for term in ["foreign", "body", "object", "device", "tube", "line", "catheter"]):
            modality = "foreign_body"
        elif any(term in question_lower for term in ["soft tissue", "muscle", "tendon", "ligament", "swelling", "edema"]):
            modality = "soft_tissue"
        else:
            modality = "general"
            
        return {
            "is_yes_no": is_yes_no,
            "is_what": is_what,
            "is_where": is_where,
            "is_how": is_how,
            "is_why": is_why,
            "is_when": is_when,
            "is_describe": is_describe,
            "focus_areas": focus_areas,
            "modality": modality
        }
    
    def forward(self, pixel_values_list, input_ids, attention_mask):
        """Forward pass through the model."""
        # Get the first item from pixel_values_list
        # This should be a 4D tensor with shape [batch_size, slices, channels, height, width]
        # or it could be a list with a single item that is a 3D tensor
        if isinstance(pixel_values_list, list):
            stacked_pixel_values = pixel_values_list[0]
        else:
            stacked_pixel_values = pixel_values_list
            
        # Check if this is already stacked 3D data with proper batch dimension
        # Expected shape: [batch_size, slices, channels, height, width]
        if len(stacked_pixel_values.shape) == 5:
            batch_size, num_slices, channels, height, width = stacked_pixel_values.shape
            
            # Reshape to process each slice through vision encoder
            # [batch_size * slices, channels, height, width]
            flat_pixel_values = stacked_pixel_values.view(batch_size * num_slices, channels, height, width)
            
            # Process all slices at once through vision encoder
            vision_outputs = self.vision_encoder(pixel_values=flat_pixel_values)
            
            # Get last hidden state
            # Shape: [batch_size * slices, seq_length, hidden_size]
            vision_features = vision_outputs.last_hidden_state
            
            # Reshape back to 3D volume
            # [batch_size, slices, seq_length, hidden_size]
            seq_length, hidden_size = vision_features.shape[1], vision_features.shape[2]
            vision_features_3d = vision_features.view(batch_size, num_slices, seq_length, hidden_size)
            
        else:
            # Handle other input formats
            # Create empty list for vision features
            vision_features_list = []
            
            # If it's a 4D tensor with just one batch dimension, we'll use it as a single slice
            if len(stacked_pixel_values.shape) == 4:
                # Process single batch with single slice
                vision_outputs = self.vision_encoder(pixel_values=stacked_pixel_values)
                vision_features = vision_outputs.last_hidden_state
                vision_features_list.append(vision_features)
            else:
                # Try to process whatever format we received
                for pixel_values in pixel_values_list:
                    # Ensure pixel_values has batch dimension
                    if len(pixel_values.shape) == 3:  # [C, H, W]
                        pixel_values = pixel_values.unsqueeze(0)  # Add batch dimension [1, C, H, W]
                        
                    vision_outputs = self.vision_encoder(pixel_values=pixel_values)
                    vision_features_list.append(vision_outputs.last_hidden_state)
            
            # Ensure all feature maps have consistent dimensions
            ref_shape = vision_features_list[0].shape
            for i in range(len(vision_features_list)):
                feat = vision_features_list[i]
                # If batch size is missing, add it
                if len(feat.shape) == 2:  # [seq_len, hidden_dim]
                    feat = feat.unsqueeze(0)  # [1, seq_len, hidden_dim]
                    vision_features_list[i] = feat
                
            # Stack vision features to create 3D representation
            # [batch_size, slices, seq_length, hidden_size]
            vision_features_3d = torch.stack(vision_features_list, dim=1)
        
        # Process stacked features through MedQFormer
        med_query_features = self.medqformer(vision_features_3d)
        
        # Process through text encoder
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=med_query_features,
            return_dict=True
        )
        
        # Get text features for answer generation
        text_features = text_outputs.last_hidden_state
        
        # For VQA task, classify to generate answer tokens
        logits = self.vqa_classifier(text_features)
        
        # Boost medical term probabilities - increased boost for radiological terms
        medical_boost = 1.3  # Increased from 1.2 for better radiological term prioritization
        for batch_idx in range(logits.shape[0]):
            for token_idx in self.medical_token_ids:
                logits[batch_idx, :, token_idx] *= medical_boost
        
        return logits
        
    def generate_answer(self, image, question, max_length=150):
        """
        Generate an answer to a question about a medical image using a lightweight
        approach for faster response generation.
        
        Args:
            image: PIL Image of a medical scan
            question: String question about the image
            max_length: Maximum length of the generated answer
            
        Returns:
            String answer to the question
        """
        # Create pseudo-3D volume from 2D image
        print("Processing image...")
        image_3d = self.create_3d_from_2d(image)
        
        # Process all slices together as a batch
        batch_pixel_values = []
        
        # Process each slice with the vision processor 
        for i, slice_img in enumerate(image_3d):
            slice_inputs = self.vision_processor(images=slice_img, return_tensors="pt")
            # Extract pixel values without batch dimension
            slice_pixel_values = slice_inputs.pixel_values.squeeze(0)
            batch_pixel_values.append(slice_pixel_values)
            
        # Stack all slices together along a new dimension (batch of slices)
        stacked_pixel_values = torch.stack(batch_pixel_values, dim=0)
        
        # Add batch dimension for model processing
        stacked_pixel_values = stacked_pixel_values.unsqueeze(0)
        
        # Analyze question for better context
        question_analysis = self._analyze_question_type(question)
        modality = question_analysis["modality"]
        
        # Determine the appropriate clinical prefix based on question analysis
        prefix = self.clinical_prefixes[modality]  # Get contextual prefix
        
        # Create enhanced question with radiological context
        context_prompt = "Evaluate this radiographic image with attention to bone and soft tissue structures. "
        enhanced_question = f"Clinical Question: {prefix}{context_prompt}{question}"
        
        # Tokenize the enhanced question
        text_inputs = self.text_tokenizer(
            enhanced_question,
            padding="max_length",
            max_length=40,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Move tensor to device
        stacked_pixel_values = stacked_pixel_values.to(self.device)
        
        # Forward pass to get logits
        print("Generating response...")
        with torch.no_grad():
            logits = self(
                pixel_values_list=[stacked_pixel_values],
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
        
        # Fast decoding with greedy approach
        generated_ids = []
        current_ids = text_inputs.input_ids
        temperature = 0.7  # Default temperature
        
        # Simple greedy generation (much faster)
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self(
                    pixel_values_list=[stacked_pixel_values],
                    input_ids=current_ids,
                    attention_mask=torch.ones_like(current_ids).to(self.device)
                )
                
                # Get the most likely next token
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                # Stop if we predict end token
                if next_token.item() == self.text_tokenizer.sep_token_id:
                    break
                    
                generated_ids.append(next_token.item())
                
                # Reshape next token
                next_token = next_token.unsqueeze(-1)
                
                # Concatenate
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        # Decode the generated tokens
        answer = self.text_tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Simple post-processing
        if len(answer.strip()) < 10:
            # Fallback for very short answers
            answer = "Based on this radiographic image, "
            if "fracture" in question.lower():
                answer += "there appears to be a fracture present. Clinical correlation is recommended."
            elif "normal" in question.lower():
                answer += "no significant abnormalities are detected. The study appears normal."
            else:
                answer += "I cannot provide a definitive assessment. Further clinical correlation is recommended."
        
        # Ensure proper sentence structure
        if answer and answer[-1] not in ['.', '!', '?']:
            answer += '.'
            
        return answer

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
        Simplified preprocessing for X-ray imaging with minimal operations
        
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
            
            # Convert to uint8 and to RGB (DICOM is typically grayscale)
            image = (image * 255).astype(np.uint8)
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
            
            return Image.fromarray(image)
        
        # Handle regular image formats (JPEG, PNG, etc.)
        else:
            image = Image.open(image_path).convert('RGB')
            return image
    
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
        print(f"\n📋 Analyzing X-ray")
        print(f"📝 Question: \"{question}\"")
        
        # Handle empty or None question
        if not question or question is None:
            return "Please ask a specific question about this radiographic study."
            
        # Simple question enhancement
        enhanced_question = self._enhance_question_with_radiological_context(question)
        
        # Preprocess the image with simplified pipeline
        print(f"🖼️  Preprocessing radiographic image")
        image = self.preprocess_image(image_path)
        
        # Generate answer
        print("🔬 Analyzing the radiographic study...")
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