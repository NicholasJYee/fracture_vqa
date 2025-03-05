import os
import gradio as gr
import tempfile
import torch
from model import XrayVQAModel
from utils import save_upload, enhance_xray_for_display, read_dicom_tags, create_directory_if_not_exists
import base64
from PIL import Image
import pydicom
import numpy as np
import traceback
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# In Hugging Face Spaces, we'll use environment variables to configure Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
HF_API_URL = os.getenv("HF_API_URL", None)  # For alternative API usage

# Check for CUDA or other accelerators
print("\n" + "="*50)
print("DEVICE CONFIGURATION")
print("="*50)

if torch.cuda.is_available():
    # Get CUDA device details
    cuda_device_count = torch.cuda.device_count()
    cuda_device_name = torch.cuda.get_device_name(0)
    print(f"✅ CUDA available: {cuda_device_count} device(s)")
    print(f"✅ Using CUDA device: {cuda_device_name}")
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = "cuda"
# Check for MPS (Metal Performance Shaders for Mac)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.backends.mps.is_built = True
    print("✅ MPS (Metal Performance Shaders) is available and enabled")
    device = "mps"
else:
    print("ℹ️ No GPU acceleration available, using CPU")
    device = "cpu"

print(f"Selected device: {device}")
print("="*50 + "\n")

# Initialize directories
create_directory_if_not_exists("uploads")
create_directory_if_not_exists("temp")

# Initialize the model - with error handling for Hugging Face environment
try:
    # Pass the device explicitly to the model initialization
    model = XrayVQAModel(ollama_url=OLLAMA_URL)
    
    # Log device info
    print(f"Model device: {model.device}")
    print(f"Using CUDA for model: {str(model.device).startswith('cuda')}")
    
    USING_OLLAMA = True
    print(f"Using Ollama API at: {OLLAMA_URL}")
except Exception as e:
    print(f"Error initializing Ollama model: {str(e)}")
    print("Please verify that Ollama is properly set up and running.")
    print("For Hugging Face Spaces, you may need to configure an external Ollama server.")
    USING_OLLAMA = False
    print("Continuing without active model...")

# Keep chat history
chat_history = []

def process_example_image(example_path):
    """Process an example image and return its path."""
    if not os.path.exists(example_path):
        return None
    return example_path

def add_text(history, text):
    """Add user text to chat history."""
    # Add user message with role and content keys
    history = history + [{"role": "user", "content": text}]
    return history, gr.Textbox(value="", interactive=False)

def add_file(history, file):
    """Add uploaded file to chat history."""
    file_path = save_upload(file)
    history = history + [{"role": "user", "content": {"image": file_path, "text": "Uploaded X-ray image"}}]
    return history

def bot(history, image_path=None):
    """Generate bot response based on the last query and image."""
    # Create a copy of history to avoid modifying in-place
    history_copy = list(history)
    
    # Extract image path from history if not provided directly
    if not image_path:
        for message in reversed(history_copy):
            if message["role"] == "user" and isinstance(message["content"], dict) and "image" in message["content"]:
                image_path = message["content"]["image"]
                break
    
    # Find the last user query
    last_query = None
    for message in reversed(history_copy):
        if message["role"] == "user" and isinstance(message["content"], str):
            last_query = message["content"]
            break
    
    # Check if we have both an image and a query
    if not image_path or not last_query:
        response = "Please upload an X-ray image and ask a question about it."
    else:
        try:
            if USING_OLLAMA:
                # Use the Ollama-based model
                response = model.answer_question(image_path, last_query)
            elif HF_API_URL:
                # Use alternative API if available
                response = "Using external API for inference (Ollama not detected)"
                # Add code here to call alternative API
            else:
                response = "⚠️ Ollama service is not available. Please make sure Ollama is running with LLaVA model installed."
                response += "\n\nTo set up Ollama:\n1. Install Ollama from ollama.ai\n2. Pull the LLaVA model: 'ollama pull llava'\n3. Ensure Ollama is running"
        except Exception as e:
            print(traceback.format_exc())
            response = f"Error generating response: {str(e)}"
    
    # Add the bot response to history
    history_copy.append({"role": "assistant", "content": response})
    
    return history_copy, gr.Textbox(interactive=True)

def upload_file(files, history):
    """Process uploaded file and add to chat history."""
    # Ensure files is a list (even for single file upload)
    if not isinstance(files, list):
        files = [files]
    
    processed_history = history.copy() if history else []
    last_file_path = None
    
    # Process each uploaded file
    for file in files:
        if file is None:
            continue
        
        try:
            # Save the uploaded file
            file_path = save_upload(file)
            last_file_path = file_path
            
            # Add to chat history
            processed_history.append({
                "role": "user", 
                "content": {
                    "image": file_path, 
                    "text": f"Uploaded X-ray image: {os.path.basename(file_path)}"
                }
            })
        except Exception as e:
            print(f"Error processing upload: {str(e)}")
            processed_history.append({
                "role": "system",
                "content": f"Error processing uploaded file: {str(e)}"
            })
    
    return processed_history, last_file_path

def display_image(file_path):
    """Process and display the uploaded image."""
    if not file_path or not os.path.exists(file_path):
        return None

    try:
        # For DICOM files, convert to PNG for displaying
        if file_path.lower().endswith('.dcm'):
            # Read DICOM file
            dicom = pydicom.dcmread(file_path)
            # Convert to array and normalize
            image = dicom.pixel_array.astype(float)
            image = (image - image.min()) / (image.max() - image.min())
            image = (image * 255).astype(np.uint8)
            
            # Convert grayscale to RGB if needed
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
                
            # Save as temporary PNG for display
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                Image.fromarray(image).save(temp.name)
                return temp.name
        else:
            # For regular image formats, return as is
            return file_path
    except Exception as e:
        print(f"Error displaying image: {str(e)}")
        return None

def format_history(raw_history):
    """Format chat history for display."""
    formatted = []
    for message in raw_history:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "user":
            if isinstance(content, str):
                formatted.append((content, None))
            elif isinstance(content, dict) and "text" in content:
                formatted.append((content["text"], None))
        elif role == "assistant":
            formatted.append((None, content))
    
    return formatted

def get_image_info(file_path):
    """Get information about the X-ray image for display."""
    if not file_path or not os.path.exists(file_path):
        return "No image uploaded"
        
    try:
        if file_path.lower().endswith('.dcm'):
            # Extract DICOM metadata
            tags = read_dicom_tags(file_path)
            # Format the tags
            info = "## DICOM Image Information\n\n"
            info += "\n".join([f"**{key}**: {value}" for key, value in tags.items()])
            return info
        else:
            # For regular images
            img = Image.open(file_path)
            info = f"## Image Information\n\n"
            info += f"**Format**: {img.format}\n"
            info += f"**Size**: {img.width} x {img.height} pixels\n"
            info += f"**Mode**: {img.mode}\n"
            info += f"**File**: {os.path.basename(file_path)}"
            return info
    except Exception as e:
        return f"Error getting image info: {str(e)}"

# Build the Gradio interface
with gr.Blocks(css="footer {visibility: hidden}") as interface:
    gr.Markdown("""
    # X-ray Visual Question Answering
    
    Upload an X-ray image and ask questions about it. The model will analyze the image and answer your questions.
    
    **Note**: This demo uses the Ollama API with LLaVA model for inference. Make sure Ollama is running with the LLaVA model installed.
    """)
    
    with gr.Row():
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                avatar_images=["user-icon.png", "assistant-icon.png"],
                height=600,
                show_label=False,
                layout="panel"
            )
            txt = gr.Textbox(
                placeholder="Ask a question about the X-ray image...",
                scale=7,
                container=False
            )
            
            with gr.Row():
                btn = gr.UploadButton("📁 Upload X-ray", file_types=["image", ".dcm"])
                
        with gr.Column(scale=3):
            image_output = gr.Image(type="filepath", label="X-ray Image")
            image_info = gr.Markdown("Upload an X-ray to see details")
    
    gr.Examples(
        examples=[
            ["example/calcaneal-fracture.jpeg", "Is there a fracture in this X-ray? If so, where?"],
            ["example/normal-chest.jpg", "Is this chest X-ray normal?"]
        ],
        inputs=[image_output, txt],
        fn=process_example_image,
        cache_examples=True,
    )
    
    # Set up event handlers
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False)
    txt_msg.then(bot, [chatbot, image_output], [chatbot, txt], queue=True)
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    
    # Change from btn.upload to btn.change
    file_msg = btn.change(upload_file, [btn, chatbot], [chatbot, image_output], queue=True)
    file_msg.then(get_image_info, [image_output], [image_info])
    
    gr.Markdown("""
    *Note: This is not a replacement for professional medical diagnosis. Consult with a healthcare provider for accurate medical advice.*
    """)

# Launch the interface with settings for Hugging Face Spaces
interface.launch(
    share=True,
    server_name="0.0.0.0",  # Important for Hugging Face Spaces
    server_port=7860,        # Default port for Hugging Face Spaces
    debug=False,             # Disable debug for production
    enable_queue=True        # Enable queue for handling multiple requests
) 