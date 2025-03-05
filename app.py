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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Ollama URL from environment variable or use default
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Enable MPS (Metal Performance Shaders) for macOS if available
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.backends.mps.is_built = True
    print("MPS (Metal Performance Shaders) is available and enabled")

# Initialize directories
create_directory_if_not_exists("uploads")

# Initialize the model with Ollama URL
model = XrayVQAModel(ollama_url=OLLAMA_URL)

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
    if image_path is None and len(history_copy) > 0:
        # Look for an image in the history
        for message in reversed(history_copy):
            if message["role"] == "user" and isinstance(message["content"], str):
                # Check if the message contains an image path reference
                if "📷 Uploaded X-ray image:" in message["content"]:
                    # Extract the path from the message
                    parts = message["content"].split(":", 1)
                    if len(parts) > 1:
                        image_path = parts[1].strip()
                    break
    
    if not image_path or image_path is None:
        # No image available
        response = "Please upload an X-ray image first."
        history_copy.append({"role": "assistant", "content": response})
        return history_copy
    
    # If the last message is from the assistant or we have no history, prompt for a question
    if len(history_copy) == 0 or history_copy[-1]["role"] == "assistant":
        response = "I'm ready to answer questions about this X-ray. What would you like to know?"
        history_copy.append({"role": "assistant", "content": response})
        return history_copy
    
    # Get the user's question - ensure it's a string
    if history_copy[-1]["role"] == "user":
        user_content = history_copy[-1]["content"]
        # If the message contains an image upload notification, skip processing
        if isinstance(user_content, str) and "📷 Uploaded X-ray image:" in user_content:
            response = "I'm ready to answer questions about this X-ray. What would you like to know?"
            history_copy.append({"role": "assistant", "content": response})
            return history_copy
            
        question = user_content if isinstance(user_content, str) else ""
        
        try:
            # Process the image and answer the question
            answer = model.answer_question(image_path, question)
            history_copy.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"Error processing the image: {str(e)}"
            print(f"Error in bot function: {error_msg}")
            print(traceback.format_exc())
            history_copy.append({"role": "assistant", "content": error_msg})
    
    return history_copy

def upload_file(files, history):
    """Handle file upload event."""
    try:
        print(f"Upload received: {type(files)}")
        
        # Initialize history if None
        if history is None:
            history = []
            
        # Make copy of history to avoid modifying in-place
        history_copy = list(history)
        
        # Verify we have files
        if not files:
            print("No files received")
            return history_copy, None
            
        if isinstance(files, list) and len(files) == 0:
            print("Empty file list received")
            return history_copy, None
        
        # Debug file object
        print(f"File object structure: {files}")
        if isinstance(files, list):
            print(f"First file type: {type(files[0])}")
        
        # Save the uploaded file
        file_path = save_upload(files)
        
        if not file_path or not os.path.exists(file_path):
            print(f"File path invalid or does not exist: {file_path}")
            return history_copy, None
        
        # Add user message with image content - format as a string for messages format
        # In messages format, we can't use a dictionary for content
        history_copy.append({
            "role": "user", 
            "content": f"📷 Uploaded X-ray image: {file_path}"
        })
        
        # Let the bot respond with initial message
        history_copy = bot(history_copy, file_path)
        
        return history_copy, file_path
        
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        print(traceback.format_exc())
        # Return the history unchanged if there's an error
        return history if history is not None else [], None

def display_image(file_path):
    """Display an image on the UI."""
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        # For DICOM files, convert to viewable format
        if file_path.lower().endswith('.dcm'):
            dicom = pydicom.dcmread(file_path)
            img_array = dicom.pixel_array
            
            # Normalize and convert to 8-bit
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            # Create a temporary PNG file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
                temp_path = temp.name
            
            # Save as PNG
            Image.fromarray(img_array).save(temp_path)
            return temp_path
        
        return file_path
    except Exception as e:
        print(f"Error displaying image: {str(e)}")
        return None

def format_history(raw_history):
    """Format chat history for display."""
    formatted = []
    for message in raw_history:
        if message["role"] == "user" and isinstance(message["content"], dict) and "image" in message["content"]:
            formatted.append(("📷 Uploaded X-ray image", message["content"]["text"]))
        else:
            formatted.append((message["content"], None))
    return formatted

def get_image_info(file_path):
    """Get information about the image file."""
    if not file_path or not os.path.exists(file_path):
        return "No image available"
    
    if file_path.lower().endswith('.dcm'):
        # DICOM file
        tags = read_dicom_tags(file_path)
        info = "## DICOM Image Information\n\n"
        for key, value in tags.items():
            info += f"**{key}:** {value}\n"
        return info
    else:
        # Regular image file
        try:
            img = Image.open(file_path)
            return f"## Image Information\n\n**Format:** {img.format}\n**Size:** {img.width} x {img.height}\n**Mode:** {img.mode}"
        except Exception as e:
            return f"Error reading image: {str(e)}"

# Create the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# X-Ray Visual Question Answering Chatbot")
    gr.Markdown("Upload an X-ray image and ask questions about it. The AI will analyze the image and provide answers.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                height=600,
                bubble_full_width=False,
                type="messages"  # Use newer 'messages' format
            )
            with gr.Row():
                txt = gr.Textbox(
                    scale=5,
                    show_label=False,
                    placeholder="Ask a question about the X-ray...",
                    container=False,
                )
                # Using File component with filepath type
                btn = gr.File(
                    label="Upload X-ray Image",
                    file_types=["image", ".dcm"],
                    type="filepath"
                )
        
        with gr.Column(scale=1):
            # Display the image but don't allow interaction with it
            image_output = gr.Image(type="filepath", label="Current X-ray Image", interactive=False)
            image_info = gr.Markdown("Upload an X-ray to see information")
            
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    
    # Change from btn.upload to btn.change
    file_msg = btn.change(upload_file, [btn, chatbot], [chatbot, image_output], queue=True)
    file_msg.then(get_image_info, [image_output], [image_info])
    
    gr.Markdown("""
    Made by Nicholas J. Yee.
    """)

# Run the interface
if __name__ == "__main__":
    interface.launch(debug=True)