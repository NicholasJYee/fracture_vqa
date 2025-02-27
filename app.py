import os
import gradio as gr
import tempfile
from model import XrayVQAModel
from utils import save_upload, enhance_xray_for_display, read_dicom_tags, create_directory_if_not_exists
import base64
from PIL import Image
import pydicom
import numpy as np

# Initialize directories
create_directory_if_not_exists("uploads")
create_directory_if_not_exists("examples")

# Initialize the model
model = XrayVQAModel()

# Keep chat history
chat_history = []

def process_example_image(example_path):
    """Process an example image and return its path."""
    if not os.path.exists(example_path):
        return None
    return example_path

def add_text(history, text):
    """Add user text to chat history."""
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

def add_file(history, file):
    """Add uploaded file to chat history."""
    file_path = save_upload(file)
    history = history + [((file_path,), None)]
    return history

def bot(history, image_path=None):
    """Generate bot response based on the last query and image."""
    if image_path is None and len(history) > 0 and isinstance(history[-1][0], tuple):
        # Get image from history if not provided directly
        image_path = history[-1][0][0]
    
    if not image_path or image_path is None:
        # No image available
        response = "Please upload an X-ray image first."
        history[-1][1] = response
        return history
    
    if len(history) == 0 or history[-1][1] is not None:
        # No question or already answered
        response = "I'm ready to answer questions about this X-ray. What would you like to know?"
        # Add a dummy entry in history
        if len(history) == 0 or isinstance(history[-1][0], tuple):
            history = history + [("", response)]
        else:
            history[-1][1] = response
        return history
    
    # Get the user's question
    question = history[-1][0]
    
    try:
        # Process the image and answer the question
        answer = model.answer_question(image_path, question)
        history[-1][1] = answer
    except Exception as e:
        history[-1][1] = f"Error processing the image: {str(e)}"
    
    return history

def upload_file(files, history):
    """Handle file upload event."""
    if not files:
        return history, None
    
    file_path = save_upload(files[0])
    
    # Update history with the new image
    if history is None:
        history = []
    history = history + [((file_path,), None)]
    
    # Let the bot respond with initial message
    history = bot(history, file_path)
    
    return history, file_path

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
    for user_msg, bot_msg in raw_history:
        if isinstance(user_msg, tuple):  # Image upload
            formatted.append(("📷 Uploaded X-ray image", bot_msg))
        else:  # Text message
            formatted.append((user_msg, bot_msg))
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
with gr.Blocks(css="footer {visibility: hidden}") as interface:
    gr.Markdown("# X-Ray Visual Question Answering Chatbot")
    gr.Markdown("Upload an X-ray image and ask questions about it. The AI will analyze the image and provide answers.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                height=600,
                avatar_images=("👤", "🤖"),
                bubble_full_width=False,
            )
            with gr.Row():
                txt = gr.Textbox(
                    scale=5,
                    show_label=False,
                    placeholder="Ask a question about the X-ray...",
                    container=False,
                )
                btn = gr.UploadButton("📷", file_types=["image", ".dcm"])
        
        with gr.Column(scale=1):
            image_output = gr.Image(type="filepath", label="Current X-ray Image")
            image_info = gr.Markdown("Upload an X-ray to see information")
            
            example_files = [
                os.path.join(os.path.dirname(__file__), "examples", f) 
                for f in ["normal_chest.jpg", "pneumonia_example.jpg", "fracture_xray.jpg"] 
                if os.path.exists(os.path.join(os.path.dirname(__file__), "examples", f))
            ]
            
            if example_files:
                gr.Markdown("### Example X-rays")
                gr.Examples(
                    examples=example_files,
                    inputs=btn,
                    outputs=[chatbot, image_output],
                    fn=upload_file,
                    cache_examples=True,
                )
    
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    
    file_msg = btn.upload(upload_file, [btn, chatbot], [chatbot, image_output], queue=True)
    file_msg.then(get_image_info, [image_output], [image_info])
    
    gr.Markdown("## Sample Questions to Ask")
    gr.Markdown("""
    - Is there any evidence of pneumonia in this chest X-ray?
    - Do you see any fractures or breaks in the bones?
    - Are there any abnormal masses or nodules visible?
    - Is there pleural effusion present?
    - Does this X-ray show signs of cardiomegaly (enlarged heart)?
    - Are the lungs clear or is there congestion?
    - Is there any sign of atelectasis (collapsed lung tissue)?
    - Does this X-ray appear normal or are there concerning findings?
    """)
    
    gr.Markdown("### Note")
    gr.Markdown("""
    - This AI assistant is not a replacement for professional medical advice.
    - Always consult with a qualified healthcare provider for proper diagnosis.
    - The AI model has limitations and may not detect all conditions.
    """)

# Create examples directory and download example files if needed
create_directory_if_not_exists("examples")

# Run the interface
if __name__ == "__main__":
    interface.launch(share=True) 