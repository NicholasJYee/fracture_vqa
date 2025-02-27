import streamlit as st
import os
import time
from PIL import Image
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from model import XrayVQAModel
from utils import save_upload, enhance_xray_for_display, read_dicom_tags, create_directory_if_not_exists

# Initialize directories
create_directory_if_not_exists("uploads")
create_directory_if_not_exists("examples")
create_directory_if_not_exists("temp")

# Page configuration
st.set_page_config(
    page_title="X-Ray VQA Chatbot",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e6f7ff;
    }
    .chat-message.bot {
        background-color: #f3f4f6;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 4px;
        width: 100%;
    }
    .stImage {
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTextInput div[data-baseweb="input"] {
        border-radius: 4px;
        border: 1px solid #ddd;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = XrayVQAModel()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if 'image_enhanced' not in st.session_state:
    st.session_state.image_enhanced = False

# Helper functions
def display_chat_history():
    for i, (role, message) in enumerate(st.session_state.chat_history):
        if role == "user":
            if isinstance(message, str):  # Text message
                st.markdown(f'<div class="chat-message user"><div class="message">👤 <b>You:</b> {message}</div></div>', unsafe_allow_html=True)
            else:  # Image message
                st.markdown(f'<div class="chat-message user"><div class="message">👤 <b>You:</b> Uploaded an X-ray image</div></div>', unsafe_allow_html=True)
        else:  # bot message
            st.markdown(f'<div class="chat-message bot"><div class="message">🤖 <b>Chatbot:</b> {message}</div></div>', unsafe_allow_html=True)

def process_file_upload(uploaded_file):
    """Process uploaded file and update session state."""
    if uploaded_file is not None:
        # Save the file to disk
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.current_image = file_path
        st.session_state.image_enhanced = False
        
        # Clear previous chat history when a new image is uploaded
        st.session_state.chat_history = []
        
        # Add image message to chat
        st.session_state.chat_history.append(("user", file_path))
        st.session_state.chat_history.append(("bot", "I've analyzed your X-ray image. What would you like to know about it?"))
        
        return True
    return False

def display_image(image_path, use_enhanced=False):
    """Display the current image in the UI."""
    if image_path and os.path.exists(image_path):
        try:
            if use_enhanced and not st.session_state.image_enhanced:
                # Enhance the image for better viewing
                enhanced_path = enhance_xray_for_display(image_path)
                st.session_state.current_image = enhanced_path
                st.session_state.image_enhanced = True
                image_path = enhanced_path
            
            # Handle DICOM format
            if image_path.lower().endswith('.dcm'):
                dicom = pydicom.dcmread(image_path)
                img_array = dicom.pixel_array
                
                # Normalize and convert to 8-bit
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                
                # Create a temporary file
                temp_path = os.path.join("temp", f"dicom_view_{int(time.time())}.png")
                
                # Save as PNG
                Image.fromarray(img_array).save(temp_path)
                image_path = temp_path
            
            image = Image.open(image_path)
            st.image(image, caption="Current X-ray Image", use_column_width=True)
            
            # Display image info
            display_image_info(image_path)
            
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
    else:
        st.info("No image uploaded yet. Please upload an X-ray image.")

def display_image_info(image_path):
    """Display information about the image."""
    if image_path.lower().endswith('.dcm'):
        # DICOM file
        try:
            tags = read_dicom_tags(image_path)
            st.markdown("### DICOM Image Information")
            for key, value in tags.items():
                st.write(f"**{key}:** {value}")
        except Exception as e:
            st.error(f"Error reading DICOM tags: {str(e)}")
    else:
        # Regular image file
        try:
            img = Image.open(image_path)
            st.markdown("### Image Information")
            st.write(f"**Format:** {img.format}")
            st.write(f"**Size:** {img.width} x {img.height}")
            st.write(f"**Mode:** {img.mode}")
        except Exception as e:
            st.error(f"Error reading image: {str(e)}")

def process_user_question(question):
    """Process user question and generate answer."""
    if not st.session_state.current_image:
        st.warning("Please upload an X-ray image first.")
        return
    
    # Add user question to chat history
    st.session_state.chat_history.append(("user", question))
    
    # Show thinking message temporarily
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(f'<div class="chat-message bot"><div class="message">🤖 <b>Chatbot:</b> Analyzing the X-ray...</div></div>', unsafe_allow_html=True)
    
    try:
        # Process the image and answer the question
        answer = st.session_state.model.answer_question(
            st.session_state.current_image, question
        )
        
        # Add bot answer to chat history
        st.session_state.chat_history.append(("bot", answer))
        
        # Remove thinking message
        thinking_placeholder.empty()
        
    except Exception as e:
        # Add error message to chat history
        st.session_state.chat_history.append(("bot", f"Error analyzing the image: {str(e)}"))
        thinking_placeholder.empty()

# Main UI layout
st.title("X-Ray Visual Question Answering Chatbot")
st.markdown("Upload an X-ray image and ask questions about it. The AI will analyze the image and provide answers.")

# Create two columns
col1, col2 = st.columns([3, 2])

with col1:
    # Chat interface
    st.markdown("### Chat")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        display_chat_history()
    
    # User input
    user_question = st.text_input("Ask a question about the X-ray", key="user_question")
    
    # Submit button for user question
    if st.button("Send Question"):
        if user_question:
            process_user_question(user_question)
            st.session_state.user_question = ""  # Clear input
            st.experimental_rerun()

with col2:
    # Image upload and display
    st.markdown("### X-ray Image")
    
    uploaded_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg", "dcm"])
    if uploaded_file is not None:
        if process_file_upload(uploaded_file):
            st.experimental_rerun()
    
    # Display current image
    if st.session_state.current_image:
        # Toggle for image enhancement
        enhance_image = st.checkbox("Enhance image contrast", value=st.session_state.image_enhanced)
        if enhance_image != st.session_state.image_enhanced:
            # User toggled enhancement
            display_image(st.session_state.current_image, use_enhanced=enhance_image)
            st.experimental_rerun()
        else:
            display_image(st.session_state.current_image, use_enhanced=enhance_image)
    
    # Example images section
    st.markdown("### Example X-rays")
    st.markdown("No examples available yet. Upload your own X-ray image.")

# Sample questions section
st.markdown("## Sample Questions to Ask")
st.markdown("""
- Is there any evidence of pneumonia in this chest X-ray?
- Do you see any fractures or breaks in the bones?
- Are there any abnormal masses or nodules visible?
- Is there pleural effusion present?
- Does this X-ray show signs of cardiomegaly (enlarged heart)?
- Are the lungs clear or is there congestion?
- Is there any sign of atelectasis (collapsed lung tissue)?
- Does this X-ray appear normal or are there concerning findings?
""")

# Disclaimer
st.markdown("### Disclaimer")
st.markdown("""
This AI assistant is not a replacement for professional medical advice. Always consult with a qualified healthcare provider for proper diagnosis. The AI model has limitations and may not detect all conditions.
""")

# Footer
st.markdown('<div class="footer">X-Ray VQA Chatbot - Powered by AI</div>', unsafe_allow_html=True) 