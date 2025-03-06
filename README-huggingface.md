# Fracture VQA - X-ray Visual Question Answering

This application allows medical professionals and researchers to upload X-ray images and ask questions about them. The AI will analyze the images and provide detailed responses based on its understanding of radiological context.

## Features

- Upload DICOM (.dcm) or common image formats (.jpg, .png, etc.)
- Ask questions about uploaded X-ray images (fractures, positioning, abnormalities)
- Get AI-powered analysis and responses based on radiological knowledge
- Enhance contrast of uploaded images for better visibility
- Browse previously uploaded images with persistent storage

## Hardware Acceleration

This application automatically detects and uses the best available hardware:

- **CUDA GPU** acceleration if available (recommended for best performance)
- **Apple Metal/MPS** for Mac users with M1/M2/M3 chips
- **CPU** as fallback if no GPU acceleration is available

The system will display which device is being used on startup.

## API Integration

The application provides a REST API that you can use to integrate with other systems:

- Upload images programmatically
- Ask questions through API calls
- Retrieve enhanced images and analysis results

## Usage Instructions

1. **Upload an X-ray image** using the upload button
2. **Enter your question** in the text box (e.g., "Is there a fracture in this X-ray?")
3. **View the AI's response** and the processed image
4. **Enhance the image** using the enhance button for better visibility if needed

## External Ollama Configuration (Optional)

For optimal performance, you can connect this Space to an external Ollama server:

1. Set up an Ollama server with the LLaVA model
2. Configure the `OLLAMA_URL` environment variable in the Space settings
3. Restart the Space to use your external Ollama server

## Technical Details

This application uses:

- **LLaVA** (Large Language and Vision Assistant) model for X-ray analysis
- **Gradio** for the user interface
- **PyTorch** with CUDA support for GPU acceleration
- **FastAPI** for the backend API
- **DICOM** processing for medical imaging

## Privacy & Security

- All image processing is done within this Space
- No data is sent to external services (unless you configure an external Ollama server)
- Images can be deleted after analysis if needed

## Known Limitations

- The model is trained on general medical datasets and may not recognize all types of fractures or medical conditions
- Response quality depends on image quality, positioning, and clarity
- Processing time may vary based on available hardware acceleration and image complexity
- Not intended as a replacement for professional medical diagnosis 