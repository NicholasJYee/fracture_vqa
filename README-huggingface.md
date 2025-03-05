# Fracture VQA - X-ray Visual Question Answering

This application allows you to upload X-ray images and ask questions about them. The AI will analyze the images and provide responses based on its understanding of the medical context.

## Features

- Upload DICOM (.dcm) or common image formats (.jpg, .png, etc.)
- Ask questions about the uploaded X-ray images
- Get AI-powered analysis and responses
- Enhance contrast of uploaded images for better visibility
- Browse previously uploaded images

## Performance

This application will use:
- CUDA GPU acceleration if available (recommended for best performance)
- Apple Metal/MPS for Mac users with M1/M2/M3 chips
- CPU as fallback if no GPU acceleration is available

## External API Configuration

For optimal performance, you can connect this Space to an external Ollama server by setting the `OLLAMA_URL` environment variable in the Space settings.

## Usage

1. Upload an X-ray image using the upload button
2. Enter your question in the text box
3. View the AI's response and the processed image
4. Optionally, enhance the image using the enhance button

## Technical Details

This application uses:
- LLaVA (Large Language and Vision Assistant) model for X-ray analysis
- Gradio for the user interface
- PyTorch with CUDA support for GPU acceleration
- FastAPI for the backend API

## Known Limitations

- The model is trained on a specific dataset and may not recognize all types of fractures or medical conditions
- Response quality depends on image quality and clarity
- Processing time may vary based on available hardware acceleration 