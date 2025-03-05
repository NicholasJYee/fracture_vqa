---
title: Fracture VQA - X-ray Visual Question Answering
emoji: 🩻
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
app_port: 7860
app_file: app-huggingface.py
---

# Fracture Visual Question Answering

A system for conducting visual question answering on radiological images (X-rays).

## Setup

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) for LLaVA model serving
- (Optional) CUDA-compatible GPU for accelerated performance

### Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Make sure Ollama is installed and running
4. Run the application: `python app.py`

## Usage

1. Upload an X-ray image (DICOM or common image formats)
2. Ask a question about the image
3. The system will analyze the image and provide a response

## Configuration

Configure the system by modifying the `.env` file:

```
OLLAMA_URL=http://localhost:11434
HF_API_URL=http://localhost:8000
```

## Troubleshooting

### Timeout errors when downloading images

If you encounter timeout errors when downloading images from URLs, you can adjust the timeout settings in the `load_from_url` method in `app.py`:

- Default timeout is set to 10 seconds
- For slow connections, consider increasing this value
- Example: `response = requests.get(url, timeout=20)`  # Increased to 20 seconds

### CUDA/GPU Acceleration

This application will automatically detect and use:
- CUDA if available (recommended for best performance)
- Apple Metal/MPS for Mac users with M1/M2/M3 chips
- CPU as fallback if no GPU acceleration is available

To check which device is being used, look for the log message at startup:
```
Device configuration: Using device=[device_name]
```

## Hugging Face Spaces Deployment

For deploying on Hugging Face Spaces, use these special files:
- `app-huggingface.py` - Optimized Gradio interface for Spaces
- `api-huggingface.py` - API backend for Spaces
- `requirements-huggingface.txt` - Dependencies for Spaces
- `Dockerfile` - Container configuration for Spaces

To deploy:
1. Create a new Space on Hugging Face
2. Select Dockerfile as the Space type
3. Upload all files to the Space
4. (Optional) Configure environment variables for external Ollama server

For detailed instructions, see the [README-huggingface.md](README-huggingface.md) file.

## API Reference

The system provides a REST API for integration:

- `GET /status` - Check API status
- `POST /upload` - Upload an X-ray image
- `POST /ask` - Ask a question about an uploaded image
- `GET /images/{filename}` - Retrieve a specific image
- `GET /images` - List all uploaded images
- `DELETE /images/{filename}` - Delete a specific image
- `POST /enhance` - Enhance the contrast of an image 