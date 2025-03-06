<<<<<<< HEAD
# X-Ray Visual Question Answering Chatbot

A Python-based chatbot that can answer questions about X-ray medical images using Ollama API with LLaVA model.

## Features

- Upload X-ray images in various formats (DICOM, JPG, PNG)
- Ask questions about the uploaded X-ray images
- Get AI-generated answers using LLaVA through Ollama API
- Simple and intuitive user interface
- Locally hosted AI for privacy and security

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running locally
- LLaVA model pulled in Ollama (`ollama pull llava`)
- Python 3.8+ installed

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables (optional):
   Create a `.env` file based on `.env.example` to configure the Ollama URL

## Ollama Setup

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the LLaVA model:
   ```
   ollama pull llava
   ```
   Alternatively, you can use a larger model for better results:
   ```
   ollama pull llava:13b
   ```
3. Ensure Ollama is running (it typically starts automatically and listens on port 11434)
4. Verify Ollama is running with LLaVA available:
   ```
   ollama list
   ```

## Usage

### Web Interface (Recommended)

Run the web interface using:

```
python app.py
```

Or with Streamlit:

```
streamlit run streamlit_app.py
```

### API-only Mode

For headless operation or backend integration:

```
python api.py
```

## Model Information

This application uses LLaVA (Large Language and Vision Assistant) through the Ollama API:

- **LLaVA**: A powerful multimodal model that can understand both text and images
- **Ollama**: Provides a lightweight local API to serve the LLaVA model
- **No Cloud Dependencies**: All inference happens locally for privacy and no API costs

## Configuration Options

- `OLLAMA_URL`: Set in .env file to configure the Ollama API URL (default: http://localhost:11434)
- Device selection: Automatically uses CUDA/MPS if available, falls back to CPU

## Project Structure

- `app.py`: Main application with Gradio interface
- `api.py`: FastAPI backend for REST API usage
- `model.py`: Integration with Ollama API for LLaVA model
- `utils/`: Helper functions for image processing, etc.
- `examples/`: Example X-ray images for testing
- `.env`: Environment variables configuration (Hugging Face token, etc.)

## Accessing Gated Datasets

To work with gated datasets from Hugging Face Hub:

1. Create a Hugging Face account if you don't have one
2. Generate an access token from your [Hugging Face settings](https://huggingface.co/settings/tokens)
3. Add the token to your `.env` file:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```
4. The application will automatically use this token when accessing models or datasets

## Limitations

- Performance depends on your local hardware (GPU recommended)
- The model's quality depends on the LLaVA version you pull from Ollama
- Not intended to replace professional medical diagnosis
- First request may be slow as the model loads into memory

## Troubleshooting

- If you encounter errors connecting to Ollama, make sure it's running with `ps aux | grep ollama`
- If LLaVA model isn't working, try pulling a specific version: `ollama pull llava:13b`
- For better quality on medical imaging, you may want to try a medically fine-tuned model if available
- If you're getting "model not found" errors, check that you've successfully pulled the model with `ollama list` 
=======
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
>>>>>>> 451b279b8e5e35715ab9a11e4a3eb284180992c1
