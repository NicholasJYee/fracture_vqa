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
- API timeouts: 
  - Connection timeout: 10 seconds for establishing connection
  - Read timeout: 120 seconds for model inference (increased for complex X-ray analyses)
  - URL fetch timeout: 30 seconds for downloading images from URLs

## Project Structure

- `app.py`: Main application with Gradio interface
- `api.py`: FastAPI backend for REST API usage
- `model.py`: Integration with Ollama API for LLaVA model
- `utils/`: Helper functions for image processing, etc.
- `examples/`: Example X-ray images for testing

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
- If you experience timeout errors:
  - For connection timeouts: Check your network connection and ensure Ollama is running
  - For read timeouts: Complex X-rays or larger models may need more time. You can increase the timeout in `model.py` by changing the `read_timeout` value in the `OllamaLLaVAModel` class.
  - For very large models: Consider using a smaller model or increasing your system resources

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
