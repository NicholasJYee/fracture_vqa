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