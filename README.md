# X-Ray Visual Question Answering Chatbot

A Python-based chatbot that can answer questions about X-ray medical images using advanced deep learning techniques.

## Features

- Upload X-ray images in various formats (DICOM, JPG, PNG)
- Ask questions about the uploaded X-ray images
- Get AI-generated answers using a specialized medical VQA model
- Simple and intuitive user interface

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Hugging Face API token to access gated datasets:
     ```
     HUGGINGFACE_TOKEN=your_huggingface_token_here
     ```
   - You can get your token from [Hugging Face settings page](https://huggingface.co/settings/tokens)

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

This application uses a fine-tuned vision-language model specifically adapted for medical imaging. The model combines:

- Image encoder: Pre-trained on medical imaging datasets
- Text encoder: Specialized for medical terminology
- Cross-modal fusion: Correlates visual features with textual queries

## Project Structure

- `app.py`: Main application with Gradio interface
- `streamlit_app.py`: Alternative Streamlit interface
- `model.py`: VQA model implementation
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

- The model is optimized for chest X-rays but may work with other X-ray types
- Performance varies based on image quality and question specificity
- Not intended to replace professional medical diagnosis 