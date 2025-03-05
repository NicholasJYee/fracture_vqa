FROM python:3.10-slim

WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements-huggingface.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-huggingface.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p uploads
RUN mkdir -p tmp
RUN mkdir -p examples
RUN mkdir -p ~/.cache/huggingface

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://localhost:11434
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose the port that Gradio will run on
EXPOSE 7860

# Run the application
CMD ["python", "app-huggingface.py"] 