FROM python:3.10-slim

WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements-huggingface.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-huggingface.txt

# Copy all application files
COPY . .

# Copy example X-ray images to the examples directory
RUN mkdir -p /app/examples

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads && chmod 777 /app/uploads
RUN mkdir -p /app/tmp && chmod 777 /app/tmp
RUN mkdir -p /app/temp && chmod 777 /app/temp
RUN mkdir -p /app/examples && chmod 777 /app/examples
RUN mkdir -p /app/cache && chmod 777 /app/cache
RUN mkdir -p /tmp && chmod 777 /tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://localhost:11434
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV UPLOAD_DIR=/app/uploads
ENV TEMP_DIR=/app/temp
ENV TMP_DIR=/app/tmp
ENV TMPDIR=/app/tmp
ENV EXAMPLES_DIR=/app/examples
ENV CACHE_DIR=/app/cache

# Configure Python to use our temp directory
ENV PYTHONPATH=${PYTHONPATH}:/app
ENV MPLCONFIGDIR=/app/tmp

# Expose the port that Gradio will run on
EXPOSE 7860

# Run the application
CMD ["python", "app-huggingface.py"] 