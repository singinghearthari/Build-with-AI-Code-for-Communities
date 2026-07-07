# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV/YOLO)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install CPU version of PyTorch first to save space/time and avoid pulling GPU libraries
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 7860 which is required by Hugging Face Spaces
EXPOSE 7860

# Set environment variable for the port
ENV PORT=7860

# Command to run the application
CMD ["python", "main.py"]
