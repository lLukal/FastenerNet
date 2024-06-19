# Use the official TensorFlow Docker image
FROM tensorflow/tensorflow:latest-gpu

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and other Python packages
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install jupyter notebook matplotlib scikit-learn opencv-python datumaro seaborn

# Install VSCode Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Expose Jupyter and VSCode Server ports
EXPOSE 8888
EXPOSE 8080

# Set the working directory
WORKDIR /workspace

# Command to start Jupyter Notebook
CMD ["sh", "-c", "jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root & code-server --bind-addr 0.0.0.0:8080 --auth none /workspace"]