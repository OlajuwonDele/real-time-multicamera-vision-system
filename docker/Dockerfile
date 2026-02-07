FROM python:3.9-slim

# System dependencies (NO python installs here)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project files
COPY . /workspace
COPY requirements.txt /workspace/requirements.txt

# Python deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r /workspace/requirements.txt

# ---- IMPORTANT: ONNX Runtime GPU ----
RUN pip uninstall -y onnxruntime || true
RUN pip install onnxruntime-gpu==1.19.2

# Clone + install ByteTrack
RUN git clone https://github.com/ifzhang/ByteTrack.git /workspace/ByteTrack \
    && pip install -e /workspace/ByteTrack --no-build-isolation

CMD ["/bin/bash"]
