# Use Ubuntu as base
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl bzip2 ca-certificates libglib2.0-0 libx11-6 \
    libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

# Create conda environment with Python 3.9.25
RUN conda create -y -n vision python=3.9.25 && conda clean -afy

SHELL ["conda", "run", "-n", "vision", "/bin/bash", "-c"]

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt /workspace/requirements.txt

# Install all Python packages
RUN pip install -r /workspace/requirements.txt

# Copy your project into the container
WORKDIR /workspace
COPY . /workspace

# Install your local packages in editable mode
RUN pip install -e ./ByteTrack --no-build-isolation
RUN pip install -e ./sort --no-build-isolation

CMD ["/bin/bash"]