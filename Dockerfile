FROM --platform=linux/amd64 nvidia/cuda:11.8.0-devel-ubuntu20.04 AS base

# Update and install required packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    cmake \
    python3-dev \
    python3-pip \
    wget \
    curl \
    git \
    zlib1g-dev \
    libssl-dev \
    libffi-dev \
    libcrypt-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Miniconda
WORKDIR /opt
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create and activate the Conda environment
COPY environment.yaml /tmp/environment.yaml
RUN conda env create -f /tmp/environment.yaml && \
    conda clean -afy

# Set the environment to automatically activate
ENV CONDA_DEFAULT_ENV=cheap
RUN echo "source activate cheap" > ~/.bashrc

# Install PyTorch with CUDA
RUN conda install -n cheap -c pytorch -c nvidia \
    pytorch=2.0.1 torchvision torchaudio cudatoolkit=11.8 && \
    conda clean -afy

# Copy the OpenFold source code
WORKDIR /app
COPY . /app

# Set environment variables for CUDA
ENV CXX=g++ CC=gcc
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Build and install OpenFold
RUN /bin/bash -c "source activate cheap && \
    pip install git+https://github.com/amyxlu/openfold.git && \
    pip install -e ."

# Default command
CMD ["/bin/bash"]