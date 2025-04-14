# Use an official base image with Python and CUDA
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    libgl1 \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Copy environment files into the container
COPY environment.yaml /tmp/environment.yaml
COPY requirements.txt /tmp/requirements.txt

# Create conda environment
RUN conda env create -f /tmp/environment.yaml && \
    conda clean -afy

# Activate conda environment and install pip dependencies
RUN /bin/bash -c "source activate cheap && pip install -r /tmp/requirements.txt"

# Stage 2: Final image
FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Set environment variables
ENV PATH="/opt/conda/bin:$PATH"
ENV CONDA_PREFIX="/opt/conda"
ENV CHEAP_CACHE="/cache"

# Copy conda environment from the base stage
COPY --from=base /opt/conda /opt/conda

# Set the working directory
WORKDIR /workspace

# Clone the repository
RUN git clone https://github.com/weroks/cheap-proteins.git /workspace

# Clean up unnecessary files to reduce image size
RUN rm -rf /opt/conda/pkgs/* && \
    conda clean -a -y && \
    rm -rf /var/lib/apt/lists/*

# Set the default command to bash
CMD ["/bin/bash"]