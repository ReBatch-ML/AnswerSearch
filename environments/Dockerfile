# Check if this base image can work with the GPU's of azure
# https://github.com/Azure/AzureML-Containers/blob/master/README.md
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04

# Install base utilities
RUN apt-get update && apt-get install -y build-essential  && apt-get install -y wget && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


COPY online_scoring_environment.yml /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/online_scoring_environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/online_scoring_environment.yml; fi \
    && rm -rf /tmp/conda-tmp

COPY download_ds.py .
COPY connection_string.txt .
RUN python -u ./download_ds.py