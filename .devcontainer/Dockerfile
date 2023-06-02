# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.238.0/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM --platform=linux/x86_64 mcr.microsoft.com/vscode/devcontainers/base:ubuntu

# Install base utilities
RUN apt-get update && apt-get install -y build-essential && apt-get install -y wget && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Update conda base environment from environment.yml
COPY environments/devcontainer_environment.yml /tmp/environment.yml
RUN if [ -f "/tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/environment.yml; fi \
    && rm -rf /tmp/environment.yml

# git autocompletion in bash
RUN wget -P /home/vscode/ https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash
RUN echo "source ~/git-completion.bash" >> /home/vscode/.bashrc