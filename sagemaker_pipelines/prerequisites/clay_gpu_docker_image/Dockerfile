
#Build from sagemaker distro image: https://gallery.ecr.aws/sagemaker/sagemaker-distribution
FROM public.ecr.aws/sagemaker/sagemaker-distribution:1.8.0-gpu

ARG NB_USER="sagemaker-user"
ARG NB_UID=1000
ARG NB_GID=100

ENV MAMBA_USER=$NB_USER

USER $ROOT

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Mamba
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
    && mv bin/micromamba /usr/local/bin/ \
    && rm -rf bin

# Set up Mamba environment
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# Copy environment files
COPY environment.yml /tmp/environment.yml
COPY requirements.txt /tmp/requirements.txt

# Create and activate the environment, install dependencies, and pip requirements
RUN micromamba create -f /tmp/environment.yml && \
    micromamba run -n claymodel pip install -r /tmp/requirements.txt && \
    micromamba clean --all --yes

# Set environment variable for the environment name
ENV ENV_NAME=claymodel

# Set the default environment for inference
ENV SAGEMAKER_JOB_CONDA_ENV=claymodel

# Set the default environment path
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH

# Copy model files
COPY  clay_assets/ /home/sagemaker-user/clay-model

# Add healthcheck to verify code is running at default path
HEALTHCHECK --interval=30s --timeout=3s \
  CMD pgrep -f "python3 /opt/ml/processing/input/code/" || exit 1

# Set the entrypoint to activate the environment
ENTRYPOINT ["/bin/bash", "-c"]
