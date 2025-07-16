ARG DISTRIBUTION_TYPE

#https://gallery.ecr.aws/sagemaker/sagemaker-distribution
FROM public.ecr.aws/sagemaker/sagemaker-distribution:2.2.1-$DISTRIBUTION_TYPE

USER root

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies with optimal flags
RUN apt-get --allow-releaseinfo-change update -y -qq && \
    apt-get install --no-install-recommends -y software-properties-common gpg-agent && \
    add-apt-repository --yes ppa:ubuntugis/ppa && \
    apt-get update -qq && \
    apt-get install --no-install-recommends -y \
        groff unzip libgdal-dev gdal-bin ffmpeg libsm6 libxext6 build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER sagemaker-user

# OPTIMIZATION 1: Install fundamental packages first 
# This speeds up the dependency solver significantly for core libraries
RUN MAMBA_NO_BANNER=1 \
    micromamba install -y -c conda-forge -n base \
    gdal=3.6.4 \
    && micromamba clean -ya

# OPTIMIZATION 2: Install geo packages
RUN MAMBA_NO_BANNER=1 \
    micromamba install -y -c conda-forge -n base \
    geopandas=0.13.2 \
    rasterio=1.3.8 \
    rioxarray=0.15.1 \
    leafmap=0.31.3 \
    && micromamba clean -ya

# Set environment variables for geospatial libraries
ENV PROJ_LIB=/opt/conda/share/proj \
    GDAL_DATA=/opt/conda/share/gdal

# OPTIMIZATION 3: Install machine learning packages separately to prevent conflicts
# Using fiona=1.9.6 to fix reading of geojson with geopandas
RUN MAMBA_NO_BANNER=1 \
    micromamba install -y -c conda-forge -n base \
    torchdata=0.7.1 \
    torchgeo=0.5.2 \
    torchvision=0.16.1 \
    scikit-image=0.22.0 \
    scikit-learn=1.4.0 \
    s3fs=2024.3.1 \
    umap-learn=0.5.7 \
    fiona=1.9.6 \
    && micromamba clean -ya

# OPTIMIZATION 4: Use pip for non-conda packages (faster than conda for these)
RUN pip install --no-cache-dir lancedb

# Add healthcheck to verify code is running at default path
HEALTHCHECK --interval=30s --timeout=3s \
  CMD pgrep -f "python3 /opt/ml/processing/input/code/" || exit 1

ENTRYPOINT ["jupyter-lab"]
CMD ["--ServerApp.ip=0.0.0.0", "--ServerApp.port=8888", "--ServerApp.allow_origin=*", "--ServerApp.token=''", "--ServerApp.base_url=/jupyterlab/default"]
