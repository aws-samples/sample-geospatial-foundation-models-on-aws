# ARG PYTHON_VERSION=3.9

# FROM --platform=linux/amd64 public.ecr.aws/lambda/python:${PYTHON_VERSION}

# WORKDIR /tmp

# RUN pip install pip -U
# RUN pip install "fastapi==0.94.0" "titiler.application==0.7.1" "mangum>=0.10.0" "aioredis==1.3.1" "aiocache" "aiocache[redis]" "msgpack" "ujson" -t /asset --no-binary pydantic

# # Reduce package size and remove useless files
# RUN cd /asset && find . -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-[2-3][0-9]//'); cp $f $n; done;
# RUN cd /asset && find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
# RUN cd /asset && find . -type f -a -name '*.py' -print0 | xargs -0 rm -f
# RUN find /asset -type d -a -name 'tests' -print0 | xargs -0 rm -rf
# RUN rm -rdf /asset/numpy/doc/ /asset/boto3* /asset/botocore* /asset/bin /asset/geos_license /asset/Misc

# COPY ./*.py /asset/

# CMD ["echo", "hello world"]

ARG PYTHON_VERSION=3.9

FROM --platform=linux/amd64 public.ecr.aws/lambda/python:${PYTHON_VERSION}

WORKDIR /tmp

# Install system dependencies
RUN yum update -y && yum install -y \
    gcc \
    gcc-c++ \
    make \
    wget \
    tar \
    gzip \
    bzip2-devel \
    openssl-devel \
    libffi-devel \
    && yum clean all

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages
RUN pip install --no-cache-dir \
    "fastapi==0.94.0" \
    "titiler.application==0.7.1" \
    "mangum>=0.10.0" \
    "aioredis==1.3.1" \
    "aiocache" \
    "aiocache[redis]" \
    "msgpack" \
    "ujson" \
    -t /asset --no-binary pydantic

# Install numexpr separately
RUN pip install --no-cache-dir numexpr -t /asset

# Reduce package size and remove useless files
RUN cd /asset && find . -type f -name '*.pyc' | while read f; do n=$(echo $f | sed 's/__pycache__\///' | sed 's/.cpython-[2-3][0-9]//'); cp $f $n; done;
RUN cd /asset && find . -type d -a -name '__pycache__' -print0 | xargs -0 rm -rf
RUN cd /asset && find . -type f -a -name '*.py' -print0 | xargs -0 rm -f
RUN find /asset -type d -a -name 'tests' -print0 | xargs -0 rm -rf
RUN rm -rdf /asset/numpy/doc/ /asset/boto3* /asset/botocore* /asset/bin /asset/geos_license /asset/Misc

COPY ./*.py /asset/

CMD ["echo", "hello world"]
