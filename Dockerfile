FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.11
ARG EN_CORE_WEB_SM_VERSION=3.7.1

ARG VLLM_VERSION=0.2.4
ARG VLLM_CUDA_VERSION=118
ARG VLLM_PYTHON_VERSION=311
ARG XFORMERS_VERSION=0.0.23
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /proconsul

RUN rm /etc/apt/sources.list.d/cuda.list

RUN apt-get update \
    && apt install -y wget gpg software-properties-common lsb-release \
    && mkdir /run/sshd \
    && add-apt-repository 'ppa:deadsnakes/ppa' -y \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' -y \
    && apt-get update \
    && apt install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dbg \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils \
    curl \
    openssh-server \
    tar \
    wget \
    cmake \
    build-essential \
    git \
    libffi-dev \
    screen \
    htop \
    nano \
    vim \
    fakeroot \
    ncurses-dev \
    xz-utils \
    libssl-dev \
    bc \
    flex \
    libelf-dev \
    bison \
    bear \
    mc \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSk https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && sed -i 's/^#\(PermitRootLogin\) .*/\1 yes/' /etc/ssh/sshd_config \
    && sed -i 's/^\(UsePAM yes\)/# \1/' /etc/ssh/sshd_config

RUN git clone https://github.com/TypingCat13/ProConSuL.git

RUN python${PYTHON_VERSION} -m pip install setuptools==69.5.1 \
    && python${PYTHON_VERSION} -m pip install https://download.pytorch.org/whl/cu118/torch-2.1.1%2Bcu118-cp311-cp311-linux_x86_64.whl \
    && cd ./ProConSuL \
    && python${PYTHON_VERSION} -m pip install -r requirements.txt\
    && python${PYTHON_VERSION} -m nltk.downloader stopwords \
    && wget  https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-${EN_CORE_WEB_SM_VERSION}/en_core_web_sm-${EN_CORE_WEB_SM_VERSION}.tar.gz \
    && python${PYTHON_VERSION} -m pip install ./en_core_web_sm-${EN_CORE_WEB_SM_VERSION}.tar.gz \
    && rm ./en_core_web_sm-${EN_CORE_WEB_SM_VERSION}.tar.gz \
    && python${PYTHON_VERSION} -m pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${VLLM_CUDA_VERSION}-cp${VLLM_PYTHON_VERSION}-cp${VLLM_PYTHON_VERSION}-manylinux1_x86_64.whl xformers==${XFORMERS_VERSION} --trusted-host=github.com --trusted-host=objects.githubusercontent.com --extra-index-url https://download.pytorch.org/whl/cu${VLLM_CUDA_VERSION} \
    && python${PYTHON_VERSION} -m pip install -e .

CMD /etc/init.d/ssh restart && while true; do sleep infinity; done

EXPOSE 22
