FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# Package version control

ARG PYTHON_VERSION=3.10
ARG CUDA_VERSION=11.3
ARG PYTORCH_VERSION=1.11
ARG CUDA_CHANNEL=nvidia
ARG INSTALL_CHANNEL=pytorch
ARG CLIP_VERSION=ViT-L/14@336px

# Setup workdir and non-root user

ARG USERNAME=nhtlong
WORKDIR /home/$USERNAME/workspace/

ENV TZ=Asia/Ho_Chi_Minh \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt-get install -y --no-install-recommends curl git sudo &&\
    useradd --create-home --shell /bin/bash $USERNAME &&\
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME &&\
    chmod 0440 /etc/sudoers.d/$USERNAME &&\
    rm -rf /var/lib/apt/lists/*


RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get -qq update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    gcc \
    tmux \
    libjpeg-dev \
    unzip bzip2 ffmpeg libsm6 libxext6 \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*


# # Install mamba
RUN curl -fsSL -v -o ~/mambaforge.sh -O  https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh  && \
    chmod +x ~/mambaforge.sh && \
    ~/mambaforge.sh -b -p /opt/conda && \
    chown -R $USERNAME:$USERNAME /opt/conda/ && \
    rm ~/mambaforge.sh 

RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/ && \
    chown -R $USERNAME:$USERNAME /opt/conda/

RUN /opt/conda/bin/mamba install -c "${INSTALL_CHANNEL}" -c "${CUDA_CHANNEL}" -y \
    python=${PYTHON_VERSION} \
    pytorch=${PYTORCH_VERSION} torchvision "cudatoolkit=${CUDA_VERSION}" && \
    /opt/conda/bin/mamba clean -ya

# Set up environment variables
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

USER $USERNAME
# # # give permission to user
RUN conda init bash
# # # Install repo dependencies 
RUN pip install git+https://github.com/openai/CLIP.git 
RUN python -c "import clip; clip.load('$CLIP_VERSION', device='cpu')"

COPY ./* $WORKDIR
RUN pip install -r requirements.txt



