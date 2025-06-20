FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

# Package version control

ARG CUDA_VERSION=11.8
ARG PYTORCH_VERSION=2.7.1
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


RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/ 

# Set up environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION ${PYTORCH_VERSION}

USER $USERNAME
# # # give permission to user
# # # Install repo dependencies 
RUN pip install git+https://github.com/openai/CLIP.git 
# RUN python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
RUN python -c "import clip; clip.load('$CLIP_VERSION', device='cpu')"

COPY ./* $WORKDIR
RUN pip install -r requirements.txt



