FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

# pay attention ARG "cuda_ver" should match base image above
ARG cuda_ver=cu102
# python 3.7.9
ARG miniconda_ver=Miniconda3-py37_4.9.2-Linux-x86_64.sh
ARG project=IMRL
ARG username=user
ARG uid=1000
ARG gid=1000
ARG password=user
ARG torch_ver=1.6.0
ARG torchvision_ver=0.7.0
ARG gym_ver=0.17.3

# Fix GPG error
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install some basic utilities and create users
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    graphviz \
    graphviz-dev \
    && rm -rf /var/lib/apt/lists/*
    
RUN groupadd -g $gid -o $username \
    && useradd -md /home/${username} -u $uid -g $gid -o -s /bin/bash ${username} \
    # user owns the home dir
    && chown -R ${username} /home/${username}/ \
    # set user password
    && echo ${username}:${password} | chpasswd \
    # add user to sudoers
    && echo ${username}" ALL=(ALL:ALL) ALL" > /etc/sudoers.d/90-user

# switch to user
USER ${username}

# to home dir
WORKDIR /home/${username}/

# download conda installer and save as "~/miniconda.sh"
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/${miniconda_ver} \
    # user owns installer
    && chmod +x ~/miniconda.sh \
    # install conda with name ~/${project}-miniconda-environment;
    # "-p" = path of installed conda env;
    # sublime open https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh to check meaning of -b -p.
    && bash ~/miniconda.sh -b -p ~/${project}-miniconda-environment \
    && rm ~/miniconda.sh

ENV CONDA_AUTO_UPDATE_CONDA=false \
    # add conda to env variables
    PATH=~/${project}-miniconda-environment/bin:$PATH

RUN ~/${project}-miniconda-environment/bin/pip install \
    # install pytorch
    torch==${torch_ver} torchvision==${torchvision_ver} -f https://download.pytorch.org/whl/${cuda_ver}/torch_stable.html \
    # install gym
    && ~/${project}-miniconda-environment/bin/pip install --upgrade pip \
    && ~/${project}-miniconda-environment/bin/pip install gym==${gym_ver} \
    && ~/${project}-miniconda-environment/bin/pip install ortools tqdm fire networkx pygraphviz

VOLUME [ "/home/${username}/IMRL/" ]
