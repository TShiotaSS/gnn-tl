FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

# python3.9
RUN apt update && apt install -y \
    python3.9-dev \
    cmake \
    libcairo2-dev \
    libeigen3-dev \
    swig \
    wget \
    vim

WORKDIR /usr/bin

RUN ln -nfs python3 python && \
    ln -nfs python3.9 python3 && \
    ln -nfs python3.9-config python3-config

WORKDIR /

# openbabel
RUN apt update && apt install -y \
    openbabel \
    libopenbabel-dev \
    python3-pip \
    python3-openbabel
RUN ln -s /usr/include/openbabel3 /usr/local/include/openbabel3

# JupyterLab
RUN pip install --upgrade pip \
&&  pip install --no-cache-dir \
    black \
    nb_black \
    jupyterlab==3.6.5 \
    jupyterlab-git \
    lckr-jupyterlab-variableinspector \
    ypy-websocket==0.8.2 \
    jupyterlab_widgets==1.1.4 \
    ipywidgets==7.7.2 \
    import-ipynb

RUN pip install --upgrade pip \
&&  pip install --no-cache-dir \
    pandas \
    matplotlib \
    japanize-matplotlib \
    mlxtend \
    seaborn \
    plotly

RUN pip install --upgrade pip \
&&  pip install --no-cache-dir \
    nglview \
    graphviz

RUN pip install --upgrade pip \
&&  pip install --no-cache-dir \
    openbabel

RUN pip install --upgrade pip \
&&  pip install --no-cache-dir \
    scikit-learn==1.2.2 \
    optuna==3.3.0

RUN pip install --upgrade pip \
&&  pip install --no-cache-dir \
    # skqulacs==0.5.0 \
    qulacs==0.5.6 \
    numpy==1.24.4

COPY times.ttf /usr/local/lib/python3.9/dist-packages/matplotlib/mpl-data/fonts/ttf/