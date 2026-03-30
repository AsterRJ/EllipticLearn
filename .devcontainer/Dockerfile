FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    git curl wget ca-certificates unzip \
    build-essential pkg-config \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils python3.11-venv \
    libgl1 libegl1 libx11-6 libxext6 libxrender1 libsm6 libxrandr2 \
    libglib2.0-0 x11-apps mesa-utils libosmesa6-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ---- Create venv (prevents pip fighting apt/distutils packages) ----
ENV VENV_PATH=/opt/venv
# RUN python3.11 -m venv ${VENV_PATH}
# ENV PATH="${VENV_PATH}/bin:${PATH}"
# Install Python 3.10 + venv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual env with Python 3.10
RUN python3.10 -m venv /opt/venv

# Ensure venv is first on PATH
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip tooling in venv
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel


RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Core packages (install everything into the venv)
RUN pip install --no-cache-dir --upgrade "jax[cuda12]"
RUN pip install --no-cache-dir --upgrade open3d
RUN pip install --no-cache-dir \
    numpy==1.26.4 scipy pandas matplotlib plotly \
    jupyterlab ipykernel tqdm rich optax PyQt5 seaborn \
    h5py torch PyDMD neuraloperator pdebench greenlearning deepxde

# Non-root user
RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /workspace
