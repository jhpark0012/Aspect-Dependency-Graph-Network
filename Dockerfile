FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        vim \
        htop \
        build-essential \
        unzip \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]