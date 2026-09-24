# GPU image with the standard pipeline.
#   docker build -t soundakira .
#   docker run --gpus all -e HF_TOKEN -v $PWD:/data -w /data soundakira \
#       process sources.txt -c config.yaml
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /opt/soundakira
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[standard]" "audio-separator[gpu]"

ENTRYPOINT ["soundakira"]
CMD ["--help"]
