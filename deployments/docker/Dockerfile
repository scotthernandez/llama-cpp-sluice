# Build stage
FROM python:3.12-slim as builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml .
RUN pip install --upgrade pip
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install .

# Final stage
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY src/ /app/src/

ENV SLUICE_MODEL_PATH=/models/model.gguf
ENV SLUICE_TOTAL_POOL=98304
ENV SLUICE_RESERVED_POOL=32768
ENV SLUICE_PORT=8001

EXPOSE 8001

ENTRYPOINT ["python", "-m", "sluice.server"]
