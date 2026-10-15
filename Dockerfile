# SceneIt - multi-stage build.
#
# Two targets from one source:
#
#   docker build --target serve -t sceneit:serve .   # ~0.9 GB, web app
#   docker build --target full  -t sceneit:full  .   # ~2.9 GB, + captioning
#
# `full` extends `serve`, so it is the whole package in one image. `serve`
# exists because captioning runs ONCE PER FRAME AT INGEST and never during a
# query -- a web replica that carries 1.95 GB of VLM weights it never loads is
# paying for RAM headroom it cannot use. Scale `serve`; run `full` as a job.
#
# The previous Dockerfile declared `as base` and never added a second stage,
# so build-essential and python3-dev shipped to production.

# ---------------------------------------------------------------- base ------
FROM python:3.12-slim-bookworm AS base

# Runtime libraries only. No compiler reaches this layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models

WORKDIR /app

# ------------------------------------------------------- llama builder ------
# llama-cpp-python publishes no wheels, so anything GGUF compiles from source.
# Compiling the upstream binary here instead keeps cmake and g++ out of the
# runtime image entirely.
FROM base AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG LLAMA_CPP_REF=b6050
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
        https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp \
     || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp

RUN cmake -S /tmp/llama.cpp -B /tmp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_CURL=OFF \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=ON \
    && cmake --build /tmp/build --target llama-server -j "$(nproc)" \
    && mkdir -p /opt/llama/bin /opt/llama/lib \
    && find /tmp/build -name 'llama-server' -type f -exec cp {} /opt/llama/bin/ \; \
    && find /tmp/build -name '*.so*' -type f -exec cp {} /opt/llama/lib/ \;

# --------------------------------------------------------- model fetch ------
# A separate stage so model downloads cache independently of code changes:
# editing app.py must not re-download 2 GB of weights.
FROM base AS models

COPY scripts/fetch_models.py /app/scripts/fetch_models.py

RUN python /app/scripts/fetch_models.py --set serve --dir /app/models
RUN python /app/scripts/fetch_models.py --set caption --dir /app/models

# --------------------------------------------------------------- deps -------
FROM base AS deps

COPY pyproject.toml README.md /app/

# Dependencies only, and NOT editable. `pip install -e` writes a .pth pointing
# back at /app, which breaks when the next stage copies site-packages without
# it. `py-modules = []` means there is no first-party code to install here, so
# this resolves the dependency tree and nothing else -- and the layer caches on
# pyproject.toml alone, so editing app.py never reinstalls numpy.
RUN pip install --no-cache-dir '.[onnx,faces,ingest]'

# -------------------------------------------------------------- serve -------
FROM base AS serve

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Search and face models only. Captioning weights are added by `full`.
COPY --from=models /app/models/clip /app/models/clip
COPY --from=models /app/models/faces /app/models/faces

COPY . /app

RUN mkdir -p output_frames temp_videos media/actors \
    && useradd --create-home --uid 10001 sceneit \
    && chown -R sceneit:sceneit /app
USER sceneit

EXPOSE 5002

# Hits a real endpoint rather than the SPA shell, so a half-initialised
# container is reported unhealthy instead of serving an empty library.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5002/api/library',timeout=5).status==200 else 1)"

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "app.py"]

# --------------------------------------------------------------- full -------
# serve + captioning. This is the complete package: Qwen3.5-2B is baked in,
# so no model is ever fetched at runtime.
FROM serve AS full

USER root

COPY --from=llama-builder /opt/llama/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-builder /opt/llama/lib/ /usr/local/lib/
RUN ldconfig && chmod +x /usr/local/bin/llama-server

# 1.28 GB weights + 0.67 GB vision projector. GGUF rather than safetensors so
# no torch is needed: fp16 would be 4.55 GB plus ~550 MB of torch.
COPY --from=models /app/models/Qwen3.5-2B-Q4_K_M.gguf /app/models/
COPY --from=models /app/models/Qwen3.5-2B-mmproj-F16.gguf /app/models/

RUN chown -R sceneit:sceneit /app/models
USER sceneit

ENV CAPTION_ENABLED=1 \
    CAPTION_SERVER_BIN=/usr/local/bin/llama-server

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "app.py"]
