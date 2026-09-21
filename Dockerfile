# ---------------------------------------------------------------------------
# RetinaScope — public screening portal
#
# Serves the REAL trained pipeline (outputs/artifacts/grader.pt +
# segmentation.pt) behind the FastAPI app in src/drscreen/api.py. Every number
# the portal shows is produced by running the uploaded image through
# DRScreeningPipeline.run on this container's CPU. Nothing is pre-baked.
#
# Build context is the repo root (see .dockerignore).
#   docker build -t retinascope .
#   docker run --rm -p 7860:7860 retinascope
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# libglib2.0-0 is the one system library opencv-python-headless still links
# against. libgomp1 backs torch's OpenMP thread pool.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs the container as uid 1000; matching that here means
# the same image works on Spaces, Render, Railway and Fly without changes.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /home/user/app

# --- dependencies (own layer, so code edits don't re-download torch) -------
# The CPU wheel index matters: the default PyPI torch bundles CUDA and lands
# at ~2.5 GB, which no free tier will build. The CPU wheel is ~200 MB and this
# model runs on CPU anyway.
COPY --chown=user deploy/requirements-serve.txt ./deploy/requirements-serve.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.13.0 torchvision==0.28.0 \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r deploy/requirements-serve.txt

# --- application ----------------------------------------------------------
COPY --chown=user src/                            ./src/
COPY --chown=user web/                            ./web/
COPY --chown=user outputs/artifacts/              ./outputs/artifacts/
COPY --chown=user outputs/verification_set/images/ ./outputs/verification_set/images/
COPY --chown=user outputs/verification_set/EXPECTED_GRADES.md ./outputs/verification_set/
COPY --chown=user outputs/validation/             ./outputs/validation/

ENV PYTHONPATH=/home/user/app/src \
    DRSCREEN_ARTIFACTS=/home/user/app/outputs/artifacts \
    DRSCREEN_SAMPLES=/home/user/app/outputs/verification_set/images \
    DRSCREEN_AUDIT_LOG=/tmp/reviews.jsonl \
    DRSCREEN_WARMUP=1 \
    PORT=7860

# One worker on purpose. Inference is CPU-bound and already multi-threaded;
# a second worker would double RAM for the model and contend for the same
# cores. Concurrency is handled by the semaphore in api.py instead.
ENV OMP_NUM_THREADS=2 TORCH_NUM_THREADS=2

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/health',timeout=8).status==200 else 1)"

# $PORT is honoured so the same image drops onto Render/Railway/Fly unchanged.
CMD ["sh", "-c", "exec uvicorn drscreen.api:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 75"]
