# Python slim keeps the image small and boots fast on HF's free CPU tier.
FROM python:3.11-slim

# Non-interactive apt, unbuffered Python — logs stream to HF's log viewer
# in real time instead of getting stuck in a buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    JAZZCASH_USE_API=false

WORKDIR /app

# System deps first — libgomp for scikit-learn, curl for the healthcheck.
# Kept minimal because every apt package is image size and attack surface.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Requirements before code so pip layer caches across code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Everything else. .dockerignore keeps .env, .venv, notebooks etc. out.
COPY . .

# HF requires the container listen on 7860.
EXPOSE 7860

# HEALTHCHECK so HF's UI shows green when Streamlit is actually serving,
# not just when the process started.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Runs via python -m so it picks up the venv-less system Python's streamlit,
# same reliability point as your local run command.
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py"]