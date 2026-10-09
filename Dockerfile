# StanceSense-RT - dashboard and offline tools in a container (CPU only).
#
#   docker compose up --build          -> dashboard at http://localhost:8501
#
# The live camera assessment is NOT run in here: Docker Desktop on Windows cannot
# pass a webcam or an OpenCV window into a Linux container. Run
# `python scripts/live_assessment.py` on the host; it writes data/stancesense.db,
# which the containerised dashboard reads through the ./data volume.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# Shared libraries OpenCV and MediaPipe load at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY assets ./assets
COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY tests ./tests
COPY models ./models

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "scripts/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
