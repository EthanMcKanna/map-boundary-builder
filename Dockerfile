# Cloudflare Containers image: runs the full Python pipeline behind the
# async web server (SSE progress events, 50 MB uploads) that app.js
# natively targets.
FROM python:3.12-slim

# cairosvg needs libcairo; opencv-headless needs glib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcairo2 libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY map_boundary_builder ./map_boundary_builder
RUN pip install --no-cache-dir .

ENV MAP_BOUNDARY_CACHE_DIR=/tmp/map-boundary-builder-cache \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

EXPOSE 8765
CMD ["map-boundary-web", "--host", "0.0.0.0", "--port", "8765"]
