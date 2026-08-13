#── Base image ──────────────────────────────────────────────────────────
# slim, not alpine: alpine's musl libc frequently breaks compiled Python
# packages (numpy, torch) in ways that cost more debugging time than the
# smaller image size saves. slim is the standard, pragmatic choice for
# ML-adjacent Python services.
FROM python:3.12-slim

WORKDIR /app

# ── System dependencies ─────────────────────────────────────────────────
# build-essential: needed to compile some Python packages with C extensions
# libgomp1: required by some numpy/torch operations at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies layer — copied and installed BEFORE app code ───────────
# This ordering matters: Docker caches each layer. As long as requirements.txt
# doesn't change, this expensive pip install layer gets reused on every
# rebuild, even after editing app code — dramatically speeding up iteration.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code — copied AFTER dependencies ─────────────────────────
COPY core/ ./core/
COPY phase_07_production/ ./phase_07_production/
COPY sample_documents/ ./sample_documents/
COPY phase_07_production/static/ ./phase_07_production/static/

# ── Runtime configuration ────────────────────────────────────────────────
# Render/Railway/Fly all inject a PORT environment variable at runtime and
# expect the app to bind to it — never hardcode a port here.
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Bind to 0.0.0.0 (not 127.0.0.1) so the container is reachable from OUTSIDE
# itself — 127.0.0.1 inside a container only accepts connections from within
# that same container, which would make the deployed app completely unreachable.
CMD ["sh", "-c", "uvicorn phase_07_production.app:app --host 0.0.0.0 --port ${PORT:-8000}"]