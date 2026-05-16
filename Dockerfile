# ── Stage 1: Build ────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

# System deps for PyMuPDF, torch, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Install the package + all deps into /install
RUN pip install --no-cache-dir --prefix=/install ".[dev]" -e .


# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /build/src /app/src

# Create runtime directories
RUN mkdir -p /app/data/chroma /app/data/faiss /app/agentic_output

# Non-root user for security
RUN useradd -m -u 1000 mmads && chown -R mmads:mmads /app
USER mmads

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "multimodal_ds.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
