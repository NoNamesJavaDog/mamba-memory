FROM python:3.12-slim AS base

WORKDIR /app

# Install build deps for hnswlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY mamba_memory/ mamba_memory/

RUN pip install --no-cache-dir -e ".[server,google]"

# Data volume
VOLUME /data

ENV MAMBA_MEMORY_DB=/data/mamba-memory.db

EXPOSE 8420

# Default: HTTP server. Override with --mcp for MCP mode.
CMD ["mamba-memory", "serve", "--http", "--host", "0.0.0.0", "--port", "8420", "--db", "/data/mamba-memory.db"]
