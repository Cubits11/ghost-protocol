# Dockerfile - Ghost Protocol v0.1 Container
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG GHOST_VERSION=0.1.0
ARG BUILD_DATE
ARG VCS_REF

# Labels for container metadata
LABEL org.opencontainers.image.title="Ghost Protocol" \
      org.opencontainers.image.description="Emotionally Sovereign AI with Constitutional Constraints" \
      org.opencontainers.image.version="${GHOST_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Ghost Protocol Team" \
      org.opencontainers.image.url="https://github.com/ghost-protocol/ghost-protocol" \
      org.opencontainers.image.documentation="https://github.com/ghost-protocol/ghost-protocol/README.md" \
      org.opencontainers.image.source="https://github.com/ghost-protocol/ghost-protocol" \
      maintainer="Ghost Protocol Team <ghost-protocol@example.com>"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libsqlcipher-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libsqlcipher0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r ghost && useradd -r -g ghost ghost

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create application directory and set ownership
WORKDIR /app
RUN chown -R ghost:ghost /app

# Copy application code
COPY --chown=ghost:ghost . .

# Create data directory for databases and logs
RUN mkdir -p /app/data/logs /app/data/databases /app/config && \
    chown -R ghost:ghost /app/data /app/config

# Switch to non-root user
USER ghost

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    GHOST_ENV=production \
    GHOST_DATA_DIR=/app/data \
    GHOST_CONFIG_DIR=/app/config \
    GHOST_LOG_LEVEL=INFO

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); from ghost_protocol_main import GhostProtocolSystem; g = GhostProtocolSystem(); print('healthy' if g.system_initialized else 'unhealthy'); g.close()"

# Volume for persistent data
VOLUME ["/app/data", "/app/config"]

# Default command - run Streamlit UI
CMD ["streamlit", "run", "ui/demo.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]

# Alternative commands (can be overridden):
# CLI mode: docker run ghost-protocol python cli.py --demo
# Benchmark: docker run ghost-protocol python cli.py --benchmark
# Test: docker run ghost-protocol python cli.py --test "Hello world"