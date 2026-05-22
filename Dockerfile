# Multi-stage build for coding-guardrails
# Stage 1: Build the package
# Stage 2: Runtime image with llama.cpp

# ── Build stage ──
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir --upgrade pip build && \
    python -m build --wheel

# ── Runtime stage ──
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/stawils/coding-guardrails"
LABEL org.opencontainers.image.description="Safe, reliable local coding agent backend"
LABEL org.opencontainers.image.licenses="MIT"

# Non-root user
RUN groupadd -r cg && useradd -r -g cg -d /home/cg -s /sbin/nologin cg

# Install the wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Config and working dirs
COPY --chown=cg:cg configs/guardrail-config.yaml /etc/coding-guardrails/config.yaml
RUN mkdir -p /home/cg/models && chown cg:cg /home/cg/models

USER cg
WORKDIR /home/cg

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8081/health')" || exit 1

# Default config
ENV CODING_GUARDRAILS_CONFIG=/etc/coding-guardrails/config.yaml

EXPOSE 8081

ENTRYPOINT ["coding-guardrails"]
CMD ["serve", "--backend-url", "http://localhost:8080", "--model", "default", "--port", "8081"]
