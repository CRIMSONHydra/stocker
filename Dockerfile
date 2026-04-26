# ── Stage 1: build the React frontend (Vite + React 19 + Tailwind v4) ──────
FROM node:20-slim AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ────────────────────────────────────────────────
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip compile pyproject.toml -o requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY server/ /app/server/
COPY data/ /app/data/
COPY inference.py client.py README.md /app/

# Pre-built council cache — gives judges instant UI responses for the 3
# stable tasks. Optional: COPY only fires if .cache/ exists in build context.
COPY .cache/ /app/.cache/

# Pull in the built frontend so FastAPI mounts the React SPA at /web
COPY --from=frontend-builder /frontend/dist /app/frontend/dist

RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
