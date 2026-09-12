# Single-container build for Hugging Face Spaces (Docker SDK):
# stage 1 builds the React dashboard, stage 2 serves it + the API with FastAPI.

FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

# Spaces run the container as a non-root user (uid 1000), so anything the app
# writes at runtime has to live somewhere that user owns. matplotlib (pulled in
# by backend/report.py) insists on a writable config dir and fails the import
# if it cannot find one — the usual cause of a Space that builds fine and then
# will not start.
ENV MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Every local module backend/main.py imports — weather_live.py included, or
# the container builds fine and then dies on startup with ModuleNotFoundError.
COPY features.py models.py data_access.py data_sources.py geo.py weather_live.py ./
COPY backend/ backend/
COPY model/ model/
COPY --from=frontend-build /app/frontend/dist frontend/dist

RUN chown -R appuser:appuser /app
USER appuser

# 7860 is the port Spaces expects (see app_port in README.md's frontmatter).
EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
