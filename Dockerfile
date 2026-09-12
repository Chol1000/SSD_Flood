# Single-container build: stage 1 builds the React dashboard, stage 2 serves
# it and the API from one FastAPI process. Runs unchanged on Render, Cloud Run
# and Hugging Face Spaces.

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

# Hosts disagree on which port to serve: Render and Cloud Run inject $PORT,
# while Spaces expects the app_port from README.md's frontmatter. Honour $PORT
# when it is set and fall back to 7860, so the same image runs on any of them.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
