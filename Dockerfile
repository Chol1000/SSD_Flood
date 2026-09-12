# Single-container build for Hugging Face Spaces (Docker SDK):
# stage 1 builds the React dashboard, stage 2 serves it + the API with FastAPI.

FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY features.py models.py data_access.py data_sources.py geo.py ./
COPY backend/ backend/
COPY model/ model/
COPY --from=frontend-build /app/frontend/dist frontend/dist

EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
