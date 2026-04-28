# --- BUILD STAGE (Frontend) ---
FROM node:18-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- FINAL STAGE (Production) ---
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy backend code
COPY app/ ./app
COPY scripts/ ./scripts
COPY models/ ./models
COPY sample_data_original/ ./sample_data_original/

# Copy built frontend from stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Serve both Frontend and Backend via FastAPI on the Hugging Face port
CMD python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
