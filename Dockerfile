# Use an explicit slim Python base image
FROM python:3.10-slim

WORKDIR /workspace

# Install system utilities required for dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency mappings
COPY requirements.txt .

# Install standard Python libraries and target explicit CPU runtime configurations for torch
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly download language dictionaries during build time so container startup is instant
RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader stopwords

# Copy backend files into execution layer
COPY ./app ./app
COPY ./models ./models

# Document endpoint visibility expectations
EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Initialize production ASGI server loop
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]