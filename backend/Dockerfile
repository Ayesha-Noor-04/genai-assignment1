FROM python:3.9-slim

# System deps for Pillow's optional image codecs (not strictly required,
# but avoids "decoder jpeg not available" surprises)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/models
ENV SAMPLE_DIR=/samples

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]