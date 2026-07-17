FROM python:3.11-slim

# Install LibreOffice and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libreoffice \
      libreoffice-writer \
      python3-magic \
      libmagic1 \
      fonts-liberation \
      && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify LibreOffice installed
RUN libreoffice --version

EXPOSE 10000

CMD gunicorn app:app \
    --workers 2 \
    --timeout 180 \
    --bind 0.0.0.0:${PORT:-10000} \
    --worker-class sync
