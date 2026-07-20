FROM python:3.11-slim

# System dependencies:
# - libreoffice-writer: docx → pdf/text conversion for exam parsing
# - libmagic1: required at runtime by python-magic
# - fonts-liberation + fonts-dejavu-core: prevents font-substitution artifacts
#   in LibreOffice conversions (common cause of garbled extracted text)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice-writer \
    libmagic1 \
    fonts-liberation \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# gevent worker matches your requirements.txt; adjust --workers/--timeout
# based on your Render plan's CPU/RAM. 120s timeout gives LibreOffice
# conversions room to finish on slower instances.
CMD gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --worker-class gevent \
    --workers 2 \
    --timeout 120