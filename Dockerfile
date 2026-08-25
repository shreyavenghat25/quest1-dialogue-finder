FROM python:3.11-slim

# --fix-missing + retries: campus/corporate networks that do HTTP traffic
# inspection can intermittently corrupt plain-HTTP package downloads
# (Debian's default mirror is HTTP, not HTTPS), surfacing as spurious
# "Hash Sum mismatch" errors unrelated to the packages themselves. Retrying
# resolves it when the corruption is transient; if it's a persistent
# network-level issue, switch networks (e.g. mobile hotspot) instead.
RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends --fix-missing \
    tesseract-ocr ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENTRYPOINT ["python", "solve.py"]
