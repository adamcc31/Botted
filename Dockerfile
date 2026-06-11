# Gunakan Python 3.11 Slim (Optimal untuk VPS)
FROM python:3.11-slim

# Set working directory di dalam kontainer
WORKDIR /app

# Instal dependensi sistem yang dibutuhkan
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    bash \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt
COPY requirements.txt .

# Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek (termasuk folder slingger/ dan scripts)
COPY . .

# Normalise line endings & set executable bit on startup script
# (Prevents \r\n issues if script was edited on Windows)
RUN dos2unix slingger/start.sh && chmod +x slingger/start.sh

# Buat folder logs untuk telemetry dan slingger pipeline
RUN mkdir -p logs slingger/logs slingger/models/candidates slingger/data

# Set Environment Variables default (bisa di-override di Railway Dashboard)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default CMD — overridden by railway.toml startCommand
CMD ["bash", "slingger/start.sh"]
