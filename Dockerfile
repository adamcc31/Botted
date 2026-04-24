# Gunakan Python 3.11 Slim (Optimal untuk VPS)
FROM python:3.11-slim

# Set working directory di dalam kontainer
WORKDIR /app

# Instal dependensi sistem yang dibutuhkan oleh Polars/Numpy (jika ada)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements.txt
COPY requirements.txt .

# Instal dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode proyek (termasuk folder scripts)
COPY . .

# Buat folder logs untuk telemetry
RUN mkdir -p logs

# Set Environment Variables default (bisa di-override di Railway Dashboard)
ENV PYTHONUNBUFFERED=1

# Perintah untuk menjalankan bot (engine baru; default dry-run)
CMD ["python", "main.py", "--mode", "dry-run"]
