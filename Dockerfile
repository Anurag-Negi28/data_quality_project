FROM python:3.10-slim-bookworm

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get upgrade -y && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

CMD ["python", "src/main.py"]