FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# MetaTrader5 is Windows-only; skip it in the Linux container so the
# dashboard + paper mode + backtester + reviewer still run.
RUN grep -v -i "^metatrader5" requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt

COPY . .

RUN mkdir -p logs

EXPOSE 8050

CMD ["python", "run.py", "--mode", "paper"]
