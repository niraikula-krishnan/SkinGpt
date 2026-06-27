FROM python:3.11-slim

RUN useradd -m -u 1000 user

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /app

COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
