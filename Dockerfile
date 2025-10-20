FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# system deps for healthcheck and docling (libGL for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# deps
COPY requirements_llamaindex.txt /app/requirements_llamaindex.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary -r requirements_llamaindex.txt

# app
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
