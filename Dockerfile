FROM python:3.12-slim

WORKDIR /app

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/server/requirements.txt && \
    pip install --no-cache-dir openenv-core huggingface_hub pydantic

COPY . /app
COPY data /app/data

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
