FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY data /app/data

RUN mkdir -p /app/artifacts/models \
    && pip install --upgrade pip \
    && pip install .

EXPOSE 8000

CMD ["churn-serve", "--host", "0.0.0.0", "--port", "8000"]
