FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

COPY main.py helpers.py rag.py ./
COPY docs ./docs

ENV IDE_BACKEND_URL=http://host.docker.internal:3001/api

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "main.py"]