FROM python:3.13-slim

LABEL org.opencontainers.image.title="Auto-ChemInstruct"
LABEL org.opencontainers.image.description="Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry"
LABEL org.opencontainers.image.authors="Deyan Stepanov, Akash Kundu"
LABEL org.opencontainers.image.source="https://github.com/auto-cheminstruct/auto-cheminstruct"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .python-version* ./

RUN uv sync --frozen || uv sync

COPY . .

ENV PYTHONUNBUFFERED=1
ENV AUTOCHEM_CONFIG=/app/configs/default.yaml
ENV FIRECRAWL_API_KEY=""

RUN mkdir -p logs datasets .checkpoints .chromadb benchmarks

CMD ["uv", "run", "python", "-m", "src.cli.main", "pipeline", "-n", "1"]
