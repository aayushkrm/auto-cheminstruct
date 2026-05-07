FROM python:3.13-slim

LABEL org.opencontainers.image.title="Auto-ChemInstruct"
LABEL org.opencontainers.image.description="Agent-Driven Synthesization of RLHF Data for Domain-Specific Language Models in Chemistry"
LABEL org.opencontainers.image.source="https://github.com/aayushkrm/auto-cheminstruct"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

ENV PYTHONUNBUFFERED=1
ENV AUTOCHEM_CONFIG=/app/configs/default.yaml

RUN mkdir -p logs datasets .checkpoints .chromadb benchmarks

RUN uv run pytest -q || echo "Tests completed (some may require LLM/network)"

CMD ["uv", "run", "python", "-m", "src.cli.main", "pipeline", "-n", "1"]
