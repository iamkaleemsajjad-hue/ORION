FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/orion
COPY packages ./packages
COPY pyproject.toml README.md ./
RUN python -m pip install --upgrade pip && \
    python -m pip install ./packages/orion-core ./packages/orion-tokenizer \
      ./packages/orion-data ./packages/orion-train ./packages/orion-eval \
      ./packages/orion-100m ./packages/orion-300m ./packages/orion-madad-adapter

RUN useradd --create-home --uid 10001 orion
USER orion
EXPOSE 8200
ENTRYPOINT ["orion300m"]
