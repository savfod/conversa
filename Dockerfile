FROM python:3.13
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app

RUN apt-get update &&\
    apt-get install -y\
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

ADD . /app
CMD ["sh", "-c", "uv run -m conversa.web.web en --host 0.0.0.0 --port 8080 --debug"]
