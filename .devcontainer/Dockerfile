FROM mcr.microsoft.com/devcontainers/python:3.14-bookworm

ENV UV_INSTALL_DIR="/usr/local/bin" \
    UV_LINK_MODE="copy"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
