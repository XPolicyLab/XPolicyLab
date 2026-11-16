ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}
ARG DEBIAN_MIRROR=https://deb.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git ripgrep ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy==2.5.3 pillow==12.3.0 imageio==2.38.0 \
    && mkdir -p /workspace /home/agent
COPY codex.sha256 /tmp/codex.sha256
# Official release packages include the CLI, its code-mode host and resources.
# Keep the package layout so helper discovery does not depend on the host.
RUN set -eux; \
    case "$(uname -m)" in x86_64) arch=x86_64 ;; aarch64) arch=aarch64 ;; *) exit 2 ;; esac; \
    archive="codex-package-${arch}-unknown-linux-musl.tar.gz"; \
    cd /tmp; \
    curl --fail --location --retry 3 --output "$archive" \
      "https://github.com/openai/codex/releases/download/rust-v0.159.2/$archive"; \
    sha256sum --check --ignore-missing codex.sha256; \
    mkdir -p /opt/codex; \
    tar -xzf "$archive" -C /opt/codex; \
    ln -s /opt/codex/bin/codex /usr/local/bin/codex; \
    test "$(codex --version)" = 'codex-cli 0.159.2'; \
    rm "$archive" codex.sha256
LABEL org.opencontainers.image.source="https://github.com/openai/codex" \
      org.opencontainers.image.version="0.159.2"
WORKDIR /workspace
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
