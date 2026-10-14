ARG BASE_IMAGE=python:3.11-slim-bookworm
ARG NODE_IMAGE=node:22-bookworm-slim

FROM ${NODE_IMAGE} AS codex-cli
ARG CODEX_VERSION=0.159.2
ARG NPM_REGISTRY=https://registry.npmjs.org
RUN npm install --global --registry="${NPM_REGISTRY}" "@openai/codex@${CODEX_VERSION}" \
    && codex --version

FROM ${BASE_IMAGE}
ARG DEBIAN_MIRROR=https://deb.debian.org
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git ripgrep ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir numpy pillow imageio \
    && mkdir -p /workspace /home/agent
COPY --from=codex-cli /usr/local/bin/node /usr/local/bin/node
COPY --from=codex-cli /usr/local/lib/node_modules/@openai/ /usr/local/lib/node_modules/@openai/
RUN ln -s /usr/local/lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex \
    && codex --version \
    && python -c 'import numpy, PIL, imageio'
WORKDIR /workspace
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
CMD ["codex", "--help"]
