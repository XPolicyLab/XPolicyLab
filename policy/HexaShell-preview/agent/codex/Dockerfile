# Agent container: Codex + robo client + interface manual. No simulator, no task code, no API key.
ARG BASE=debian:bookworm-slim
FROM ${BASE}

ARG APT_MIRROR=deb.debian.org
ARG WITH_PYTHON_TOOLS=1
RUN sed -i "s#deb.debian.org#${APT_MIRROR}#g" /etc/apt/sources.list.d/debian.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3 ca-certificates \
 && if [ "${WITH_PYTHON_TOOLS}" = "1" ]; then apt-get install -y --no-install-recommends python3-numpy; fi \
 && rm -rf /var/lib/apt/lists/*

COPY codex-bin/ /usr/local/bin/
COPY robo.py /opt/robo/robo.py
RUN printf '#!/bin/sh\nexec python3 /opt/robo/robo.py "$@"\n' > /usr/local/bin/robo && chmod +x /usr/local/bin/robo /usr/local/bin/codex /usr/local/bin/codex-code-mode-host

RUN useradd -m -u 1000 agent && mkdir -p /work && chown -R agent:agent /work /home/agent
USER agent
WORKDIR /work
ENV HOME=/home/agent ROBO_SERVER=http://egress:18700 ROBO_OBS_DIR=/work/obs GATEWAY_PLACEHOLDER=unused
