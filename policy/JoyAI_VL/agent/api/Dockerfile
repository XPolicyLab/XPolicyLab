# Public rebuild of the lightweight agent runtime; see docs/reproducibility.md.
ARG BASE=python:3.11-slim-bookworm
FROM ${BASE}
RUN python3 -m pip install --no-cache-dir numpy==1.26.0 Pillow==11.3.0
RUN useradd -m -u 1000 agent && mkdir -p /work && chown -R agent:agent /work /home/agent
USER agent
WORKDIR /work
ENV ROBO_SERVER=http://egress:18700 ROBO_OBS_DIR=/work/obs PYTHONPATH=/opt/robo/client:/opt/robo PYTHONDONTWRITEBYTECODE=1
