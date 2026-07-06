# station

XPolicyLab-side evaluation station agent for the x-policy-web control plane.

This package bridges the web platform and on-machine eval clients:

- HTTP daemon (`station.daemon`) for dispatch/start/stop from x-policy-web
- WebSocket policy server (`client_server.ws.model_server`) for env ↔ policy communication
- Trial orchestration, artifact upload, and finish webhooks back to the web backend

It is not part of the XPolicyLab policy core. Install with:

```bash
pip install -e ".[station]"
```

Run the env client daemon:

```bash
python -m station.daemon --help
```
