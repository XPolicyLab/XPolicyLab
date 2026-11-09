# Services

The model processes that skills and model roles call: one manifest per service
(`<name>.toml`: kind, protocol, command or XPolicyLab policy, port, measured VRAM, client
config, weights). `servers/` holds the servers that are ours; a policy runs on
XPolicyLab's own server, so any XPolicyLab policy becomes a learned skill with one file.
Where they run is decided per machine by `configs/deploy/<machine>.toml`.

- How it works: [docs/concepts/services.md](../docs/concepts/services.md)
- Commands (`harness up`, `doctor`, `fetch`, ...): [docs/how-to/services.md](../docs/how-to/services.md)
- Add a VLA: [docs/tutorials/add-a-vla.md](../docs/tutorials/add-a-vla.md)
- Every service: the documentation site's *Reference → Services* (generated)
