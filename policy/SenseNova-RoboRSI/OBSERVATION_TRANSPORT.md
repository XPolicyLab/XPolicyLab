# Observation transport alignment

The native evaluation observation path already transports raw RGB arrays
losslessly. It does not call `encode_image_bit` before RPC:

1. Official `env/observation_manager/obs_manager.py`, `ObsManager.get_obs`,
   takes the renderer's RGB annotator array and retains the first three channels.
2. Official `src/eval_client/eval_env.py`, `get_obs_batch`, copies those arrays;
   `get_obs` delegates to the same batch path for environment 0.
3. This policy's `deploy.py` forwards that observation with
   `model_client.call("update_obs", ...)`.
4. `client_server/ws/model_client.py`, `WsModelClient.call`, forwards the array
   to the protocol client. `client_server/ws/protocol/codec.py` serializes it
   with `msgpack_numpy.encode`, preserving shape, dtype, channel order and bytes.
5. `client_server/ws/model_server.py`, `PolicyServer._handle_call`, invokes
   `decode_obs_images` before the model. That shared helper passes already
   decoded arrays through unchanged.

The controller's image receipts are PNG. The shared `encode_image_bit` helper
still creates marked JPEG buffers when a caller explicitly encodes images,
including `DEBUG_OBS_ENCODED=1` fixtures and dataset storage. JPEG quality 100
does not guarantee pixel equality. That optional encoded path must not be
confused with the native raw observation RPC above; legacy and marked JPEGs
continue to decode only through the shared helper.

`tests/test_observation_transport.py` runs the real synchronous client, protocol
framing and server dispatch against an in-memory transport. It checks exact
pixel equality for nonuniform, strided RGB frames, multiple cameras and batch
observations, and verifies both JPEG formats retain shared decoder semantics.
It opens no sockets and does not run the simulator.
