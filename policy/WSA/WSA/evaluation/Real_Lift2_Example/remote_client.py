from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

try:
    from .request_builder import build_wsa_base_request, prepare_history_frame
    from .websocket_client import WebsocketClientPolicy
except ImportError:
    from request_builder import build_wsa_base_request, prepare_history_frame
    from websocket_client import WebsocketClientPolicy


class RealLift2RemoteClient:
    """Small helper for real-robot loops that query the WSABase websocket server."""

    def __init__(
        self,
        host: str,
        *,
        port: int | None = None,
        prompt: str = "Clear the junk and items off the desktop.",
        image_history_interval: int = 15,
        state_dim: int = 14,
        max_history: int | None = None,
        send_image_height: int | None = None,
        send_image_width: int | None = None,
    ) -> None:
        self._policy = WebsocketClientPolicy(host=host, port=port)
        self._prompt = prompt
        self._image_history_interval = image_history_interval
        self._state_dim = state_dim
        self._max_history = max_history or (image_history_interval + 1)
        self._send_image_height = send_image_height
        self._send_image_width = send_image_width
        self._image_histories: dict[str, list[np.ndarray]] = defaultdict(list)
        self._send_reset = True

    def _append_images_to_history(self, images: Mapping[str, Any]) -> None:
        for camera_name, image in images.items():
            prepared = prepare_history_frame(
                image,
                send_image_height=self._send_image_height,
                send_image_width=self._send_image_width,
            )
            self._image_histories[camera_name].append(prepared)
            if len(self._image_histories[camera_name]) > self._max_history:
                self._image_histories[camera_name].pop(0)

    @property
    def metadata(self) -> dict:
        return self._policy.get_server_metadata()

    def reset(self) -> None:
        self._image_histories.clear()
        self._send_reset = True

    def observe(self, images: Mapping[str, Any]) -> None:
        """Update local image history without sending a websocket request."""
        self._append_images_to_history(images)

    def close(self) -> None:
        self._policy.close()

    def infer_step(
        self,
        *,
        images: Mapping[str, Any],
        qpos: np.ndarray,
        timestep: int,
        prompt: str | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: np.ndarray | None = None,
        prev_chunk_left_over_processed: np.ndarray | None = None,
        update_history: bool = True,
    ) -> dict:
        if update_history:
            self._append_images_to_history(images)

        request = build_wsa_base_request(
            qpos=qpos,
            image_histories=self._image_histories,
            prompt=prompt or self._prompt,
            timestep=timestep,
            image_history_interval=self._image_history_interval,
            state_dim=self._state_dim,
            send_image_height=self._send_image_height,
            send_image_width=self._send_image_width,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            prev_chunk_left_over_processed=prev_chunk_left_over_processed,
        )
        request["reset"] = bool(self._send_reset or request["reset"])
        self._send_reset = False
        return self._policy.infer(request)
