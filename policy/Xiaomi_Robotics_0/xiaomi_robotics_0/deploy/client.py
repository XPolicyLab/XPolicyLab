# Copyright (C) 2026 Xiaomi Corporation.
import math
import os
import time
import pickle
import socket
import struct

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

torch.set_printoptions(3, sci_mode=False)


class Client:
    def __init__(self, host="localhost", port=10086):
        self.host = host
        self.port = port
        self._connect_with_retry(max_retries=None, retry_interval=1)
        print(f"Client connected to server at {self.host}:{self.port}.")

    def _connect_with_retry(self, max_retries=None, retry_interval=1):
        """Connect with retry logic. max_retries=None implies infinite."""
        retry_count = 0
        while True:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.host, self.port))
                return
            except (ConnectionRefusedError, socket.error) as e:
                retry_count += 1
                time.sleep(retry_interval)
                if max_retries is not None and retry_count >= max_retries:
                    raise ConnectionError(f"Failed to connect to {self.host}:{self.port} after {retry_count} retries: {e}") from e

    def _send_with_length_prefix(self, data):
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        self.client_socket.sendall(struct.pack(">I", len(serialized)) + serialized)

    def _recv_with_length_prefix(self):
        len_data = self.client_socket.recv(4)
        if not len_data or len(len_data) < 4:
            raise ConnectionError("Failed to receive response length prefix.")
        data_len = struct.unpack(">I", len_data)[0]
        data = b""
        while len(data) < data_len:
            packet = self.client_socket.recv(data_len - len(data))
            if not packet:
                raise ConnectionError("Connection closed while receiving response.")
            data += packet
        return pickle.loads(data)

    def __call__(self, **data):
        self._send_with_length_prefix(data)
        response = self._recv_with_length_prefix()

        return response

    def close(self):
        self.client_socket.close()
        print("Client connection closed.")
