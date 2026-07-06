from .utils import *
import socket
import time

class ModelClient:
    def __init__(self, host="localhost", port=9999, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = 1000
        retry_delay = 5

        while attempts < max_attempts:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                attempts += 1
                if self.sock:
                    self.sock.close()
                if attempts < max_attempts:
                    print(f"⚠️ Connection attempt {attempts} failed: {str(e)}")
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(f"Failed to connect to server after {max_attempts} attempts: {str(e)}")

    def _send(self, data):
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode("utf-8")

            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, "big"))
            self.sock.sendall(json_data)

        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _send_recv(self, data):
        """Send request and receive response with numpy array support"""
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode("utf-8")

            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, "big"))
            self.sock.sendall(json_data)
            # Receive and deserialize response
            response = self._recv_response()
            return response

        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _recv_response(self):
        """Receive response with numpy array reconstruction"""
        # Read response length
        
        len_data = self.sock.recv(4)

        if not len_data:
            raise ConnectionError("Connection closed by server")

        size = int.from_bytes(len_data, "big")

        # Read complete response
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)
        # Deserialize with numpy reconstruction
        return json_to_numpy(b"".join(chunks).decode("utf-8"))

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        if "res" in response.keys():
            return response["res"]
        return None

    def close(self):
        """Close the connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
                print("🔌 Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    ModelClient()