"""Carry unchanged WebSocket bytes through a mounted Unix socket."""

import contextlib
import selectors
import socket
import socketserver
import threading


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        family, address = self.server.destination
        with socket.socket(family, socket.SOCK_STREAM) as peer:
            peer.connect(address)
            with selectors.DefaultSelector() as selector:
                selector.register(self.request, selectors.EVENT_READ, peer)
                selector.register(peer, selectors.EVENT_READ, self.request)
                while True:
                    for key, _ in selector.select():
                        chunk = key.fileobj.recv(1024 * 1024)
                        if not chunk:
                            return
                        key.data.sendall(chunk)


class UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class TcpServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True


@contextlib.contextmanager
def forward(address, destination, *, unix=False):
    with (UnixServer if unix else TcpServer)(address, Relay) as server:
        server.destination = destination
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address
        finally:
            server.shutdown()
            thread.join(timeout=5)
