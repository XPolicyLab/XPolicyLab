"""Transport-only WAM adapter. RGB/action/commit semantics remain unchanged."""
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import urllib.error
import urllib.request
import uuid


def identity(spec):
    data = json.dumps(spec, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf8")
    return hashlib.sha256(data).hexdigest()


class RemoteWam:
    def __init__(self, spec, deployment):
        from XPolicyLab.client_server.ws.protocol.codec import encode_frame, decode_frame
        self.encode, self.decode = encode_frame, decode_frame
        settings = json.loads(Path(os.environ["WRSI_POLICY_CLIENT_CONFIG"]).read_text())
        settings = settings.get("profiles", {}).get(spec["profile"], settings)
        self.url, self.token = settings["url"], settings["token"]
        if not self.url.startswith(("https://", "http://127.0.0.1:")):
            raise ValueError("Remote WAM requires HTTPS (loopback is allowed for local verification)")
        self.timeout = settings.get("timeout_s", 1200)
        self.profile = "memory" if spec["profile"] == "memory" else "flash"
        self.session = uuid.uuid4().hex
        self.trial = os.environ.get("WRSI_EPISODE_ID") or deployment["_wam_trial_id"]
        self.seq = 0
        self.instance = None
        self.closed = self.failed = False
        reply = self._exchange(dict(op="open", session=self.session, trial=self.trial,
                                    profile=self.profile, identity=identity(spec)))
        self.instance = reply["instance"]
        self._lease_stop = threading.Event()
        self.lease_token = reply.get("lease_token")
        self.lease_seconds = reply.get("lease_seconds", 0)
        if not self.lease_token or self.lease_seconds < 1:
            raise RuntimeError("Remote Policy requires the lease-capable service")
        if reply["identity"] != identity(spec):
            raise RuntimeError("Remote WAM returned an unpinned model identity")

        self._heartbeat_thread = threading.Thread(target=self._keep_lease, daemon=True)
        self._heartbeat_thread.start()

    def _keep_lease(self):
        while not self._lease_stop.wait(self.lease_seconds / 3):
            if self.closed or self.failed:
                return
            try:
                self._exchange(dict(op="heartbeat", instance=self.instance,
                                    session=self.session, lease_token=self.lease_token),
                               timeout=min(10, self.lease_seconds / 3))
            except Exception:
                self.failed = True
                return

    def _exchange(self, message, *, timeout=None):
        body = self.encode(message)
        for attempt in range(2):
            request = urllib.request.Request(self.url, body, headers={
                "Authorization": "Bearer "+self.token, "Content-Type": "application/msgpack"})
            try:
                with urllib.request.urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                    data = self.decode(response.read(16*1024*1024))
                if not data.get("ok"):
                    raise RuntimeError(data.get("error", "Remote WAM rejected the request"))
                return data.get("result")
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError):
                if attempt:
                    self.failed = True
                    raise
                # Exact same session/instance/sequence/body, served from the
                # server's in-flight request or cached reply; never new inference.
        raise AssertionError("unreachable")

    def _call(self, operation, *args):
        if self.closed or (self.failed and operation != "close"):
            raise RuntimeError("Uncertain or closed WAM cannot continue")
        self.seq += 1
        try:
            return self._exchange(dict(op=operation, args=list(args), seq=self.seq,
                                      instance=self.instance, session=self.session))
        except BaseException:
            self.failed = True
            raise

    def reset(self):
        return self._call("reset")

    def update_obs(self, observation):
        return self._call("update_obs", observation)

    def get_action(self):
        return self._call("get_action")

    def commit_action(self, acknowledgement):
        return self._call("commit_action", acknowledgement)

    def discard_action(self, disposition):
        return self._call("discard_action", disposition)

    def get_runtime_info(self):
        return {"requires_explicit_commit": True, "remote_policy": True,
                "instance": self.instance, "supports_batch": False}

    def close(self):
        if not self.closed:
            try:
                self._call("close")
            finally:
                self.closed = True
                self._lease_stop.set()
                self._heartbeat_thread.join(timeout=min(25, self.lease_seconds))
