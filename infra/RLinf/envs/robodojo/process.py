"""Private inherited-pipe RPC, independent of Ray/Torch/Isaac imports.

Only the child endpoint can send pickle messages. No TCP listener or shell is
used, and simulator logging cannot corrupt the dedicated binary channel.
"""

from __future__ import annotations

import importlib
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys
import traceback


class SimulatorProcess:
    def __init__(
        self,
        python,
        root,
        config,
        *,
        timeout=300,
        render_gpu=None,
        factory="infra.RLinf.envs.robodojo.runtime:SimulatorBackend",
    ):
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("Simulator timeout must be positive.")
        kit_args = []
        if render_gpu is not None:
            if type(render_gpu) is not int or render_gpu < 0:
                raise ValueError("render_gpu must be a nonnegative Kit device index.")
            # Kit enumerates graphics devices separately from masked CUDA ordinals.
            kit_args.append(f"--/renderer/activeGpu={render_gpu}")
        self.connection, child = Pipe(duplex=True)
        env = os.environ.copy()
        root = Path(root).resolve()
        # RoboDojo owns env/src/utils. Load infra and client_server from the
        # active adapter even when the simulator has another submodule revision.
        adapter_root = Path(__file__).resolve().parents[4]
        env["PYTHONPATH"] = os.pathsep.join((str(root), str(adapter_root)))
        env.pop("PYTHONHOME", None)
        self.process = None
        try:
            self.process = subprocess.Popen(
                [
                    str(python),
                    "-P",  # Keep the adapter's env.py off the host's import path.
                    str(Path(__file__).resolve()),
                    str(child.fileno()),
                    factory,
                    *kit_args,
                ],
                cwd=root,
                env=env,
                pass_fds=(child.fileno(),),
            )
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()
        try:
            self.metadata = self.call("initialize", config)
        except BaseException:
            self.close(force=True)
            raise

    def call(self, operation, *args):
        try:
            self.connection.send((operation, args))
            if not self.connection.poll(self.timeout):
                raise TimeoutError(
                    f"RoboDojo {operation} timed out after {self.timeout}s."
                )
            ok, result = self.connection.recv()
            if not ok:
                raise RuntimeError(f"RoboDojo {operation} failed:\n{result}")
            return result
        except TimeoutError:
            self.close(force=True)
            raise
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.close(force=True)
            raise RuntimeError(
                f"RoboDojo process disconnected during {operation}."
            ) from exc
        except BaseException:
            self.close(force=True)
            raise

    def reset(self, layout_ids):
        return self.call("reset", layout_ids)

    def step(self, actions, env_indices):
        return self.call("step", actions, env_indices)

    def close(self, force=False):
        process = self.process
        if process is None:
            self.connection.close()
            return
        if process.poll() is None:
            if not force:
                try:
                    self.connection.send(("close", ()))
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    force = True
            if force and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self.connection.close()


def serve(fd, factory):
    connection = Connection(fd)
    backend = None
    try:
        while True:
            operation, args = connection.recv()
            if operation == "close":
                break
            try:
                if operation == "initialize" and backend is None:
                    module, name = factory.split(":")
                    backend = getattr(importlib.import_module(module), name)(*args)
                    result = backend.metadata()
                elif backend is not None and operation in ("reset", "step"):
                    result = getattr(backend, operation)(*args)
                else:
                    raise ValueError(f"Invalid simulator operation: {operation}")
                connection.send((True, result))
            except BaseException:
                connection.send((False, traceback.format_exc()))
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if backend is not None:
            backend.close()
        connection.close()


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2])
