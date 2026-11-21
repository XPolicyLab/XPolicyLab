"""Expose the 1.5 robot API inside Preview's existing Python kernel."""

import os

import robo
from hexashell.robot.proxy import Blocked, Dropped, NotExecuted, Robot, RobotError
from hexashell.robot.results import NotFound, decode, encode
from robolib import EpisodeOver


cell_images = None


def call(name, args, kwargs):
    try:
        reply = robo.post({"cmd": "robot15", "name": name, "args": encode(args),
                           "kwargs": encode(kwargs), "structured": cell_images is not None})
    except OSError as exc:
        raise RobotError(f"robot.{name}: no reply from robo-server ({exc})") from exc
    if "error" in reply:
        errors = {"Blocked": Blocked, "Dropped": Dropped, "EpisodeOver": EpisodeOver,
                  "NotExecuted": NotExecuted, "NotFound": NotFound,
                  "TypeError": TypeError, "ValueError": ValueError}
        raise errors.get(reply.get("kind"), RobotError)(reply["error"])
    if "files" in reply:
        captions = reply.get("captions", {})
        robo.write_files(reply)
        for path in reply["files"]:
            if cell_images is None:
                print(f"View {os.path.relpath(path)} with the image viewing tool.")
            else:
                cell_images.append({"path": path, "caption": captions.get(os.path.basename(path), "")})
    return decode(reply["result"])


robot = Robot(call)
