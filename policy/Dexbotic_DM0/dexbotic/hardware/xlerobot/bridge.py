import argparse
import json
import logging
import pickle
import threading
import time
from collections import deque
from concurrent import futures
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Iterator, List

import cv2
import grpc
import numpy as np
import requests
import torch
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks

# Set global level to WARNING to silence library "Starting receiver" logs
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s %(asctime)s %(name)s: %(message)s')

# Set local logger to INFO to ensure Bridge-specific logs are visible
logger = logging.getLogger("Bridge")
logger.setLevel(logging.INFO)

ACTION_DIM = 16

CAMERA_NAMES = ("head", "wrist_left", "wrist_right")

STATE_FIELD_NAMES = (
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
    "head_motor_1.pos",
    "head_motor_2.pos",
    "x.vel",
    "theta.vel",
)


@dataclass
class TimedAction:
    timestamp: float
    timestep: int
    action: torch.Tensor


class InferenceClient:
    def __init__(self, vla_url: str, prompt: str):
        self.vla_url = vla_url
        self.prompt = prompt

    def request_actions(self, images: List[np.ndarray],
                        states: List[float]) -> np.ndarray:
        start_time = time.time()

        # 1. Image Encoding & Files Preparation
        files = []
        for i, image in enumerate(images):
            success, encoded_image = cv2.imencode('.png', image)
            if not success:
                logger.error(f"Failed to encode image from camera {CAMERA_NAMES[i]}")
                return []

            files.append(
                ("image",
                 (f"{CAMERA_NAMES[i]}.png",
                  encoded_image.tobytes(),
                  "image/png")))

        # 2. VLA Inference
        data = {"text": self.prompt}
        if states is not None:
            data["states"] = json.dumps(states)

        try:
            ret = requests.post(self.vla_url, data=data, files=files, timeout=10)

            if ret.status_code != 200:
                logger.error(f"Inference server error {ret.status_code}")
                return []

            # Expected response shape from VLA: [Batch, ChunkSize, ActionDim] (e.g.,
            # [1, 50, 16])
            actions = np.array(ret.json().get('response', []))

            if actions.size == 0:
                logger.error("Received empty actions from VLA")
                return []

            # Use verified shape (1, 50, 16) -> (50, 16)
            actions = actions[0]

            duration = time.time() - start_time
            logger.info(f"Received {actions.shape[0]} actions in {duration:.2f}s")

            return actions
        except Exception as e:
            logger.error(f"VLA Request Failed: {e}")
            return []


class BridgeService(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, inference_client: InferenceClient, show_images: bool = False):
        self.inference_client = inference_client
        self.show_images = show_images
        self.action_queue = deque()
        self.observation_queue = Queue(maxsize=1)
        self.vis_queue = Queue(maxsize=1)
        self.shutdown_event = threading.Event()
        self.step_counter = 0

    def Ready(self, request: services_pb2.Empty,
              context: grpc.ServicerContext) -> services_pb2.Empty:
        self.observation_queue = Queue(maxsize=1)
        while not self.vis_queue.empty():
            try:
                self.vis_queue.get_nowait()
            except Empty:
                break

        self.action_queue.clear()
        self.shutdown_event.clear()
        self.step_counter = 0
        logger.info("Robot Connected and Ready.")
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request: services_pb2.PolicySetup,
                               context: grpc.ServicerContext) -> services_pb2.Empty:
        return services_pb2.Empty()

    def SendObservations(self,
                         request_iterator: Iterator[services_pb2.Observation],
                         context: grpc.ServicerContext) -> services_pb2.Empty:
        data = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event)
        if data is None:
            return services_pb2.Empty()

        obs = pickle.loads(data)

        for cam in CAMERA_NAMES:
            obs.observation[cam] = cv2.cvtColor(obs.observation[cam], cv2.COLOR_RGB2BGR)

        self.observation_queue.put(obs)
        return services_pb2.Empty()

    def GetActions(self, request: services_pb2.Empty,
                   context: grpc.ServicerContext) -> services_pb2.Actions:
        try:
            # Heartbeat: Sync with robot observation
            timed_obs = self.observation_queue.get(timeout=2.0)
            raw_obs = timed_obs.observation
            images = [raw_obs[cam] for cam in CAMERA_NAMES]

            if not self.action_queue:
                # 1. State extraction
                current_state = []
                for k in STATE_FIELD_NAMES:
                    val = raw_obs.get(k, 0.0)
                    if hasattr(val, "item"):
                        val = val.item()
                    current_state.append(float(val))

                # 2. Inference
                actions = self.inference_client.request_actions(
                    images, states=current_state)
                if len(actions) == 0:
                    return services_pb2.Empty()

                # 3. Action Logic
                self.action_queue.extend(actions)

                # 4. Visualization
                if self.show_images:
                    canvas = np.hstack(images)
                    if self.vis_queue.full():
                        try:
                            self.vis_queue.get_nowait()
                        except Empty:
                            pass
                    self.vis_queue.put(canvas)

            if self.action_queue:
                act_np = self.action_queue.popleft()
                self.step_counter += 1
                t_action = TimedAction(
                    timestamp=timed_obs.timestamp,
                    timestep=timed_obs.timestep + 1,
                    action=torch.from_numpy(act_np).float()
                )
                return services_pb2.Actions(data=pickle.dumps([t_action]))

            return services_pb2.Empty()

        except Empty:
            return services_pb2.Empty()
        except Exception as e:
            logger.error(f"Error in GetActions: {e}", exc_info=True)
            return services_pb2.Empty()


def serve(port: int, vla_url: str, prompt: str, show_images: bool = False) -> None:
    inference_client = InferenceClient(vla_url, prompt)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    bridge = BridgeService(inference_client, show_images=show_images)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(bridge, server)
    server.add_insecure_port(f"[::]:{port}")
    logger.info(f"Bridge started on port {port}")
    server.start()

    try:
        if show_images:
            cv2.namedWindow("XLeRobot Bridge", cv2.WINDOW_AUTOSIZE)
            while not bridge.shutdown_event.is_set():
                try:
                    canvas = bridge.vis_queue.get(timeout=0.05)
                    cv2.imshow("XLeRobot Bridge", canvas)
                except Empty:
                    pass
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    bridge.shutdown_event.set()
                    break
            cv2.destroyAllWindows()
        else:
            server.wait_for_termination()

    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        bridge.shutdown_event.set()
        server.stop(0)
        logger.info("Bridge stopped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XLeRobot Inference Bridge")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--vla_url", type=str, default="http://localhost:7891")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Pick up scattered longans from the table and place them into the box")
    parser.add_argument(
        "--show_images",
        action="store_true",
        help="Show real-time camera feeds")
    args = parser.parse_args()
    serve(args.port, args.vla_url, args.prompt, show_images=args.show_images)
