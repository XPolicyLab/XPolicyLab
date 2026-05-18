import argparse
import pickle
import sys
import queue
import threading
import time
from pathlib import Path
from threading import Thread
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import APIRouter, FastAPI, Response, Query, Body
from fastapi.responses import JSONResponse
from loguru import logger

from mock_rc_robot import MockRCRobot
import mock_settings
from mock_settings import REALSENSE_DEVICE_IDS

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_config import get_robot_type, ROBO_CHALLENGE_TASKS

from utils import resize_with_pad_single

HOME_POSITION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

cmd_Q = queue.Queue()

latest_images = {
    "images": {"high": None, "right": None, "left": None},
    "lock": threading.Lock()
}


def make_jsonable(obj):
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


class FlaskWorker(Thread):
    def __init__(self, server_port, robot_alpha: MockRCRobot, dashboard_instance: 'RobotDashboard'):
        super().__init__()
        self.server_port = server_port
        self.image_size = (224, 224)  # (w,h)
        # image_type['left_hand','right_hand','high']
        self.action_type = None  # 'pos','joint','leftpos','rightpos','leftjoint','rightjoint'
        self.robot_alpha = robot_alpha
        self.dashboard_instance = dashboard_instance
        self.router = APIRouter()
        self.router.add_api_route('/clock-sync', self.clock_sync, methods=['GET'])
        self.router.add_api_route('/state.pkl', self.get_state, methods=['GET'])
        self.router.add_api_route('/action', self.post_action, methods=['POST'])
        self.router.add_api_route('/start_motion', dashboard_instance.start_motion, methods=['GET'])
        self.router.add_api_route('/end_motion', dashboard_instance.end_motion, methods=['GET'])
        self.app = FastAPI()
        self.app.include_router(self.router)
        self.idx = 0

    def clock_sync(self):
        t2 = time.time()
        return {'timestamp': t2}

    def get_action(self,idx, action_type):
        if 'left' in action_type:
            if self.robot_alpha.left_get_enable():
                if 'pos' in action_type:
                    action = self.robot_alpha.left_get_pose(idx)
                else:
                    action = self.robot_alpha.left_get_joint(idx)
                arm_state = "normal"
            else:
                action = None
                arm_state = "abnormal"
        elif 'right' in action_type:
            if self.robot_alpha.right_get_enable():
                if 'pos' in action_type:
                    action = self.robot_alpha.right_get_pose(idx)
                else:
                    action = self.robot_alpha.right_get_joint(idx)
                arm_state = "normal"
            else:
                print('187: self.robot_alpha.right_get_enable() is False')
                action = None
                arm_state = "abnormal"
        else:
            if self.robot_alpha.left_get_enable() and self.robot_alpha.right_get_enable():
                if 'pos' in action_type:
                    action = self.robot_alpha.left_get_pose(idx) + self.robot_alpha.right_get_pose(idx)
                else:
                    action = self.robot_alpha.left_get_joint(idx) + self.robot_alpha.right_get_joint(idx)
                arm_state = "normal"
            else:
                print('196: self.robot_alpha.left_get_enable() and self.robot_alpha.right_get_enable() are both False')
                action = None
                arm_state = "abnormal"
        return action, arm_state

    def get_state(self, width: int = 224, height: int = 224, image_type: List[str] = Query(default=None), action_type: str = None, resize_name: str = None):
        self.action_type = action_type
        image_size = (width, height)
        if image_size is None or action_type is None:
            state_data = {
                "state": "size_none"
            }
            return Response(pickle.dumps(state_data), media_type='application/octet-stream')
        self.idx = min(self.idx, self.robot_alpha.get_frame_number() - 1)
        ret_imgs = self.robot_alpha.get_imgs(self.idx)
        images_dict = {}
        if resize_name == 'padding':
            resize_method = resize_with_pad_single
        else:
            resize_method = cv2.resize
        if 'high' in image_type:
            img_high = resize_method(ret_imgs[2][1], image_size)
            images_dict['high'] = cv2.imencode('.png', img_high)[-1].tobytes()
        if 'left_hand' in image_type:
            img_left = resize_method(ret_imgs[0][1], image_size)
            images_dict['left_hand'] = cv2.imencode('.png', img_left)[-1].tobytes()
        if 'right_hand' in image_type:
            img_right = resize_method(ret_imgs[1][1], image_size)
            images_dict['right_hand'] = cv2.imencode('.png', img_right)[-1].tobytes()
        
        action, arm_state = self.get_action(self.idx, action_type)
        gt_actions = []
        for i in range(self.idx+1, self.idx + 51):
            idx_action, idx_arm_state = self.get_action(min(i, self.robot_alpha.get_frame_number() - 1), action_type)
            gt_actions.append(idx_action)
        state_data = {
            "images": images_dict,
            "action": action,
            "pending_actions": cmd_Q.qsize(),
            "timestamp": time.time(),
            "state": arm_state,
            "gt_actions": gt_actions,
        }
        self.idx += 1
        return Response(content=pickle.dumps(state_data), media_type='application/octet-stream')

    def post_action(self, data: dict = Body(...), action_type: str = None):
        try:
            actions = data.get("actions", [])
            duration = data.get("duration", [])
            for action in actions:
                if not cmd_Q.full():
                    if 'left' in action_type:
                        if 'pos' in action_type:
                            if len(action) == self.robot_alpha.pos_num + 1:
                                cmd_Q.put({'left_action': np.array(action, dtype=np.float32), 'right_action': None, 'duration': duration, 'action_type': action_type})
                            else:
                                self.dashboard_instance.format_error()
                                return {'result': "error", 'message': 'The number and type of actions do not match!'}
                        else:
                            if len(action) == self.robot_alpha.dof_num + 1:
                                cmd_Q.put({'left_action': np.array(action, dtype=np.float32), 'right_action': None, 'duration': duration, 'action_type': action_type})
                            else:
                                self.dashboard_instance.format_error()
                                return {'result': "error", 'message': 'The number and type of actions do not match!'}
                    elif 'right' in action_type:
                        if 'pos' in action_type:
                            if len(action) == self.robot_alpha.pos_num + 1:
                                cmd_Q.put({'left_action': None, 'right_action': np.array(action, dtype=np.float32), 'duration': duration, 'action_type': action_type})
                            else:
                                self.dashboard_instance.format_error()
                                return {'result': "error", 'message': 'The number and type of actions do not match!'}
                        else:
                            if len(action) == self.robot_alpha.dof_num + 1:
                                cmd_Q.put({'left_action': None, 'right_action': np.array(action, dtype=np.float32), 'duration': duration, 'action_type': action_type})
                            else:
                                self.dashboard_instance.format_error()
                                return {'result': "error", 'message': 'The number and type of actions do not match!'}
                    else:
                        if 'pos' in action_type:
                            if len(action) == (self.robot_alpha.pos_num + 1) * 2:
                                cmd_Q.put(
                                    {'left_action': np.array(action[:self.robot_alpha.pos_num + 1], dtype=np.float32), 'right_action': np.array(action[self.robot_alpha.pos_num + 1:], dtype=np.float32), 'duration': duration, 'action_type': action_type})
                            else:
                                self.dashboard_instance.format_error()
                                return {'result': "error", 'message': 'The number and type of actions do not match!'}
                        else:
                            if len(action) == (self.robot_alpha.dof_num + 1) * 2:
                                cmd_Q.put(
                                    {'left_action': np.array(action[:self.robot_alpha.dof_num + 1], dtype=np.float32), 'right_action': np.array(action[self.robot_alpha.dof_num + 1:], dtype=np.float32), 'duration': duration, 'action_type': action_type})
                            else:
                                self.dashboard_instance.format_error()
                                return {'result': "error", 'message': 'The number and type of actions do not match!'}
                else:
                    return {"result": "error", "message": "Queue full"}
            return {"result": "success"}
        except Exception as e:
            logger.info(f"Error in post_action: {e}")
            return JSONResponse({"result": "error", "message": str(e)}, status_code=400)

    def run(self):
        uvicorn.run(self.app, host="0.0.0.0", port=self.server_port, access_log=False)


class RobotWorker(Thread):
    def __init__(self, velocity_thre, robot_alpha: MockRCRobot, dashboard_instance: 'RobotDashboard'):
        super().__init__()
        self.running = True
        self.current_position = HOME_POSITION.copy()
        self.velocity_thre = velocity_thre
        self.robot_alpha = robot_alpha
        self.dashboard_instance = dashboard_instance

    def run(self):
        period = 1 / 20
        while self.running:
            t_cur = time.time()
            try:
                action_duration = cmd_Q.get()
                if action_duration['left_action'] is not None:
                    try:
                        if self.robot_alpha.left_get_enable():
                            if 'pos' in action_duration['action_type']:
                                self.robot_alpha.left_go_pose(action_duration['left_action'])
                            else:
                                self.robot_alpha.left_go_joint(action_duration['left_action'])
                        else:
                            print("left arm fault")
                    except Exception as e:
                        print(f"left arm joint movement failed: {e}")
                if action_duration['right_action'] is not None:
                    try:
                        if self.robot_alpha.right_get_enable():
                            if 'pos' in action_duration['action_type']:
                                self.robot_alpha.right_go_pose(action_duration['right_action'])
                            else:
                                self.robot_alpha.right_go_joint(action_duration['right_action'])
                        else:
                            print("right arm fault")
                    except Exception as e:
                        print(f"right arm joint movement failed: {e}")
                last_pos = self.current_position.copy()
                if action_duration['left_action'] is not None and action_duration['right_action'] is None:
                    self.current_position = np.array(action_duration['left_action'])
                elif action_duration['left_action'] is None and action_duration['right_action'] is not None:
                    self.current_position = np.array(action_duration['right_action'])
                elif action_duration['left_action'] is not None and action_duration['right_action'] is not None:
                    self.current_position = np.concatenate((action_duration['left_action'], action_duration['right_action']))
                if last_pos.shape!= self.current_position.shape:
                    last_pos=self.current_position.copy()
                period = action_duration['duration']
                velocity = (self.current_position - last_pos) / period
                if np.abs(velocity).max() > self.velocity_thre:
                    period *= np.abs(velocity[:-1]).max() / self.velocity_thre

            except queue.Empty:
                continue
            except Exception as e:
                logger.info(f"Error in goto_worker: {e}")
            t_end = time.time()
            if t_end - t_cur < period:
                time.sleep(period - (t_end - t_cur))
            else:
                print(f"run time:{t_end - t_cur} {period}")

    def stop(self):
        self.running = False


class RobotDashboard:
    def __init__(
        self,
        server_port=8083,
        velocity_thre=1000000,
        robot_tag: Optional[str] = None,
        record_data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.start_time = time.time()
        self.max_speed_seen = 0.0
        self.duration = 1 / 30
        self.current_joint_state = HOME_POSITION.copy()
        self.server_port = server_port
        self.velocity_thre = velocity_thre

        self.frpc_thread = None
        self.logging_start_time: Optional[float] = None

        tag = robot_tag
        data_dir = record_data_dir
        assert tag is not None and data_dir is not None, f"tag={tag}, data_dir={data_dir}"
        self.robot_alpha = MockRCRobot.create_robot(tag, REALSENSE_DEVICE_IDS, record_data_dir=data_dir)

        self.setup_threads()
        time.sleep(5.0)

    def setup_threads(self):

        self.flask_worker = FlaskWorker(self.server_port, self.robot_alpha, self)
        self.flask_worker.daemon = True
        self.flask_worker.start()

        self.robot_worker = RobotWorker(self.velocity_thre, self.robot_alpha, self)
        self.robot_worker.start()

    def get_image(self, number: int = 0):
        ret_imgs = self.robot_alpha.get_imgs()
        if number < len(ret_imgs) and ret_imgs[number][1] is not None:
            img = ret_imgs[number][1]
            img_bytes = cv2.imencode('.png', img)[-1].tobytes()
            return Response(content=img_bytes, media_type='image/png')
        else:
            return Response('image number out of bound', status_code=404)

    def start_motion(self):
        with cmd_Q.mutex:
            cmd_Q.queue.clear()

    def end_motion(self):
        logger.info('end_motion')
        with cmd_Q.mutex:
            cmd_Q.queue.clear()

    def format_error(self):
        logger.warning('format error, stopping')
        self.end_motion()


if __name__ == "__main__":
    logger.info('starting server')
    parser = argparse.ArgumentParser(description="Robot Dashboard Arguments")
    parser.add_argument('-s', '--server_port', type=int, default=9098)
    parser.add_argument(
        '--task_name',
        type=str,
        help='Task name. Robot type from task_config, data path from mock_settings.TASK_DATA_DIRS.',
    )
    args = parser.parse_args()

    robot_tag = None
    record_data_dir = None
    if args.task_name:
        if get_robot_type is None:
            raise RuntimeError("task_config not found.")
        robot_tag = get_robot_type(args.task_name)
        if not robot_tag:
            raise ValueError(f"Unknown task_name: {args.task_name}. Valid: {list(ROBO_CHALLENGE_TASKS.keys())}")
        record_data_dir = getattr(mock_settings, "TASK_DATA_DIRS", {}).get(args.task_name)
        if not record_data_dir:
            raise ValueError(
                f"task_name '{args.task_name}' not in mock_settings.TASK_DATA_DIRS. "
                "Add it with your data path."
            )
        logger.info(f"task_name={args.task_name} -> robot_tag={robot_tag}, record_data_dir={record_data_dir}")

    dashboard = RobotDashboard(
        server_port=args.server_port,
        robot_tag=robot_tag,
        record_data_dir=record_data_dir,
    )
