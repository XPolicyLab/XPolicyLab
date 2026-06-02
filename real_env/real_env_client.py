import os
import time
import random
import importlib
from .constants                            import XONE_ROOT
from .data_handler                         import camera_meta, create_move_data, build_state, load_yaml, str_to_bool
from robot.robot                           import get_robot
from threading                             import Event, Lock
from XPolicyLab.client_server.model_client import ModelClient

class RealEnv:
    def __init__(self, deploy_cfg):
        self.m_base_cfg = load_yaml(XONE_ROOT / "config" / f"{deploy_cfg['base_cfg']}.yml")
        self.m_base_cfg.pop("collect", None) # 评测时去掉 collect 段，避免创建 CollectAny 实例
        self.m_robot = get_robot(base_cfg=self.m_base_cfg)

        self.m_task_info     = load_yaml(XONE_ROOT / "task_info" / f"{deploy_cfg['task_name']}.json")
        self.m_deploy_cfg    = deploy_cfg
        self.m_robot_lock    = Lock()
        self.m_model_client  = ModelClient(host=deploy_cfg["host"], port=deploy_cfg["port"])
        self.m_current_step  = 0
        self.m_instruction   = random.choice(self.m_task_info.get("instructions") or [self.m_deploy_cfg["task_name"]])
        self.m_stop_event    = Event()
        self.m_stop_reason   = ""

        self.m_robot.set_up(teleop=False)
        
    @property
    def deploy_cfg(self):
        return self.m_deploy_cfg

    @property
    def task_info(self):
        return self.m_task_info

    @property
    def should_stop(self):
        return self.m_stop_event.is_set()

    @property
    def stop_reason(self):
        return self.m_stop_reason

    def request_stop(self, reason="operator stop"):
        self.m_stop_reason = str(reason or "operator stop")
        self.m_stop_event.set()

    def clear_stop(self):
        self.m_stop_reason = ""
        self.m_stop_event.clear()

    def get_obs(self, env_idx=0):
        with self.m_robot_lock:
            controller_data, sensor_data = self.m_robot.get_obs()

        if not sensor_data:
            raise ValueError("[RealEnv::get_obs()] sensor_data is None/Empty.")
        if not controller_data:
            raise ValueError("[RealEnv::get_obs()] controller_data is None/Empty.")

        obs = {
            "data_format_version": "v1.0",
            "instruction": self.m_instruction,
            "env_idx": env_idx,
            "additional_info": {
                "frequency": self.m_base_cfg.get("frequency", 30)
            },
            "vision": {
                cam_name: camera_meta(
                    self.m_base_cfg.get("robot", {}).get("camera_info", {}),
                    cam_name,
                    cam_data,
                )
                for cam_name, cam_data in sensor_data.items()
                if cam_data and cam_data.get("color")
            },
            "state": build_state(controller_data),
        }
        
        if not obs["vision"]:
            raise ValueError("[RealEnv::get_obs()] vision is Empty.")
        return obs
    
    def get_obs_batch(self, env_idx_list):
        return [self.get_obs(env_idx) for env_idx in env_idx_list]

    def eval_one_episode(self):
        policy_name = self.m_deploy_cfg["policy_name"]
        eval_module = importlib.import_module(f"policy.{policy_name}.deploy")
        eval_module.eval_one_episode(TASK_ENV=self, model_client=self.m_model_client)

    def eval_one_episode_batch(self):
        policy_name = self.m_deploy_cfg["policy_name"]
        eval_module = importlib.import_module(f"policy.{policy_name}.deploy")
        eval_module.eval_one_episode_batch(TASK_ENV=self, model_client=self.m_model_client)

    def take_action(self, action):
        if self.is_episode_end():
            return
        
        move_data = create_move_data(action)
        with self.m_robot_lock:
            if self.should_stop:
                return
            self.m_robot.move(move_data)
            if self.m_deploy_cfg.get("force_reach_mode"):
                while self.m_robot.is_move():
                    if self.should_stop:
                        return
                    time.sleep(0.01)
            else:
                time.sleep(1 / self.m_base_cfg.get("frequency", 30))
        self.m_current_step += 1
    
    def take_action_batch(self, action_list, env_idx_list):
        if len(action_list) != 1 or len(env_idx_list) != 1:
            raise NotImplementedError("batch evaluation only supports batch size 1.")
        self.take_action(action_list[0])
    
    def finish_episode(self):
        self.reset_robot()
        self.m_model_client.call(func_name="reset")
        self.m_current_step = 0
        self.m_instruction = random.choice(
            self.m_task_info.get("instructions") or [self.m_deploy_cfg["task_name"]]
        )

    def reset_robot(self) -> None:
        self.clear_stop()
        with self.m_robot_lock:
            self.m_robot.reset()
            time.sleep(2.0)

    def reset(self):
        self.reset_robot()
        self.m_model_client.call(func_name="reset")
        self.m_current_step = 0
        self.m_instruction = random.choice(
            self.m_task_info.get("instructions") or [self.m_deploy_cfg["task_name"]]
        )

    def get_running_env_idx_list(self):
        return [0]
    
    def is_episode_end(self):
        return self.should_stop or self.m_current_step >= self.m_task_info["step_lim"]

    def close(self):
        self.m_model_client.close()
