import argparse
from client_server.model_client import ModelClient
import numpy as np

Batch_Size = 1

class TestEnv:
    def __init__(self, deploy_cfg):
        self.success_num, self.episode_num = 0, 0
        self.deploy_cfg = deploy_cfg
        self.episode_step_limit = 5
        
        self.model_client = ModelClient(port=deploy_cfg['port'])

    def get_obs(self, env_idx_list):
        # v1.0
        demo_obs_list = []
        for i in range(Batch_Size):
            demo_obs = { # aloha
                "vision": {
                    "cam_head": {
                        "color": np.zeros((480, 640, 3), dtype=np.uint8),
                        "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                        "intrinsic_matrix": [
                            [615.0, 0.0, 320.0],
                            [0.0, 615.0, 240.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "extrinsics_matrix": [
                            [1.0, 0.0, 0.0, 0.10],
                            [0.0, 1.0, 0.0, 1.20],
                            [0.0, 0.0, 1.0, 1.50],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "shape": (480, 640),
                    },
                    "cam_left_wrist": {
                        "color": np.zeros((480, 640, 3), dtype=np.uint8),
                        "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                        "intrinsic_matrix": [
                            [615.0, 0.0, 320.0],
                            [0.0, 615.0, 240.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "extrinsics_matrix": [
                            [1.0, 0.0, 0.0, 0.10],
                            [0.0, 1.0, 0.0, 1.20],
                            [0.0, 0.0, 1.0, 1.50],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "shape": (480, 640),
                    },
                    "cam_right_wrist": {
                        "color": np.zeros((480, 640, 3), dtype=np.uint8),
                        "depth": np.zeros((480, 640, 3), dtype=np.uint8),
                        "intrinsic_matrix": [
                            [615.0, 0.0, 320.0],
                            [0.0, 615.0, 240.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "extrinsics_matrix": [
                            [1.0, 0.0, 0.0, 0.10],
                            [0.0, 1.0, 0.0, 1.20],
                            [0.0, 0.0, 1.0, 1.50],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "shape": (480, 640),
                    },
                },

                "state": {
                    "left_arm_joint_state": np.zeros((7), dtype=np.uint8), 
                    "left_ee_joint_state": np.zeros((1), dtype=np.uint8), 
                    "left_ee_pose": np.zeros((7), dtype=np.uint8), 
                    "left_tcp_pose": np.zeros((7), dtype=np.uint8), 
                    "left_delta_ee_pose": np.zeros((7), dtype=np.uint8), 

                    "right_arm_joint_state": np.zeros((7), dtype=np.uint8), 
                    "right_ee_joint_state": np.zeros((1), dtype=np.uint8), 
                    "right_ee_pose": np.zeros((7), dtype=np.uint8), 
                    "right_tcp_pose": np.zeros((7), dtype=np.uint8), 
                    "right_delta_ee_pose": np.zeros((7), dtype=np.uint8), 

                    "mobile": {
                        "base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # x,y,z + quat
                        "base_twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],      # vx,vy,vz, wx,wy,wz
                    },
                },

                "additional_info": {
                    "frequency": 30,  # Hz
                },

                "data_format_version": "v1.0",
                "env_idx": i
            }
            demo_obs_list.append(demo_obs)
        demo_obs_list = [demo_obs_list[i] for i in env_idx_list] 
        return demo_obs_list

    def eval_one_episode(self):
        policy_name = self.deploy_cfg['policy_name']
        try:
            eval_module = __import__(f'XPolicyLab.{policy_name}.deploy', fromlist=['eval_one_episode_batch'])
        except ImportError as e:
            print("[TestEnv]", f"Failed to import policy module: XPolicyLab.{policy_name}.deploy. Error: {e}", "ERROR")
            raise e
            
        if not hasattr(eval_module, 'eval_one_episode_batch'):
            print("[TestEnv]", f"Module '.{policy_name}.deploy' does not have 'eval_one_episode_batch' function", "ERROR")
            raise AttributeError(f"Missing eval_one_episode_batch in policy module")
            
        eval_module.eval_one_episode_batch(TASK_ENV=self, model_client=self.model_client)

    def reset(self):
        self.model_client.call(func_name="reset")
        self.episode_step = 0

    def get_instruction(self):
        instruction = "Language instruction for the task"  # Replace with actual instruction retrieval logic
        print("[TestEnv] Get Instruction:", instruction)
        return instruction

    def take_action(self, action):
        print(f"[TestEnv] Action Step: {self.episode_step} / {self.episode_step_limit} (step_limit)")
        self.episode_step += 1
        # check action validity here if needed

    def is_episode_end(self):
        print("[TestEnv] Check Episode End:", self.episode_step >= self.episode_step_limit)
        return self.episode_step >= self.episode_step_limit
    
    def finish_episode(self):
        print("[TestEnv] Episode finished")
    
    def get_running_env_idx_list(self):
        # For demonstration, we assume all envs are running. Replace with actual logic if needed.
        return list(range(Batch_Size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--env_cfg", type=str, required=True)
    parser.add_argument("--policy_name", type=str, required=True, help="XPolicyLab module name for deployment")
    parser.add_argument("--port", type=int, required=True, help="server port")
    parser.add_argument("--eval_episode_num", type=int, default=100, help="number of evaluation episodes")

    args_cli = parser.parse_args()
    deploy_cfg = vars(args_cli)          # 或 args_cli.__dict__.copy()
    test_env = TestEnv(deploy_cfg)

    # Load XPolicyLab
    for idx in range(10):
        print(f"\033[94m🚀 Running Episode {idx}\033[0m")
        test_env.reset() # reset model, robot, and environment
        test_env.eval_one_episode()
        test_env.finish_episode()