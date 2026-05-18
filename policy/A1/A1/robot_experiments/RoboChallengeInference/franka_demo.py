import argparse
import logging
import numpy as np
import requests
import base64
from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop
from task_config import get_prompt

logging.basicConfig(
    filename='mylogfile.log',  # Log file name
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s %(levelname)s:%(message)s'  # Log format
)

def rpy_to_quaternion(roll, pitch, yaw):
    """
    RPY(roll, pitch, yaw) -> 四元数 (x, y, z, w)
    输入单位：弧度
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    # ZYX (yaw-pitch-roll) 组合
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr

    return [x, y, z, w]

def quaternion_to_rpy(quaternion) -> np.ndarray:
    """
    将四元数转换为RPY（Roll-Pitch-Yaw）欧拉角。
    
    参数:
        quaternion: 四元数
    
    返回:
        rpy: [roll, pitch, yaw] 欧拉角（弧度）
    """
    q = np.asarray(quaternion)
    
    w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
    
    # 归一化四元数
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    
    # 计算RPY角（ZYX顺序，即先绕Z轴旋转yaw，再绕Y轴旋转pitch，最后绕X轴旋转roll）
    # Roll (绕X轴)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (绕Y轴)
    sinp = 2 * (w * y - z * x)
    # 处理边界情况，避免数值误差
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # 使用90度
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (绕Z轴)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

class DummyPolicy:
    """
    Example policy class.
    Users should implement the __init__ and run_policy methods according to their own logic.
    """

    def __init__(self, base_url: str = "http://localhost:6789", prompt: str = None, action_nums: int = 20):
        """
        Initialize the policy.
        Args:
            port (str): Port to the model checkpoint file.
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.prompt = prompt
        self.action_nums = action_nums
        self.mean_mse = 0
        self.mean_l1 = 0
        self.num_samples = 0

    def run_policy(self, input_data):
        """
        Run inference using the policy/model.
        Args:
            input_data: Input data for inference.
        Returns:
            list: Inference results.
        """
        proprio_data = input_data['action']
        proprio_data[-1] *= 10
        proprio_data = proprio_data[:3]+list(quaternion_to_rpy(proprio_data[3:7]))+[proprio_data[7]]
        # print('proprio_data',proprio_data)
        payload = {
            "instruction": self.prompt,
            "images": [
                base64.b64encode(input_data['images']['right_hand']).decode('utf-8'), 
                base64.b64encode(input_data['images']['left_hand']).decode('utf-8'), 
                base64.b64encode(input_data['images']['high']).decode('utf-8')
            ],
            "proprio_data": [proprio_data],
        }
        response = self.session.post(
            f"{self.base_url}/inference", 
            json=payload, 
            timeout=60  # Longer timeout for inference
        )
        response.raise_for_status()
        result = response.json()
        actions = result['predicted_actions']
        actions = np.array(actions)
        actions[:,-1] = actions[:,-1]/10
        actions = actions.tolist()
        try:
            gt_actions = input_data['gt_actions']
            test_actions = np.array(actions)
            test_gt_actions = np.array(gt_actions)
            tmp_actions = []
            for action in test_gt_actions:
                tmp_actions.append(list(action[:3])+list(quaternion_to_rpy(action[3:7]))+[action[7]])
            test_gt_actions = np.array(tmp_actions)
            test_actions[:,3] = 0
            test_gt_actions[:,3] = 0

            mse = np.mean(np.square(test_actions - test_gt_actions))  
            l1 = np.mean(np.abs(test_actions - test_gt_actions))
            self.mean_mse += mse
            self.mean_l1 += l1
            self.num_samples += 1
            print(f"mse={mse} l1={l1} mean_mse={self.mean_mse/self.num_samples} mean_l1={self.mean_l1/self.num_samples}")
        except Exception as e:
            pass
        
        actions = np.array(actions)
        actions[actions[:, -1] < 0.02, -1] = 0
        actions = actions.tolist()
        post_actions = []
        for action in actions:
            post_actions.append(action[:3]+rpy_to_quaternion(action[3], action[4], action[5])+[action[6]])
        actions = post_actions[:self.action_nums]
        actions = actions + actions[-1:] * 20
        return actions

class GPUClient:
    """
    Inference client class.
    """

    def __init__(self, policy):
        """
        Initialize the inference client with a policy.
        Args:
            policy (DummyPolicy): An instance of the policy class.
        """
        self.policy = policy

    def infer(self, state):
        """
        Main entry point for inference.
        Args:
            state: Input state for the policy. Refer to README.md#get-state response example for details. It's unpickled and passed as a dict here.
        Returns:
            list: Inference results from the policy. Refer to README.md#post-action request parameters for details. This will be the `actions` field in the request.
        """
        result = self.policy.run_policy(state)
        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--user_token', type=str, required=True, help='User token')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID. Get it from the detail page of your submission')
    parser.add_argument('--url', type=str, required=True, help='URL')
    parser.add_argument('--task_name', type=str, required=True, help='Task name (e.g. move_objects_into_box, press_three_buttons). Prompt is resolved from task_config.')
    parser.add_argument('--action_nums', type=int, default=20, help='Number of actions returned per inference step.')
    args = parser.parse_args()

    prompt = get_prompt(args.task_name)
    if not prompt:
        raise ValueError(f"Unknown task_name: {args.task_name}. Check task_config.ROBO_CHALLENGE_TASKS for valid names.")

    # these args are generally not changed during evaluation, so we put them here.
    image_size = [640, 480] # this refers to README.md#get-state request parameter `width` and `height`
    image_type = ["high", "left_hand", "right_hand"] # this refers to README.md#get-state request parameter `image_type`
    action_type = "leftpos" # this refers to both README.md#get-state and README.md#post-action parameters `action_type`
    duration = 0.05 # this refers to README.md#post-action request parameter `duration`

    client = InterfaceClient(args.user_token)
    if args.action_nums <= 0:
        raise ValueError("--action_nums must be a positive integer.")
    policy = DummyPolicy(args.url, prompt, action_nums=args.action_nums)
    gpu_client = GPUClient(policy)  # add your own parameters

    # main job loop. This function monitors when jobs are ready to eval and do the evaluation
    job_loop(client, gpu_client, args.run_id, image_size, image_type, action_type, duration)


if __name__ == '__main__':
    main()
