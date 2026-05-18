import argparse
import logging
import numpy as np
import requests
import base64
from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop
from task_config import get_prompt

logging.basicConfig(
    filename='ur5_demo_joint.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)
class DummyPolicy:
    def __init__(self, base_url: str = "http://localhost:5050", prompt: str = None, action_nums: int = 20):
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
        proprio_data = input_data['action']
        proprio_data[6] /= 255.0
        payload = {
            "instruction": self.prompt,
            "images": [
                base64.b64encode(input_data['images']['right_hand']).decode('utf-8'),
                base64.b64encode(input_data['images']['left_hand']).decode('utf-8'),
            ],
            "proprio_data": [proprio_data],
        }
        response = self.session.post(
            f"{self.base_url}/inference",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        actions = np.array(result['predicted_actions'])
        try:
            gt_actions = np.array(input_data['gt_actions'])
            gt_gripper = gt_actions[:, 6] / 255.0
            
            gt_actions[:, 6] = gt_gripper            

            mse = np.mean(np.square(actions - gt_actions))
            l1 = np.mean(np.abs(actions - gt_actions))
            self.mean_mse += mse
            self.mean_l1 += l1
            self.num_samples += 1
            print(f"gt_actions: {gt_actions[0]}")
            print(f"test_actions: {actions[0]}")
            print(f"mse={mse} l1={l1} mean_mse={self.mean_mse/self.num_samples} mean_l1={self.mean_l1/self.num_samples}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            pass
        actions[:, 6] *= 255.0
        actions = actions[:self.action_nums].tolist()
        actions = actions + actions[-1:] *20
        return actions

class GPUClient:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, state):
        result = self.policy.run_policy(state)
        return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--user_token', type=str, required=True, help='User token')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID')
    parser.add_argument('--url', type=str, required=True, help='URL')
    parser.add_argument('--action_nums', type=int, default=20, help='Number of actions returned per inference step.')
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        help='Task name (e.g. shred_scrap_paper, sort_books). Prompt is resolved from task_config.',
    )

    args = parser.parse_args()
    prompt = get_prompt(args.task_name)
    if not prompt:
        raise ValueError(
            f"Unknown task_name: {args.task_name}. Check task_config.ROBO_CHALLENGE_TASKS for valid names."
        )

    image_size = [1280, 720]
    image_type = ["left_hand", "right_hand"]
    action_type = "leftjoint"
    duration = 0.05
    client = InterfaceClient(args.user_token)
    if args.action_nums <= 0:
        raise ValueError("--action_nums must be a positive integer.")
    policy = DummyPolicy(args.url, prompt, action_nums=args.action_nums)
    gpu_client = GPUClient(policy)

    job_loop(client, gpu_client, args.run_id, image_size, image_type, action_type, duration)

if __name__ == '__main__':
    main()