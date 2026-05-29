import argparse
import logging
import numpy as np
import requests
import base64
import cv2
from io import BytesIO
from PIL import Image
from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop
from task_config import get_prompt

logging.basicConfig(
    filename='mylogfile.log',  # Log file name
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s %(levelname)s:%(message)s'  # Log format
)


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

    def encode_image_to_base64(self, image_data, idx=0):
        return base64.b64encode(image_data).decode('utf-8')

        # image = cv2.imdecode(
        #     np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        # )
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(f'demo_image_{idx}.png', image)
        # image_data = cv2.imencode('.png', image)[-1].tobytes()
        # return base64.b64encode(image_data).decode('utf-8')

        # image = Image.fromarray(image)
            
        # # Convert to base64
        # buffer = BytesIO()
        # image.save(buffer, format='PNG')
        # image_bytes = buffer.getvalue()
        # return base64.b64encode(image_bytes).decode('utf-8')

    def run_policy(self, input_data):
        """
        Run inference using the policy/model.
        Args:
            input_data: Input data for inference.
        Returns:
            list: Inference results.
        """
        input_data['action'] = np.array(input_data['action'])
        input_data['action'][...,[6,-1]] *= 10
        input_data['action'] = input_data['action'].tolist()
        payload = {
            "instruction": self.prompt,
            "images": [
                self.encode_image_to_base64(input_data['images']['high'],0), 
                self.encode_image_to_base64(input_data['images']['left_hand'],1), 
                self.encode_image_to_base64(input_data['images']['right_hand'],2)
            ],
            "proprio_data": [input_data['action']],
        }
        # # 保存payload
        # import json
        # with open('payload.json', 'w') as f:
        #     json.dump(payload, f)
        # assert False

        response = self.session.post(
            f"{self.base_url}/inference", 
            json=payload, 
            timeout=60  # Longer timeout for inference
        )
        response.raise_for_status()
        result = response.json()
        actions = result['predicted_actions']

        # actions = np.array(actions)
        # actions[actions[:,6]<0.03, 6] = -0.003
        # actions[actions[:,-1]<0.03, -1] = -0.003
        # actions[actions[:,6]>=0.03, 6] = 0.1
        # actions[actions[:,-1]>=0.03, -1] = 0.1
        # actions = actions.tolist()

        actions = np.array(actions)
        actions[:,6] = actions[:,6]/10
        actions[:,-1] = actions[:,-1]/10
        # actions[actions[:,-1]>=0.05, -1] = 0.1
        # actions[actions[:,-1]<=-0.05, -1] = 0
        actions = actions.tolist()
        try:
            gt_actions = input_data['gt_actions']
            mse = np.mean(np.square(np.array(actions) - np.array(gt_actions)))
            l1 = np.mean(np.abs(np.array(actions) - np.array(gt_actions)))
            self.mean_mse += mse
            self.mean_l1 += l1
            self.num_samples += 1
            print(f"mse={mse} l1={l1} mean_mse={self.mean_mse/self.num_samples} mean_l1={self.mean_l1/self.num_samples}")
        except Exception as e:
            pass
        # print('actions',np.array(actions)[:,-1].max(),np.array(actions)[:,-1].min())
        actions = actions[:self.action_nums]
        actions = actions + actions[-1:]*20
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
    parser.add_argument('--task_name', type=str, required=True, help='Task name (e.g. clean_dining_table, stack_bowls). Prompt is resolved from task_config.')
    parser.add_argument('--action_nums', type=int, default=20, help='Number of actions returned per inference step.')
    args = parser.parse_args()

    prompt = get_prompt(args.task_name)
    if not prompt:
        raise ValueError(f"Unknown task_name: {args.task_name}. Check task_config.ROBO_CHALLENGE_TASKS for valid names.")

    # these args are generally not changed during evaluation, so we put them here.
    image_size = [640, 480] # this refers to README.md#get-state request parameter `width` and `height`
    image_type = ["high", "left_hand", "right_hand"] # this refers to README.md#get-state request parameter `image_type`
    action_type = "joint" # this refers to both README.md#get-state and README.md#post-action parameters `action_type`
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
