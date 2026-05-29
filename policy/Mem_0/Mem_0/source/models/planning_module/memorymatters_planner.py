import os
import sys
import time
import base64
import io
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from termcolor import cprint

from openai import OpenAI

class MemoryMattersPlanner(nn.Module):
    def __init__(self, 
        config: Optional[dict] = None, 
        device: Optional[torch.device] = None, 
        vllm_url: str = "http://localhost:8000",
        global_task: str = None,
        **kwargs):
        """
        Initialize the Qwen3-VL wrapper.
        """
        super().__init__()
        # get qwen_vl config
        qwenvl_config = config.planning_module.get("qwen_vl", {})
        self.model_path = qwenvl_config.get("model_path", "checkpoints/Qwen3-VL-8B-Thinking")
        
        # set qwen vl model
        # self.qwen_model = Qwen3VL_Encapsulation(model_path, device=device)
        self.vllm_client = OpenAI(api_key="EMPTY", base_url=vllm_url, timeout=3600)
        self.system_prompt = qwenvl_config.get("system_prompt", None)    # system prompt for Qwen-VL.
        self.global_task = global_task    # global task instruction
        
        # organize memory information bank
        self.key_information = []    # key information for in-episode memory
        self.finished_subtasks = []    # subtask language instructions for in-episode memory
        self.initial_observation = None    # initial observation for in-episode memory
        
    
    def _image_to_data_url(self, image: Union[Image.Image, str]) -> str:
        """
        Convert PIL Image or image path to base64 data URL for vLLM API.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            base64 encoded image string in format: "data:image/png;base64,..."
        """
        # Load image if path string
        if isinstance(image, str):
            # Check if it's already a URL (http/https)
            if image.startswith(('http://', 'https://', 'data:', 'file://')):
                return image
            # Otherwise, treat as file path
            img = Image.open(image)
        else:
            img = image
        
        # Convert to RGB if necessary (vLLM expects RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_base64}"
    
    def _video_to_data_url(self, video: str) -> str:
        """
        Convert local video file path to base64 data URL for vLLM API.
        
        Args:
            video: Path to video file (e.g., .mp4, .avi, etc.)
            
        Returns:
            base64 encoded video string in format: "data:video/mp4;base64,..."
        """
        # Check if it's already a URL (http/https/data URL)
        if isinstance(video, str) and video.startswith(('http://', 'https://', 'data:')):
            return video
        
        # Read video file and convert to base64
        if not os.path.exists(video):
            raise FileNotFoundError(f"Video file not found: {video}")
        
        # Determine MIME type based on file extension
        video_ext = os.path.splitext(video)[1].lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
        }
        mime_type = mime_types.get(video_ext, 'video/mp4')
        
        # Read video file as binary
        with open(video, 'rb') as video_file:
            video_data = video_file.read()
        
        # Encode to base64
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        
        return f"data:{mime_type};base64,{video_base64}"
    
    def update_initial_observation(self, observation: Union[Image.Image, str]):
        """
        Update the initial observation of the high-level policy.
        Accepts either a PIL.Image.Image object or a path string to the observation file.
        The observation is an image.
        """
        self.initial_observation = observation
        
    def update_reference_task_demo(self, reference_task_demo: str, subtasks_sequence: str):
        """
        Update the reference task demo of the high-level policy.
        the string in reference_task_demo is the path to the reference task demo video file.
        """
        self.reference_task_demo = reference_task_demo
        self.subtasks_sequence = subtasks_sequence
        
    def update_image_or_video_input(self, image_or_video_inputs: List[Union[Image.Image, str]], subtasks: Optional[List[str]] = None):
        """
        Update the image or video input of the high-level policy.
        For images: accepts PIL.Image.Image objects or path strings.
        For videos: accepts path strings (video file paths).
        the string in subtasks is the subtask language instruction.
        """
        if subtasks is not None:
            for image_or_video_input, subtask in zip(image_or_video_inputs, subtasks):
                self.key_information.append(image_or_video_input)
                self.finished_subtasks.append(subtask)
                
    def prepare_qwen_input(self):
        """
        Prepare the input for Qwen-VL.
        """
        system_prompt_msg = [{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}]
        
        # Build user content list
        user_content = []
        
        # Add global task text
        glocal_task_msg = {"type": "text", "text": "<global_task>: "+self.global_task+"\n"}
        user_content.append(glocal_task_msg)
        
        # Add initial observation image
        assert self.initial_observation is not None, "Initial observation must be provided"
        initial_obs_text_msg = {"type": "text", "text": "<initial_observation>: "}
        user_content.append(initial_obs_text_msg)
        # Convert image to data URL for vLLM API
        image_url = self._image_to_data_url(self.initial_observation)
        initial_observation_msg = {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }
        user_content.append(initial_observation_msg)
        initial_obs_text_msg = {"type": "text", "text": ".\n"}
        user_content.append(initial_obs_text_msg)

        
        # Add finished subtasks text
        if len(self.finished_subtasks) > 0:
            finished_subtasks_text_begin_msg = {"type": "text", "text": "<finished_subtasks>: "}
            user_content.append(finished_subtasks_text_begin_msg)
            for i in range(len(self.finished_subtasks)):
                finished_subtask_text_msg = {"type": "text", "text": str(i)+": "+self.finished_subtasks[i]+"The corresponding image is: "}
                user_content.append(finished_subtask_text_msg)
                # Convert image to data URL for vLLM API
                image_url = self._image_to_data_url(self.key_information[i])
                finished_subtask_image_msg = {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
                user_content.append(finished_subtask_image_msg)
                if i < len(self.finished_subtasks) - 1:
                    finished_subtask_text_msg = {"type": "text", "text": ". "}
                    user_content.append(finished_subtask_text_msg)
            finished_subtasks_text_end_msg = {"type": "text", "text": ".\n"}
            user_content.append(finished_subtasks_text_end_msg)
        else:
            finished_subtasks_text_msg = {"type": "text", "text": "finished_subtasks: null.\n"}
            user_content.append(finished_subtasks_text_msg)    
        
        user_prompt_msg = [{"role": "user", "content": user_content}]
        
        full_prompt_msg = system_prompt_msg + user_prompt_msg
        
        return full_prompt_msg
    
    def generate_anwser(self, inputs):
        """
        Generate the answer from the Qwen-VL model.
        """        
        start = time.time()
        response = self.vllm_client.chat.completions.create(
            model=self.model_path,
            messages=inputs,
            max_tokens=2048,
            # extra_body={"mm_processor_kwargs": {"fps": 1, "do_sample_frames": True}},
            # temperature=0.6,
            # top_p=0.95,
        )
        print(f"Response costs: {time.time() - start:.2f}s")
        return response.choices[0].message.content

        

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./source/config/planning_module_inference.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # -------------- #
    # Image with Reference
    # -------------- #

    cfg = OmegaConf.load(args.config_yaml)
    # global_task = "There are two batteries and a battery slot on the table. Combining the two batteries in different orientations causes the dashboard needle to rotate."
    global_task = "On the table, red, green, and blue blocks are arranged randomly along with three lids. From the current viewpoint, cover the blocks from right to left using the lids, and then uncover them again in the sequence red, green, and blue."
    high_level_policy = MemoryMattersPlanner(cfg, global_task=global_task)
    
    # initialize the initial observation (must)
    high_level_policy.update_initial_observation(os.getcwd() + "/llamafactory_data/rmbench_data_Mn_lerobot/rmbench_data_Mn_lerobot_images/episode_148/initial_observation.png")
    
    qwen_inputs = high_level_policy.prepare_qwen_input()
    # print(qwen_inputs)
    answer = high_level_policy.generate_anwser(qwen_inputs)
    cprint(f"Answer: {answer}", "cyan")
    subtask = answer.split("next_subtask: ")[-1].split(".")[0]+"."
    cprint(f"Subtask: {subtask}", "green")

    
    high_level_policy.update_image_or_video_input([os.getcwd() + "/llamafactory_data/rmbench_data_Mn_lerobot/rmbench_data_Mn_lerobot_images/episode_148/frame_1.png"], [subtask])
    qwen_inputs = high_level_policy.prepare_qwen_input()    
    answer = high_level_policy.generate_anwser(qwen_inputs)
    cprint(f"Answer: {answer}", "cyan")
    subtask = answer.split("next_subtask: ")[-1].split(".")[0]+"."
    cprint(f"Subtask: {subtask}", "green")

    
    high_level_policy.update_image_or_video_input([os.getcwd() + "/llamafactory_data/rmbench_data_Mn_lerobot/rmbench_data_Mn_lerobot_images/episode_148/frame_2.png"], [subtask])
    qwen_inputs = high_level_policy.prepare_qwen_input()
    answer = high_level_policy.generate_anwser(qwen_inputs)
    cprint(f"Answer: {answer}", "cyan")
    subtask = answer.split("next_subtask: ")[-1].split(".")[0]+"."
    cprint(f"Subtask: {subtask}", "green")

    
    high_level_policy.update_image_or_video_input([os.getcwd() + "/llamafactory_data/rmbench_data_Mn_lerobot/rmbench_data_Mn_lerobot_images/episode_148/frame_3.png"], [subtask])
    qwen_inputs = high_level_policy.prepare_qwen_input()
    answer = high_level_policy.generate_anwser(qwen_inputs)
    cprint(f"Answer: {answer}", "cyan")
    subtask = answer.split("next_subtask: ")[-1].split(".")[0]+"."
    cprint(f"Subtask: {subtask}", "green")
    
    high_level_policy.update_image_or_video_input([os.getcwd() + "/llamafactory_data/rmbench_data_Mn_lerobot/rmbench_data_Mn_lerobot_images/episode_148/frame_4.png"], [subtask])
    qwen_inputs = high_level_policy.prepare_qwen_input()
    answer = high_level_policy.generate_anwser(qwen_inputs)
    cprint(f"Answer: {answer}", "cyan")
    subtask = answer.split("next_subtask: ")[-1].split(".")[0]+"."
    cprint(f"Subtask: {subtask}", "green")
    
    high_level_policy.update_image_or_video_input([os.getcwd() + "/llamafactory_data/rmbench_data_Mn_lerobot/rmbench_data_Mn_lerobot_images/episode_148/frame_5.png"], [subtask])
    qwen_inputs = high_level_policy.prepare_qwen_input()
    answer = high_level_policy.generate_anwser(qwen_inputs)
    cprint(f"Answer: {answer}", "cyan")
    subtask = answer.split("next_subtask: ")[-1].split(".")[0]+"."
    cprint(f"Subtask: {subtask}", "green")