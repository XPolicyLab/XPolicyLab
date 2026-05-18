import json
import os
import cv2
import numpy as np
import random
import mediapy
from tqdm import tqdm
from VLABench.envs import load_env
from VLABench.utils.utils import euler_to_quaternion,depth_to_rgb

class Evaluator:
    def __init__(self, 
                 tasks,
                 n_episodes,
                 episode_config=None,
                 max_substeps=10,
                 tolerance=1e-3,
                 metrics=["success_rate"],
                 save_dir=None,
                 visulization=False,
                 eval_unseen=False,
                 **kwargs
                 ):
        """
        Basic evaluator of policy
        params:
            tasks: list of task names to evaluate, e.g. ["task1", "task2"]
            n_episodes: number of episodes to evaluate in each task
            episode_config: dict or path of config file for episode generation
            max_substeps: maximum number of substeps for env.step
            metrics: list of metrics to evaluate
            save_dir: directory to save the evaluation results
            visulization: whether to visualize the evaluation progress as videos
            eval_unseen: whether to evaluate the unseen object categories
        """
        if isinstance(episode_config, str):
            with open(episode_config, "r") as f:
                self.episode_config = json.load(f)
        else:self.episode_config = episode_config
        if self.episode_config is None:
            print("Load the task episodes by seeds, instead of episodes")
        else:
            for task in tasks:
                assert len(self.episode_config[task]) >= n_episodes, "The number of episodes should be less than the number of configurations"
        self.eval_tasks = tasks
        self.n_episodes = n_episodes 
        
        self.max_substeps = max_substeps
        self.tolerance = tolerance
        self.target_metrics = metrics
        self.intention_score_threshold = kwargs.get("intention_score_threshold", 0.1)
        self.eval_unseen = eval_unseen
        
        # log, store and visualization
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.visulization = visulization
        
    def evaluate(self, agent):
        """
        Evaluate the agent on all tasks defined in the evaluator.
        """   
        metrics = {}
        for task in self.eval_tasks:
            task_infos = []
            for i in tqdm(range(self.n_episodes), desc=f"Evaluating {task} of {agent.name}"):
                agent.reset()
                kwargs = {
                    "unnorm_key": task,
                    "mp_env": agent.name in ["gea_policy", "pi0_mp", "pi0_mp_depth", "pi0_mp_point", "pi0_mp_point2", "A1_point"]
                }
                
                if self.episode_config is None: 
                    info = self.evaluate_single_episode(agent, task, i, None, seed=42+i, **kwargs)
                else: 
                    info = self.evaluate_single_episode(agent, task, i, self.episode_config[task][i], **kwargs)
                task_infos.append(info)
            
            metric_score = self.compute_metric(task_infos)       
            metrics[task] = metric_score
            
            if self.save_dir is not None:
                os.makedirs(os.path.join(self.save_dir, agent.name), exist_ok=True)
                if os.path.exists(os.path.join(self.save_dir, agent.name, "metrics.json")):
                    with open(os.path.join(self.save_dir, agent.name, "metrics.json"), "r") as f:
                        previous_metrics = json.load(f)
                else:
                    previous_metrics = {}
                previous_metrics.update(metrics)
                with open(os.path.join(self.save_dir, agent.name, "metrics.json"), "w") as f:
                    json.dump(previous_metrics, f, indent=4)
                with open(os.path.join(self.save_dir, agent.name, task, f"detail_info.json"), "w") as f:
                    json.dump(task_infos, f, indent=4)
        return metrics
        
    def evaluate_single_episode(self, agent, task_name, episode_id, episode_config, seed=42, max_episode_length=200, **kwargs):
        """
        If episode_config is given, the task and scene will load deterministically.
        params:
            agent: policy to evaluate
            task_name: name of the task
            episode_id: id of the episode
            episode_config: configuration of the task
            seed: seed for the random number generator, if episode_config is None
            max_episode_length: maximum length of the episode
        """
        task_mapping = {
            'add_condiment': 0,
            'insert_flower': 1,
            'select_book': 2,
            'select_chemistry_tube': 3,
            'select_drink': 4,
            'select_ingredient': 5,
            'select_mahjong': 6,
            'select_poker': 7,
            'select_painting': 8,
            'select_toy': 9,
            'select_fruit': 10
        }
        if episode_config is None: # use random seed to ditermine the task
            np.random.seed(seed)
            random.seed(seed)
        if episode_config is not None:
            env = load_env(task_name, episode_config=episode_config, random_init=False, eval=self.eval_unseen, mp_env=kwargs.get("mp_env", False), mp_eval=True)
        else:
            env = load_env(task_name, random_init=True, eval=self.eval_unseen, mp_env=kwargs.get("mp_env", False), mp_eval=True)
        env.reset()
        success = False
        info = {}
        frames_to_save = []
        masks_to_save = []
        for i in range(max_episode_length):
            observation = env.get_observation()
            observation["instruction"] = env.task.get_instruction()
            observation['robot_position'] = env.robot.robot_config["position"]
            observation['task_id'] = task_mapping[task_name]
            if self.save_dir is not None and self.visulization:
                frames_to_save.append(observation["rgb"])
            if agent.control_mode == "ee":
                pos, euler, gripper_state = agent.predict(observation, **kwargs)
                quat = euler_to_quaternion(*euler)
                action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)[:7]
                action = np.concatenate([action, gripper_state])
            elif agent.control_mode == "joint":
                qpos, gripper_state = agent.predict(observation, **kwargs)
                action = np.concatenate([qpos, gripper_state])
            else:
                raise NotImplementedError(f"Control mode {agent.control_mode} is not implemented")    
            if self.save_dir is not None and self.visulization and kwargs.get("mp_env", False):
                masks_to_save.append(agent.get_frames())

            for _ in range(self.max_substeps):
                timestep = env.step(action)
                if timestep.last():
                    success=True
                    break
                current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                # print('current_qpos',_,current_qpos,np.array(action)[:7])
                if np.max(current_qpos-np.array(action)[:7]) < self.tolerance \
                    and np.min(current_qpos - np.array(action)[:7]) > -self.tolerance:
                    break
            if success:
                break
        info["task"] = task_name
        info["success"] = success
        info["consumed_step"] = i
        info["intention_score"] = env.get_intention_score(threshold=self.intention_score_threshold)
        info["progress_score"] = env.get_task_progress()
        
        env.close()
        if self.save_dir is not None and self.visulization:
            os.makedirs(os.path.join(self.save_dir, agent.name, task_name), exist_ok=True)
            self.save_video(frames_to_save, os.path.join(self.save_dir, agent.name, task_name, f"{episode_id}_{success}.mp4"), add_instruction=env.task.get_instruction())
            if kwargs.get("mp_env", False):
                self.save_video(masks_to_save, os.path.join(self.save_dir, agent.name, task_name, f"{episode_id}_mask_{success}.mp4"))
            
        return info
        
    def compute_metric(self, infos):
        """
        Compute the metric scores for the evaluation
        param:
            infos: list of episode information
        """
        metric = {}
        for key in self.target_metrics:
            if key == "success_rate": # compute the success rate
                success = [info["success"] for info in infos]
                sucess_rate = np.mean(success)
                metric["success_rate"] = sucess_rate
            elif key == "intention_score":
                intention_score = [info["intention_score"] for info in infos]
                avg_intention_score = np.mean(intention_score)
                metric["intention_score"] = avg_intention_score
            elif key == "progress_score":
                progress_score = [info["progress_score"] for info in infos]
                avg_progress_score = np.mean(progress_score)
                metric["progress_score"] = avg_progress_score
            else:
                raise NotImplementedError(f"Metric {key} is not implemented")
        return metric
    
    def save_video(self, frames, save_dir, add_instruction=None):
        frames_to_save = [] 
        for frame in frames:
            frame = np.vstack([np.hstack(frame[:2]), np.hstack(frame[2:4])])
            if add_instruction is not None:
                frame = cv2.putText(frame, add_instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            frames_to_save.append(frame)
        mediapy.write_video(save_dir, 
                            frames_to_save, fps=10) 