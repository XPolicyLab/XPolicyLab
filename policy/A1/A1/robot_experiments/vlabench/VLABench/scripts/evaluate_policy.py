import os
import argparse
from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.gea_policy import gea_policy
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.tasks import *
from VLABench.robots import *

os.environ["MUJOCO_GL"]= "egl"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=None, help="Specific tasks to run, work when eval-track is None")
    parser.add_argument('--eval-track', default=None, type=str, choices=["track_1_in_distribution", "track_2_cross_category", "track_3_common_sense", "track_4_semantic_instruction"], help="The evaluation track to run")
    parser.add_argument('--n-episode', default=1, type=int, help="The number of episodes to evaluate for a task")
    parser.add_argument('--policy', default="openvla", help="The policy to evaluate")
    parser.add_argument('--model_ckpt', default="/remote-home1/sdzhang/huggingface/openvla-7b", help="The base model checkpoint path")
    parser.add_argument('--lora_ckpt', default="/remote-home1/pjliu/openvla/weights/vlabench/select_fruit+CSv1+lora/", help="The lora checkpoint path")
    parser.add_argument('--save-dir', default="logs", help="The directory to save the evaluation results")
    parser.add_argument('--visulization', action="store_true", default=False, help="Whether to visualize the episodes")
    parser.add_argument('--metrics', nargs='+', default=["success_rate"], choices=["success_rate", "intention_score", "progress_score"], help="The metrics to evaluate")
    args = parser.parse_args()
    return args

def evaluate(args):
    episode_config = None
    if args.eval_track is not None:
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/evaluation/tracks", f"{args.eval_track}.json"), "r") as f:
            episode_config = json.load(f)
            tasks = list(episode_config.keys())
    if args.tasks is not None:
        tasks = args.tasks
    assert isinstance(tasks, list)

    evaluator = Evaluator(
        tasks=tasks,
        n_episodes=args.n_episode,
        episode_config=episode_config,
        max_substeps=10,   
        save_dir=args.save_dir,
        visulization=args.visulization,
        metrics=args.metrics
    )
    if args.policy.lower() == "openvla":
        policy = OpenVLA(
            model_ckpt=args.model_ckpt,
            lora_ckpt=args.lora_ckpt,
            norm_config_file=os.path.join(os.getenv("VLABENCH_ROOT"), "configs/model/openvla_config.json") # TODO: re-compuate the norm state by your own dataset
        )
    elif args.policy.lower() == "gea_policy":
        policy = gea_policy(args.model_ckpt)
    result = evaluator.evaluate(policy)
    with open(os.path.join(args.save_dir, args.policy, "evaluation_result.json"), "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    args = get_args()
    evaluate(args)