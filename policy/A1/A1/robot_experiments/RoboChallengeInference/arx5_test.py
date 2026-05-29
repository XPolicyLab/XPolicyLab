import argparse
import logging
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from arx5_demo import GPUClient, DummyPolicy
from robot.interface_client import InterfaceClient
from task_config import get_prompt

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

DEFAULT_USER_ID = "test_user"
DEFAULT_JOBS = ["test_job"]
DEFAULT_ROBOT_ID = "test_robot"


class ActionCollector:
    """收集预测动作和真实动作用于开环图可视化"""

    def __init__(self):
        self.gt_actions = []  # 真实动作
        self.pred_actions = []  # 预测动作
        self.timestamps = []
        self.segment_starts = []

    def add(self, gt_action, pred_action, timestamp=None, is_segment_start: bool = False):
        self.gt_actions.append(gt_action)
        self.pred_actions.append(pred_action)
        self.timestamps.append(timestamp if timestamp is not None else len(self.timestamps))
        self.segment_starts.append(bool(is_segment_start))

    def save_open_loop_plot(self, save_path=None, title="Action Comparison"):
        if len(self.gt_actions) == 0:
            logging.warning("No data to plot")
            return

        gt = np.array(self.gt_actions)
        pred = np.array(self.pred_actions)
        timesteps = np.arange(len(gt))

        segment_starts = np.array(self.segment_starts, dtype=bool)
        if segment_starts.shape[0] != gt.shape[0]:
            logging.warning(
                "segment_starts length (%d) != number of steps (%d)",
                segment_starts.shape[0],
                gt.shape[0],
            )
            min_len = min(segment_starts.shape[0], gt.shape[0])
            segment_starts = segment_starts[:min_len]
            timesteps = timesteps[:min_len]
            gt = gt[:min_len]
            pred = pred[:min_len]

        plot_timesteps = timesteps.astype(float)
        plot_gt = gt.astype(float, copy=True)
        plot_pred = pred.astype(float, copy=True)
        plot_segment_starts = segment_starts.copy()

        start_positions = np.where(segment_starts)[0]
        if start_positions.size > 1:
            insert_positions = start_positions[1:]
            offset = 0
            nan_row = np.full((gt.shape[1],), np.nan, dtype=float)
            for s in insert_positions:
                ins = int(s + offset)
                plot_timesteps = np.insert(plot_timesteps, ins, float(s) - 0.5)
                plot_gt = np.insert(plot_gt, ins, nan_row, axis=0)
                plot_pred = np.insert(plot_pred, ins, nan_row, axis=0)
                plot_segment_starts = np.insert(plot_segment_starts, ins, False)
                offset += 1

        n_dims = gt.shape[1]
        fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
        if n_dims == 1:
            axes = [axes]

        for i in range(n_dims):
            ax = axes[i]
            ax.plot(
                plot_timesteps,
                plot_gt[:, i],
                label="Ground Truth",
                color="#1f77b4",
                linewidth=1.5,
                alpha=0.8,
            )
            ax.plot(
                plot_timesteps,
                plot_pred[:, i],
                label="Prediction",
                color="#ff7f0e",
                linewidth=1.5,
                alpha=0.8,
            )

            start_idx = np.where(segment_starts)[0]
            if start_idx.size > 0:
                gt_start_label = "GT Segment Start" if i == 0 else "_nolegend_"
                pred_start_label = "Pred Segment Start" if i == 0 else "_nolegend_"
                ax.scatter(
                    timesteps[start_idx],
                    gt[start_idx, i],
                    s=25,
                    c="#2ca02c",
                    marker="o",
                    alpha=0.9,
                    label=gt_start_label,
                )
                ax.scatter(
                    timesteps[start_idx],
                    pred[start_idx, i],
                    s=35,
                    c="#9467bd",
                    marker="x",
                    alpha=0.9,
                    label=pred_start_label,
                )

            ax.set_ylabel(f"Action Dim {i}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Timestep")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"open_loop_comparison_{timestamp_str}.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logging.info(f"Open loop plot saved to {save_path}")
        plt.close()


def process_job(client, gpu_client, job_id, robot_id, image_size, image_type, action_type, duration, max_wait=600,
                action_collector: ActionCollector | None = None):
    try:
        start_time = time.time()
        idx = 0
        while True:
            client.start_motion()
            logging.info("Started robot")
            state = client.get_state(image_size, image_type, action_type)
            if not state:
                time.sleep(0.5)
                continue
            if state['state'] == "size_none":
                client.post_size()
                time.sleep(0.5)
                continue
            if state['state'] != "normal" or state['pending_actions'] != 0:
                time.sleep(0.5)
                continue
            logging.info("get_robot_state time: %.2f", time.time() - state['timestamp'])
            result = gpu_client.infer(state)

            if action_collector is not None and "gt_actions" in state:
                gt_actions = np.array(state["gt_actions"])
                pred_actions = np.array(result)
                for idx_step, (gt_action, pred_action) in enumerate(
                    zip(gt_actions, pred_actions)
                ):
                    action_collector.add(
                        gt_action,
                        pred_action,
                        is_segment_start=(idx_step == 0),
                    )

            client.post_actions(result, duration, action_type)
            idx += 1
            if idx>50:
                break
            if time.time() - start_time > max_wait:
                logging.warning(f"Job {job_id} exceeded max wait time.")
                break
        client.end_motion()
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        client.end_motion()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, required=True, help='URL')
    parser.add_argument(
        '--robot-url',
        type=str,
        default=None,
        help='Robot state server URL. If not provided, uses robot/interface_client.py default mock_url.',
    )
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        help='Task name (e.g. arrange_flowers, open_the_drawer). Prompt is resolved from task_config.',
    )
    parser.add_argument(
        '--save-plot',
        type=str,
        default='arx5_open_loop.png',
        help='Path to save open loop comparison plot.',
    )
    args = parser.parse_args()
    prompt = get_prompt(args.task_name)
    if not prompt:
        raise ValueError(f"Unknown task_name: {args.task_name}. Check task_config.ROBO_CHALLENGE_TASKS for valid names.")

    image_size = [640, 480]
    image_type = ["high", "left_hand", "right_hand"]
    action_type = "leftjoint"
    duration = 0.05

    client = InterfaceClient(DEFAULT_USER_ID, mock=True, mock_url_override=args.robot_url)
    client.update_job_info(DEFAULT_JOBS[0], DEFAULT_ROBOT_ID)

    policy = DummyPolicy(args.url, prompt)
    gpu_client = GPUClient(policy)

    action_collector = ActionCollector()

    jobs = DEFAULT_JOBS

    while jobs:
        for job_id in jobs[:]:
            try:
                process_job(
                    client,
                    gpu_client,
                    job_id,
                    DEFAULT_ROBOT_ID,
                    image_size,
                    image_type,
                    action_type,
                    duration,
                    action_collector=action_collector,
                )
                jobs.remove(job_id)
            except Exception as e:
                logging.error(f"Error processing job {job_id}: {e}")
                import traceback
                logging.error(traceback.format_exc())

    if len(action_collector.gt_actions) > 0:
        action_collector.save_open_loop_plot(
            save_path=args.save_plot,
            title=f"Action Comparison for {args.task_name}",
        )

    logging.info("All jobs processed.")
    return True

if __name__ == "__main__":
    main()
