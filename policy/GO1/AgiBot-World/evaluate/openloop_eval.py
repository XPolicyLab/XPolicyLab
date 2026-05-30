import importlib
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from evaluate.deploy import GO1Infer
from go1.lerobot.dataset_lerobot import WrappedLeRobotDataset


def plot_line(result_list, model_path, save_path):
    PRED = []
    GT = []

    for d in result_list:
        pred, gt = d["pred"], d["gt_action"]
        PRED.append(pred)
        GT.append(gt)

    PRED = np.concatenate(PRED, axis=0)
    GT = np.concatenate(GT, axis=0)
    print(PRED.shape)
    print(GT.shape)

    nrows = 4
    ncols = math.ceil(GT.shape[-1] / nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(25, 15))
    fig.suptitle(f"pred vs gt\n{model_path}")
    axs_flat = axs.flatten()

    for i in range(GT.shape[-1]):
        ax = axs_flat[i]
        ax.set_title(f"Index: {i}")
        ax.plot(PRED[:, i], color="blue", label="Pred")
        ax.plot(GT[:, i], color="green", linestyle="dashed", label="GT")
        ax.legend()

    file_name = os.path.join(save_path, "_".join(model_path.split("/")[-2:]) + ".jpg")
    fig.savefig(file_name, bbox_inches="tight")
    print(f"save to {file_name}")


def main():
    model_path = "/path/to/your/checkpoint"  # Update this to your model path
    exp_path = model_path.rsplit("/", 1)[0]

    if exp_path not in sys.path:
        sys.path.append(exp_path)

    module_name = [name for name in os.listdir(exp_path) if name[-3:] == ".py"][0][:-3]
    cfg = importlib.import_module(module_name)
    dataset_args = cfg.DatasetArguments()
    space_args = cfg.SpaceArguments()

    model = GO1Infer(
        model_path=model_path,
        data_stats_path=os.path.join(exp_path, "dataset_stats.json"),
    )

    ds = WrappedLeRobotDataset(
        root=dataset_args.data_root_dir,
        action_chunk_size=model.config.action_chunk_size,
        transforms=None,
        text_tokenizer=model.text_tokenizer,
        num_image_token=model.num_image_token,
        image_size=model.image_size,
        pad2square=model.config.pad2square,
        dynamic_image_size=model.dynamic_image_size,
        use_thumbnail=model.config.use_thumbnail,
        min_dynamic_patch=model.config.min_dynamic_patch,
        max_dynamic_patch=model.config.max_dynamic_patch,
        space_args=space_args,
        debug=True,
    )
    infer_interval = model.config.action_chunk_size

    data_length = len(ds)
    print(f"data_length: {data_length}")
    result_list = []

    for i in tqdm.tqdm(range(0, data_length, infer_interval)):
        raw_target = ds[i]
        gt_action = raw_target["action_gts"][:infer_interval].numpy()

        pred = model.predict_action(raw_target)[:infer_interval]

        out = {
            "pred": pred,
            "gt_action": gt_action,
        }
        result_list.append(out)

    save_path = "tmp_image"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plot_line(result_list, model_path, save_path)


if __name__ == "__main__":
    main()
