import torch
from torch.utils.tensorboard import SummaryWriter
import os
import json

def main():
    # create SummaryWriter 
    pythia = "410M"
    log_p = f'/data/private/wenjj/llava-pythia/checkpoint_all/pythia_{pythia}/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-w_state_huber/log'
    
    trainint_state_p = f"/data/private/wenjj/llava-pythia/checkpoint_all/pythia_{pythia}/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-w_state_huber/trainer_state.json"
    
    os.makedirs(log_p, exist_ok=True)

    writer = SummaryWriter(log_dir=log_p)

    with open(trainint_state_p, "r") as f:
        data = json.load(f)

    # save loss in SummaryWriter
    for each in data['log_history']:
        if not 'loss' in each.keys():
            continue
        step, loss = each['step'], each['loss']
        writer.add_scalar('train/loss', loss, step)

    writer.close()

if __name__ == "__main__":
    main()
