import os
import sys
import argparse
from utils import import_custom_class
import sys
ROOT = os.path.dirname(os.path.abspath(__file__)) 

sys.path.append(os.path.join(ROOT, "runner"))
sys.path.append(os.path.join(ROOT, "data"))
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--runner_class_path', type=str, default="runner/finetune_trainer.py")
    parser.add_argument('--runner_class', type=str, default="Trainer")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to trained checkpoint, used in inference stage only')
    parser.add_argument('--n_validation', type=int, default=1, help='num of samples to predict, used in inference stage only')
    parser.add_argument('--n_chunk_action', type=int, default=1, help='num of action chunks to predict, used in action inference stage only')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save outputs, used in inference stage only')
    args = parser.parse_args()
    Runner = import_custom_class(
        args.runner_class, args.runner_class_path, 
    )
    
    runner = Runner(args.config_file)
    runner.prepare_dataset()
    runner.prepare_models()
    runner.prepare_trainable_parameters()
    runner.prepare_optimizer()
    runner.prepare_for_training()
    runner.prepare_trackers()
    runner.train()



if __name__ == "__main__":
  
    main()
    