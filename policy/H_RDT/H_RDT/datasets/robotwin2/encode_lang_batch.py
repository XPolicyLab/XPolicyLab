import os
import pandas as pd
import torch
import yaml
import sys
from pathlib import Path

# Add the project root to sys.path for importing models
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.encoder.t5_encoder import T5Embedder


def main():
    """
    Batch encode task instructions from CSV file and save embeddings to lang_embeddings folder.
    This script reads task_instructions.csv and generates T5 embeddings for each task.
    """
    
    # Configuration
    GPU = 0
    MODEL_PATH = os.environ.get('T5_MODEL_PATH', './t5-v1_1-xxl')
    CONFIG_PATH = os.environ.get('HRDT_CONFIG_PATH', '../../configs/hrdt_finetune.yaml')
    
    # Get current script directory
    current_dir = Path(__file__).parent
    
    # Input and output paths (relative to current script)
    CSV_PATH = current_dir / 'task_instructions.csv'
    OUTPUT_DIR = current_dir / 'lang_embeddings'
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"Reading task instructions from: {CSV_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"T5 model path: {MODEL_PATH}")
    print(f"Config path: {CONFIG_PATH}")
    
    # Load configuration
    try:
        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)
    except FileNotFoundError:
        print(f"Error: Config file not found at {CONFIG_PATH}")
        print("Please ensure the config file exists or set HRDT_CONFIG_PATH environment variable")
        return
    
    # Initialize T5 embedder
    device = torch.device(f"cuda:{GPU}")
    print(f"Using device: {device}")
    
    try:
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH, 
            model_max_length=config["dataset"]["tokenizer_max_length"], 
            device=device,
            use_offload_folder=None  # Set to None for now, can be configured if needed
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        print("T5 embedder initialized successfully")
    except Exception as e:
        print(f"Error initializing T5 embedder: {e}")
        print("Please check T5_MODEL_PATH environment variable and model availability")
        return
    
    # Read task instructions from CSV
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} tasks from CSV")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {CSV_PATH}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Process each task
    success_count = 0
    total_tasks = len(df)
    
    for idx, row in df.iterrows():
        task_name = row['task_name']
        instruction = row['instruction']
        
        print(f"\nProcessing [{idx+1}/{total_tasks}]: {task_name}")
        print(f"Instruction: {instruction}")
        
        try:
            # Tokenize instruction
            tokens = tokenizer(
                instruction, 
                return_tensors="pt",
                padding="longest",
                truncation=True
            )["input_ids"].to(device)
            
            tokens = tokens.view(1, -1)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = text_encoder(tokens).last_hidden_state.detach().cpu()
            
            # Save embeddings
            save_path = OUTPUT_DIR / f"{task_name}.pt"
            embedding_data = {
                "name": task_name,
                "instruction": instruction,
                "embeddings": embeddings
            }
            
            torch.save(embedding_data, save_path)
            
            print(f"✓ Saved embedding with shape {embeddings.shape} to {save_path}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ Error processing {task_name}: {e}")
            continue
    
    print(f"\n" + "="*50)
    print(f"Batch encoding completed!")
    print(f"Successfully processed: {success_count}/{total_tasks} tasks")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if success_count < total_tasks:
        print(f"Warning: {total_tasks - success_count} tasks failed to process")


if __name__ == "__main__":
    main()