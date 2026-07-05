import os
import json
import multiprocessing as mp
from multiprocessing import Process, Queue
import time

import torch
import yaml
import h5py
from tqdm import tqdm

import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.environ.get('HRDT_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(PROJECT_ROOT)
from models.encoder.t5_encoder import T5Embedder

# Get paths from environment variables or use defaults
MODEL_PATH = os.environ.get('T5_MODEL_PATH', "/data/lingxuan/weights/t5-v1_1-xxl")
CONFIG_PATH = os.environ.get('HRDT_CONFIG_PATH', os.path.join(PROJECT_ROOT, "configs/hrdt_pretrain.yaml"))
TARGET_DIR = os.environ.get('EGODEX_DATA_ROOT', "/share/hongzhe/datasets/egodex/")

# Multi-process configuration (can be overridden by environment variables)
NUM_GPUS = int(os.environ.get('NUM_GPUS', 8))  # Total number of GPUs
PROCESSES_PER_GPU = int(os.environ.get('PROCESSES_PER_GPU', 4))  # Processes per GPU
TOTAL_PROCESSES = NUM_GPUS * PROCESSES_PER_GPU  # Total processes


def collect_all_files():
    """Collect all files that need to be processed"""
    all_files = []
    
    # Get all directories that need to be processed
    dataset_dirs = []
    for item in os.listdir(TARGET_DIR):
        item_path = os.path.join(TARGET_DIR, item)
        if os.path.isdir(item_path):
            if item in ['extra', 'test'] or item.startswith('part'):
                dataset_dirs.append(item_path)
    
    dataset_dirs.sort()
    
    # Collect all files
    for dataset_dir in dataset_dirs:
        dataset_name = os.path.basename(dataset_dir)
        
        for task_name in os.listdir(dataset_dir):
            task_dir = os.path.join(dataset_dir, task_name)
            if os.path.isdir(task_dir):
                hdf5_files = [f for f in os.listdir(task_dir) if f.endswith('.hdf5')]
                hdf5_files.sort(key=lambda x: int(x.split('.')[0]))
                
                for hdf5_file in hdf5_files:
                    hdf5_path = os.path.join(task_dir, hdf5_file)
                    file_index = hdf5_file.split('.')[0]
                    pt_file = f"{file_index}.pt"
                    pt_path = os.path.join(task_dir, pt_file)
                    
                    # Only add files that haven't been processed yet
                    if not os.path.exists(pt_path):
                        all_files.append({
                            'hdf5_path': hdf5_path,
                            'pt_path': pt_path,
                            'task_name': task_name,
                            'dataset_name': dataset_name,
                            'file_index': file_index
                        })
    
    return all_files


def worker_process(process_id, gpu_id, file_list, progress_queue):
    """Worker process function"""
    try:
        # Set CUDA device
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        
        # Load configuration
        with open(CONFIG_PATH, "r") as fp:
            config = yaml.safe_load(fp)
        
        # Initialize T5 encoder
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH, 
            model_max_length=config["dataset"]["tokenizer_max_length"], 
            device=device
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        
        print(f"Process {process_id} (GPU {gpu_id}) starting to process {len(file_list)} files")
        
        processed_count = 0
        failed_count = 0
        
        for file_info in file_list:
            try:
                hdf5_path = file_info['hdf5_path']
                pt_path = file_info['pt_path']
                task_name = file_info['task_name']
                dataset_name = file_info['dataset_name']
                file_index = file_info['file_index']
                
                # Check again if file already exists (prevent duplicate processing by other processes)
                if os.path.exists(pt_path):
                    processed_count += 1
                    progress_queue.put(('processed', process_id))
                    continue
                
                # Extract language instructions from HDF5 file
                instructions = []
                with h5py.File(hdf5_path, 'r') as f:
                    # Extract main description
                    if 'llm_description' in f.attrs:
                        instruction = f.attrs['llm_description']
                        if isinstance(instruction, bytes):
                            instruction = instruction.decode('utf-8')
                        instructions.append(instruction)
                    
                    # Extract second description (if exists)
                    if 'llm_description2' in f.attrs:
                        instruction2 = f.attrs['llm_description2']
                        if isinstance(instruction2, bytes):
                            instruction2 = instruction2.decode('utf-8')
                        instructions.append(instruction2)
                
                if not instructions:
                    print(f"Process {process_id}: Warning: No language instructions found in {hdf5_path}")
                    failed_count += 1
                    progress_queue.put(('failed', process_id))
                    continue
                
                # Encode language instructions
                tokenized_res = tokenizer(
                    instructions, return_tensors="pt",
                    padding="longest",
                    truncation=True
                )
                tokens = tokenized_res["input_ids"].to(device)
                attn_mask = tokenized_res["attention_mask"].to(device)
                
                with torch.no_grad():
                    text_embeds = text_encoder(
                        input_ids=tokens,
                        attention_mask=attn_mask
                    )["last_hidden_state"].detach().cpu()
                
                attn_mask = attn_mask.cpu().bool()
                
                # Process encoding for each instruction and save
                if len(instructions) == 1:
                    # Only one instruction
                    text_embed = text_embeds[0][attn_mask[0]]
                    torch.save({
                        "instruction": instructions[0],
                        "embeddings": text_embed,
                        "task_name": task_name,
                        "dataset": dataset_name,
                        "file_index": file_index
                    }, pt_path)
                else:
                    # Multiple instructions
                    text_embed_1 = text_embeds[0][attn_mask[0]]
                    text_embed_2 = text_embeds[1][attn_mask[1]] if len(instructions) > 1 else None
                    
                    torch.save({
                        "instruction": instructions[0],
                        "instruction2": instructions[1] if len(instructions) > 1 else None,
                        "embeddings": text_embed_1,
                        "embeddings2": text_embed_2,
                        "task_name": task_name,
                        "dataset": dataset_name,
                        "file_index": file_index
                    }, pt_path)
                
                processed_count += 1
                progress_queue.put(('processed', process_id))
                
            except Exception as e:
                print(f"Process {process_id}: Error: Processing file {file_info['hdf5_path']} encountered problem: {str(e)}")
                failed_count += 1
                progress_queue.put(('failed', process_id))
                continue
        
        print(f"Process {process_id} (GPU {gpu_id}) completed: successful {processed_count}, failed {failed_count}")
        progress_queue.put(('done', process_id, processed_count, failed_count))
        
    except Exception as e:
        print(f"Process {process_id} encountered serious error: {str(e)}")
        progress_queue.put(('error', process_id, str(e)))


def progress_monitor(total_files, progress_queue, num_processes):
    """Progress monitoring function"""
    processed = 0
    failed = 0
    finished_processes = 0
    
    pbar = tqdm(total=total_files, desc="Processing progress")
    
    while finished_processes < num_processes:
        try:
            msg = progress_queue.get(timeout=1)
            
            if msg[0] == 'processed':
                processed += 1
                pbar.update(1)
            elif msg[0] == 'failed':
                failed += 1
                pbar.update(1)
            elif msg[0] == 'done':
                finished_processes += 1
                process_id, proc_processed, proc_failed = msg[1], msg[2], msg[3]
                print(f"\nProcess {process_id} completed: successful {proc_processed}, failed {proc_failed}")
            elif msg[0] == 'error':
                finished_processes += 1
                process_id, error_msg = msg[1], msg[2]
                print(f"\nProcess {process_id} error: {error_msg}")
                
        except:
            # Timeout, continue waiting
            continue
    
    pbar.close()
    return processed, failed


def main():
    print("Starting to collect all files to be processed...")
    all_files = collect_all_files()
    
    if not all_files:
        print("No files found to process!")
        return
    
    print(f"Found files to process")
    
    # Distribute files to different processes
    files_per_process = len(all_files) // TOTAL_PROCESSES
    file_lists = []
    
    for i in range(TOTAL_PROCESSES):
        start_idx = i * files_per_process
        if i == TOTAL_PROCESSES - 1:  # Last process handles all remaining files
            end_idx = len(all_files)
        else:
            end_idx = start_idx + files_per_process
        
        file_lists.append(all_files[start_idx:end_idx])
    
    print(f"Will use {NUM_GPUS} GPUs with {PROCESSES_PER_GPU} processes each, total {TOTAL_PROCESSES} processes")
    for i, file_list in enumerate(file_lists):
        gpu_id = i // PROCESSES_PER_GPU
        print(f"Process {i} (GPU {gpu_id}): {len(file_list)} files")
    
    # Create progress queue
    progress_queue = Queue()
    
    # Start progress monitoring process
    monitor_process = Process(
        target=progress_monitor, 
        args=(len(all_files), progress_queue, TOTAL_PROCESSES)
    )
    monitor_process.start()
    
    # Create and start worker processes
    processes = []
    for i in range(TOTAL_PROCESSES):
        gpu_id = i // PROCESSES_PER_GPU  # Calculate GPU ID
        process = Process(
            target=worker_process,
            args=(i, gpu_id, file_lists[i], progress_queue)
        )
        process.start()
        processes.append(process)
        time.sleep(0.5)  # Slight delay to avoid GPU initialization conflicts
    
    # Wait for all processes to complete
    for process in processes:
        process.join()
    
    # Wait for progress monitoring to complete
    monitor_process.join()
    
    print("\nAll processes completed!")


if __name__ == "__main__":
    # Set multi-process start method
    mp.set_start_method('spawn', force=True)
    main()
