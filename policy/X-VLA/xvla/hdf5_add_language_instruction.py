import h5py
import numpy as np
from pathlib import Path
import os
import json

# Data directory path
data_dir = Path("rmbench_dataset/rearrange_blocks/demo_clean/data") # change to your data directory
# Instructions directory path
instructions_dir = Path("rmbench_dataset/rearrange_blocks/demo_clean/instructions") # change to your instructions directory

print(f"Data directory: {data_dir}")
print(f"Instructions directory: {instructions_dir}")
print(f"\nStarting to process all hdf5 files in {data_dir}...\n")

# Iterate through all hdf5 files
hdf5_files = sorted(data_dir.glob("*.hdf5"))
print(f"Found {len(hdf5_files)} hdf5 files\n")

for hdf5_file in hdf5_files:
    print(f"Processing: {hdf5_file.name}")
    
    # Extract episode name from hdf5 filename (remove extension)
    episode_name = hdf5_file.stem  # e.g., episode0.hdf5 -> episode0
    
    # Build corresponding JSON file path
    json_file = instructions_dir / f"{episode_name}.json"
    
    # Check if JSON file exists
    if not json_file.exists():
        print(f"  ❌ Warning: Corresponding JSON file {json_file.name} not found, skipping this file")
        continue
    
    # Read JSON file
    try:
        with open(json_file, 'r', encoding='utf-8') as jf:
            json_data = json.load(jf)
        
        # Get "seen" key value (take first element of array)
        if "seen" not in json_data:
            print(f"  ❌ Warning: 'seen' key not found in JSON file, skipping this file")
            continue
        
        seen_list = json_data["seen"]
        if not seen_list or len(seen_list) == 0:
            print(f"  ❌ Warning: 'seen' array is empty, skipping this file")
            continue
        
        language_instruction = seen_list[0]  # Take first element
        print(f"  📝 Read instruction from {json_file.name}: '{language_instruction[:50]}...'")
        
    except json.JSONDecodeError as e:
        print(f"  ❌ Error: JSON file parsing failed - {e}, skipping this file")
        continue
    except Exception as e:
        print(f"  ❌ Error: Failed to read JSON file - {e}, skipping this file")
        continue
    
    # Open hdf5 file in read-write mode
    try:
        with h5py.File(hdf5_file, "r+") as f:
            # If language_instruction key already exists, delete it first to overwrite
            if "language_instruction" in f.keys():
                del f["language_instruction"]
                print(f"  ⚠️  {hdf5_file.name} already has language_instruction, will overwrite")
            
            # Add or overwrite language_instruction key
            # Use string type to store a single string (scalar)
            f.create_dataset("language_instruction", data=language_instruction, dtype=h5py.string_dtype(encoding='utf-8'))
            print(f"  ✓ Added/overwritten language_instruction: '{language_instruction[:60]}...'")
    except Exception as e:
        print(f"  ❌ Error: Failed to process hdf5 file - {e}, skipping this file")
        continue

print(f"\nAll files processed successfully!")

