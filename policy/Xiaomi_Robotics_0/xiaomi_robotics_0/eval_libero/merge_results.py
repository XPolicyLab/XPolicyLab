# Copyright (C) 2026 Xiaomi Corporation.
import json
import os
import glob
import requests  # Add missing import
from typing import Dict  # Add missing import


def merge_json_results(result_dir="./results"):
    """
    Merge JSON results from multiple task files and calculate statistics
    """
    # Find all JSON files
    json_pattern = os.path.join(result_dir, "result_*.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print("No JSON files found matching pattern:", json_pattern)
        return

    print(f"Found {len(json_files)} JSON files:")
    for f in sorted(json_files):
        print(f"  - {os.path.basename(f)}")
    print()

    # Store all results
    all_tasks = []
    task_ids = set()
    success_rates = {}

    # Process each JSON file
    for json_file in sorted(json_files):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            print(f"Processing {os.path.basename(json_file)}:")
            print(f"  Raw data type: {type(data).__name__}")

            # Handle the case where data is a list of tasks
            if isinstance(data, list):
                for task in data:
                    if isinstance(task, dict):
                        task_id = int(task.get("task_id", "unknown"))
                        instruction = task.get("instruction", "unknown")
                        success_rate = task.get("success_rate", 0)

                        task_ids.add(task_id)
                        success_rates[task_id] = success_rate
                        all_tasks.append({"task_id": task_id, "instruction": instruction, "success_rate": success_rate, "source_file": os.path.basename(json_file)})

                        print(f"  Task {task_id}: {success_rate} - {instruction[:50]}{'...' if len(instruction) > 50 else ''}")

            # Handle the case where data is a single task dict
            elif isinstance(data, dict):
                task_id = data.get("task_id", "unknown")
                instruction = data.get("instruction", "unknown")
                success_rate = data.get("success_rate", 0)

                task_ids.add(task_id)
                success_rates[task_id] = success_rate
                all_tasks.append({"task_id": task_id, "instruction": instruction, "success_rate": success_rate, "source_file": os.path.basename(json_file)})

                print(f"  Task {task_id}: {success_rate} - {instruction}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if not all_tasks:
        print("No valid task data found in JSON files")
        return

    # Print individual task results
    print("\nIndividual Task Results:")
    print("=" * 80)
    print(f"{'Task ID':<10} {'Success Rate':<15} {'Instruction'}")
    print("-" * 80)

    sorted_task_ids = sorted(task_ids)
    total_success_rate = 0

    for task_id in sorted_task_ids:
        success_rate = success_rates.get(task_id, 0)
        instruction = ""

        # Find the corresponding task to get instruction
        for task in all_tasks:
            if task["task_id"] == task_id:
                instruction = task["instruction"][:50] + "..." if len(task["instruction"]) > 50 else task["instruction"]
                break

        print(f"{task_id:<10} {success_rate:<15.4f} {instruction}")
        total_success_rate += success_rate

    # Calculate overall averages
    num_tasks = len(sorted_task_ids)
    if num_tasks > 0:
        overall_success_rate = total_success_rate / num_tasks

        print("-" * 80)
        print(f"{'Overall':<10} {overall_success_rate:<15.4f}")
        print()

        # Print summary
        print("Summary Statistics:")
        print(f"  Total Tasks: {num_tasks}")
        print(f"  Overall Success Rate: {overall_success_rate:.4f}")

    # Save merged results
    merged_data = {
        "individual_results": [{"task_id": task["task_id"], "instruction": task["instruction"], "success_rate": task["success_rate"]} for task in all_tasks],
        "summary": {"total_tasks": num_tasks, "overall_success_rate": overall_success_rate if num_tasks > 0 else 0},
    }

    output_file = os.path.join(result_dir, "merged_results.json")
    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(f"\nMerged results saved to: {output_file}")

    return merged_data


if __name__ == "__main__":
    # You can specify the directory containing JSON files
    import sys

    result_dir = sys.argv[1] if len(sys.argv) > 1 else "./results"
    merge_json_results(result_dir)
