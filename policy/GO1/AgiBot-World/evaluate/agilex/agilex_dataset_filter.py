import os
import h5py
import numpy as np
import argparse


def analyze_outliers_by_std(root_dir: str, data_key: str, std_threshold: float = 5.0):
    """
    Analyzes HDF5 files to find outliers based on the global mean and standard deviation.
    An outlier is defined as a value outside of mean ± (std_threshold * std).

    Args:
        root_dir (str): Directory containing the .hdf5 files.
        data_key (str): Key to the dataset within the HDF5 files.
        std_threshold (float): Number of standard deviations to use as the threshold.
    """
    files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".hdf5")]
    if not files:
        print(f"No .hdf5 files found in {root_dir}")
        return

    # --- 1. First pass: Calculate global mean and std ---
    print(f"--- Analyzing {data_key} ---")
    print("Step 1: Calculating global statistics across all files...")
    all_data = []
    for file_path in files:
        try:
            with h5py.File(file_path, "r") as dataset:
                if data_key in dataset:
                    all_data.append(dataset[data_key][:])
        except Exception as e:
            print(f"Warning: Could not process file {file_path} during stats calculation: {e}")

    if not all_data:
        print(f"No data found for key '{data_key}' in any files.")
        return

    combined_data = np.concatenate(all_data, axis=0)
    global_mean = np.mean(combined_data, axis=0)
    global_std = np.std(combined_data, axis=0)

    # Avoid division by zero for dimensions with no variance
    global_std[global_std == 0] = 1e-9

    print("Global statistics calculated.")

    # --- 2. Second pass: Find files with outliers ---
    print(f"\nStep 2: Detecting files with values outside mean ± {std_threshold}*std...")
    outlier_files = {}

    for file_path in files:
        try:
            with h5py.File(file_path, "r") as dataset:
                if data_key not in dataset:
                    continue

                data = dataset[data_key][:]

                # Calculate z-score for all data points
                z_scores = np.abs((data - global_mean) / global_std)

                # Find where z_score exceeds the threshold
                outlier_mask = z_scores > std_threshold

                if np.any(outlier_mask):
                    # Find the location and value of the most extreme outlier in this file
                    max_z_score_index = np.unravel_index(np.argmax(z_scores), z_scores.shape)
                    row, dim = max_z_score_index

                    outlier_info = {"value": data[row, dim], "dim": dim, "z_score": z_scores[row, dim]}
                    outlier_files[os.path.basename(file_path)] = outlier_info

        except Exception as e:
            print(f"Warning: Could not process file {file_path} during outlier detection: {e}")

    # --- 3. Report results ---
    sorted_files = []
    if not outlier_files:
        print("\nNo files with extreme outliers found.")
    else:
        print(f"\nFound {len(outlier_files)} file(s) with outliers (showing most extreme outlier per file):")
        print("-" * 120)
        print(f"{'File':<70} | {'Dimension':<10} | {'Value':>12} | {'Z-Score (>5)':>12}")
        print("-" * 120)
        # Sort by Z-score to show the most extreme files first
        sorted_files = sorted(outlier_files.items(), key=lambda item: item[1]["z_score"], reverse=True)
        for file_name, info in sorted_files:
            print(f"{root_dir+file_name:<70} | {info['dim']:<10} | {info['value']:12.4f} | {info['z_score']:12.2f}")
    print("-" * 120)
    return sorted_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument(
        "--std_threshold", type=float, default=5.0, help="Standard deviation threshold for outlier detection"
    )
    args = parser.parse_args()
    root_dir = args.root_dir
    outlier_files = analyze_outliers_by_std(root_dir, data_key="observations/qpos", std_threshold=args.std_threshold)
