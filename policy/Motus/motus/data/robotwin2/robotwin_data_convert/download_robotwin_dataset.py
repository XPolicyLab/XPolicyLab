#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RobotTwin2.0 Dataset Downloader

Downloads specific ZIP files from TianxingChen/RoboTwin2.0 dataset on HuggingFace.
Only downloads the required aloha-agilex_clean_50.zip and aloha-agilex_randomized_500.zip 
files for each task, skipping other data types.

Usage:
    python download_robotwin_dataset.py --output_dir /path/to/save
    python download_robotwin_dataset.py --tasks adjust_bottle clean_mirror --output_dir /path/to/save

"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import zipfile
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RobotTwinDownloader:
    """
    Downloader for RobotTwin2.0 dataset ZIP files
    """
    
    def __init__(self, use_mirror: bool = True):
        self.repo_id = "TianxingChen/RoboTwin2.0"
        self.required_files = [
            "aloha-agilex_clean_50.zip",
            "aloha-agilex_randomized_500.zip"
        ]
        
        # Setup HF mirror for faster downloads in China
        if use_mirror:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            logger.info("Using HuggingFace mirror: https://hf-mirror.com")
    
    def get_available_tasks(self) -> List[str]:
        """
        Get list of available tasks from HuggingFace repository
        
        Returns:
            List of task names
        """
        try:
            from huggingface_hub import list_repo_files
            
            # Get all files in the dataset directory
            files = list_repo_files(self.repo_id, repo_type="dataset")
            
            # Extract task names from dataset/ files
            tasks = set()
            for file in files:
                if file.startswith("dataset/") and file.endswith(".zip"):
                    # Format: dataset/{task_name}/{file}.zip
                    parts = file.split("/")
                    if len(parts) >= 3:
                        task_name = parts[1]
                        zip_name = parts[2]
                        # Only include tasks that have our required files
                        if zip_name in self.required_files:
                            tasks.add(task_name)
            
            return sorted(list(tasks))
            
        except Exception as e:
            logger.error(f"Failed to get available tasks: {e}")
            return []
    
    def download_task_files(self, task_name: str, output_dir: str) -> bool:
        """
        Download required ZIP files for a specific task
        
        Args:
            task_name: Name of the task (e.g., 'adjust_bottle')
            output_dir: Directory to save downloaded files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from huggingface_hub import hf_hub_download
            
            task_output_dir = Path(output_dir) / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            
            for zip_file in self.required_files:
                try:
                    # Download from dataset/{task_name}/{zip_file}
                    repo_file_path = f"dataset/{task_name}/{zip_file}"
                    local_file_path = task_output_dir / zip_file
                    
                    logger.info(f"Downloading {repo_file_path}...")
                    
                    downloaded_file = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=repo_file_path,
                        repo_type="dataset",
                        local_dir=output_dir,
                        local_dir_use_symlinks=False
                    )
                    
                    logger.info(f"✓ Downloaded: {zip_file}")
                    success_count += 1
                    
                except Exception as e:
                    logger.warning(f"✗ Failed to download {zip_file} for {task_name}: {e}")
            
            if success_count > 0:
                logger.info(f"Downloaded {success_count}/{len(self.required_files)} files for {task_name}")
                return True
            else:
                logger.error(f"Failed to download any files for {task_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading files for {task_name}: {e}")
            return False
    
    def extract_zip_files(self, task_dir: Path, cleanup: bool = True) -> bool:
        """
        Extract downloaded ZIP files
        
        Args:
            task_dir: Directory containing ZIP files
            cleanup: Whether to delete ZIP files after extraction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success_count = 0
            
            for zip_file in self.required_files:
                zip_path = task_dir / zip_file
                if zip_path.exists():
                    try:
                        logger.info(f"Extracting {zip_file}...")
                        
                        # Extract to task directory
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(task_dir)
                        
                        logger.info(f"✓ Extracted: {zip_file}")
                        success_count += 1
                        
                        # Remove ZIP file if cleanup is requested
                        if cleanup:
                            zip_path.unlink()
                            logger.debug(f"Cleaned up: {zip_file}")
                            
                    except Exception as e:
                        logger.warning(f"✗ Failed to extract {zip_file}: {e}")
                else:
                    logger.warning(f"ZIP file not found: {zip_file}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error extracting ZIP files: {e}")
            return False
    
    def download_dataset(
        self, 
        output_dir: str, 
        tasks: Optional[List[str]] = None,
        extract: bool = True,
        cleanup: bool = True
    ) -> bool:
        """
        Download RobotTwin2.0 dataset
        
        Args:
            output_dir: Directory to save the dataset
            tasks: List of specific tasks to download (None for all)
            extract: Whether to extract ZIP files
            cleanup: Whether to delete ZIP files after extraction
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Install huggingface_hub if needed
            try:
                import huggingface_hub
            except ImportError:
                logger.info("Installing huggingface_hub...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
                import huggingface_hub
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get available tasks
            if tasks is None:
                logger.info("Fetching available tasks...")
                available_tasks = self.get_available_tasks()
                if not available_tasks:
                    logger.error("No tasks found")
                    return False
                tasks = available_tasks
                logger.info(f"Found {len(tasks)} tasks: {', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''}")
            else:
                logger.info(f"Downloading specific tasks: {', '.join(tasks)}")
            
            # Download each task
            successful_tasks = 0
            
            for task_name in tasks:
                logger.info(f"Processing task: {task_name}")
                
                # Download ZIP files
                if self.download_task_files(task_name, str(output_path)):
                    successful_tasks += 1
                    
                    # Extract if requested
                    if extract:
                        task_dir = output_path / "dataset" / task_name
                        if task_dir.exists():
                            self.extract_zip_files(task_dir, cleanup)
                else:
                    logger.warning(f"Failed to download task: {task_name}")
            
            logger.info(f"Successfully processed {successful_tasks}/{len(tasks)} tasks")
            
            # Final summary
            if successful_tasks > 0:
                total_size = sum(
                    f.stat().st_size 
                    for f in output_path.rglob("*") 
                    if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)
                
                logger.info("=" * 50)
                logger.info("DOWNLOAD COMPLETE!")
                logger.info(f"Downloaded: {successful_tasks}/{len(tasks)} tasks")
                logger.info(f"Total size: {size_mb:.1f} MB")
                logger.info(f"Location: {output_dir}")
                logger.info("=" * 50)
                return True
            else:
                logger.error("No tasks were successfully downloaded")
                return False
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Download RobotTwin2.0 dataset ZIP files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/share/dataset/robotwin/dataset/",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Specific tasks to download (default: all available)"
    )
    parser.add_argument(
        "--no_extract",
        action="store_true",
        help="Don't extract ZIP files"
    )
    parser.add_argument(
        "--keep_zip",
        action="store_true", 
        help="Keep ZIP files after extraction"
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List available tasks and exit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no_mirror",
        action="store_true",
        help="Don't use HuggingFace mirror (slower in China)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize downloader
    downloader = RobotTwinDownloader(use_mirror=not args.no_mirror)
    
    # List tasks if requested
    if args.list_tasks:
        logger.info("Fetching available tasks...")
        tasks = downloader.get_available_tasks()
        if tasks:
            print("\nAvailable tasks:")
            for i, task in enumerate(tasks, 1):
                print(f"  {i:2d}. {task}")
            print(f"\nTotal: {len(tasks)} tasks")
        else:
            print("No tasks found")
        return
    
    # Download dataset
    success = downloader.download_dataset(
        output_dir=args.output_dir,
        tasks=args.tasks,
        extract=not args.no_extract,
        cleanup=not args.keep_zip
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
