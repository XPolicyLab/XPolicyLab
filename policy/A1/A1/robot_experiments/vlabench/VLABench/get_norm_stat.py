#!/usr/bin/env python3
"""
计算动作数据的统计信息并保存为YAML文件。
该脚本用于处理HDF5文件中的轨迹数据，计算动作的标准化统计信息。
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import h5py
import yaml
from pathlib import Path


class ActionStatisticsCalculator:
    """计算动作数据统计信息的类"""
    
    def __init__(self, dataset_path: str):
        """
        初始化计算器
        
        Args:
            dataset_path: 数据集根目录的路径
        """
        self.dataset_path = Path(dataset_path)
        self.actions: List[np.ndarray] = []
        
    def load_trajectory_data(self) -> None:
        """从HDF5文件加载轨迹数据并计算动作"""
        for data_path in self.dataset_path.iterdir():
            if not data_path.is_dir():
                continue
                
            for file_path in data_path.glob("*.hdf5"):
                self._process_hdf5_file(file_path)
    
    def _process_hdf5_file(self, file_path: Path) -> None:
        """
        处理单个HDF5文件
        
        Args:
            file_path: HDF5文件的路径
        """
        try:
            with h5py.File(file_path, 'r') as file:
                data = file['data'][list(file['data'].keys())[0]]
                trajectory = data['trajectory'][:]

                for i in range(1, len(trajectory)):
                    action = trajectory[i].copy()
                    # 计算相对位置变化
                    # action[:3] = trajectory[i][:3] - trajectory[i-1][:3]
                    
                    # 如果阶段发生变化，重置动作
                    if data['stage'][i] != data['stage'][i-1]:
                        action[:6] = 0
                        
                    self.actions.append(action)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            raise e
    
    def calculate_statistics(self) -> Dict[str, np.ndarray]:
        """
        计算动作数据的统计信息
        
        Returns:
            包含均值、标准差、最大值和最小值的字典
        """
        actions_array = np.array(self.actions)
        
        # 初始化统计值
        # mean = np.zeros_like(actions_array).mean(axis=0)
        # std = np.abs(actions_array).max(axis=0)
        mean = np.mean(actions_array, axis=0)
        std = np.std(actions_array, axis=0)
        max_vals = actions_array.max(axis=0)
        min_vals = actions_array.min(axis=0)
        
        # # 设置旋转部分的归一化参数
        # mean[3:] = 0
        # std[3:] = 1
        
        return {
            'mean': mean,
            'std': std,
            'max': max_vals,
            'min': min_vals
        }
    
    def save_statistics(self, stats: Dict[str, np.ndarray], output_path: str) -> None:
        """
        将统计信息保存到YAML文件
        
        Args:
            stats: 统计信息字典
            output_path: 输出文件路径
        """
        stats_dict = {
            key: value.tolist() for key, value in stats.items()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(stats_dict, f)
        
        print(f'统计信息已保存到: {output_path}')
    
    def print_normalization_info(self, stats: Dict[str, np.ndarray]) -> None:
        """
        打印归一化信息
        
        Args:
            stats: 统计信息字典
        """
        actions_array = np.array(self.actions)
        norm_actions = (actions_array - stats['mean']) / stats['std']
        
        print('统计信息:')
        print(f'均值: {stats["mean"]}')
        print(f'标准差: {stats["std"]}')
        print(f'最大值: {stats["max"]}')
        print(f'最小值: {stats["min"]}')
        print('\n归一化后的范围:')
        print(f'位置分量范围: [{norm_actions[:,:3].min():.3f}, {norm_actions[:,:3].max():.3f}]')
        print(f'旋转分量范围: [{norm_actions[:,3:6].min():.3f}, {norm_actions[:,3:6].max():.3f}]')


def main():
    """主函数"""
    # 配置路径
    dataset_path = "/mnt/data/zhangkaidong/VLABench/dataset/vlabench_mp"
    output_path = "VLABench/configs/action_stat.yaml"
    
    # 创建计算器实例并处理数据
    calculator = ActionStatisticsCalculator(dataset_path)
    calculator.load_trajectory_data()
    
    # 计算并保存统计信息
    stats = calculator.calculate_statistics()
    calculator.save_statistics(stats, output_path)
    calculator.print_normalization_info(stats)


if __name__ == "__main__":
    main()