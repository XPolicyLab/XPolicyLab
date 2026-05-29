import os
import h5py
import numpy as np
from tqdm import tqdm
import time

def add_task_info(folder_path):
    # 定义任务ID映射
    task_mapping = {
        'add_condiment': 0,
        'insert_flower': 1,
        'select_book': 2,
        'select_chemistry_tube': 3,
        'select_drink': 4,
        'select_ingredient': 5,
        'select_mahjong': 6,
        'select_poker': 7,
        'select_painting': 8,
        'select_toy': 9,
        'select_fruit': 10
    }
    
    # 遍历所有任务文件夹
    tasks_path = os.listdir(folder_path)
    for task_path in tasks_path:
        if not os.path.isdir(os.path.join(folder_path, task_path)):
            continue
            
        print(f"处理任务: {task_path}")
        task_id = task_mapping.get(task_path, -1)
        if task_id == -1:
            print(f"警告: 未知任务类型 {task_path}")
            continue
            
        # 遍历该任务下的所有hdf5文件
        task_dir = os.path.join(folder_path, task_path)
        for file_name in tqdm(os.listdir(task_dir)):
            if not file_name.endswith('.hdf5'):
                continue
                
            file_path = os.path.join(task_dir, file_name)
            while True:
                try:
                    with h5py.File(file_path, 'r+') as f:
                        # 获取数据组的键
                        data_key = list(f['data'].keys())[0]
                        data_group = f['data'][data_key]
                        
                        # 检查meta_info是否存在
                        if 'meta_info' not in data_group:
                            meta_info = data_group.create_group('meta_info')
                        else:
                            meta_info = data_group['meta_info']
                        
                        # 添加或更新任务信息
                        if 'task_name' in meta_info:
                            del meta_info['task_name']
                        meta_info.create_dataset('task_name', data=task_path.encode('utf-8'))
                        
                        if 'task_id' in meta_info:
                            del meta_info['task_id']
                        meta_info.create_dataset('task_id', data=np.array([task_id], dtype=np.int32))
                        break
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
                    time.sleep(1)
                    

if __name__ == "__main__":
    folder_path = "dataset/vlabench_mp"
    add_task_info(folder_path)
    print("完成添加任务信息!") 