import os
import h5py
import numpy as np
import cv2
import open3d as o3d
import mediapy
import random
import json

norm_stat_path = "/mnt/data/zhangkaidong/VLABench/third_party/openpi/pi0-fast-primitive/assets/joey/vlabench_primitive/norm_stats.json"
# with open(norm_stat_path, 'r') as f:
#     norm_stat = json.load(f)
#     norm_stat = norm_stat['norm_stats']
#     state = norm_stat['state']
#     actions  = norm_stat['actions']
#     min = (np.array(state['q01'])-np.array(state['mean']))/np.array(state['std'])
#     max = (np.array(state['q99'])+np.array(state['mean']))/np.array(state['std'])
#     print(min,max)
#     min = (np.array(actions['q01'])-np.array(actions['mean']))/np.array(actions['std'])
#     max = (np.array(actions['q99'])+np.array(actions['mean']))/np.array(actions['std'])
#     print(min,max)
# 打开一个HDF5文件
file_name = '/mnt/data/zhangkaidong/VLABench/dataset/vlabench_primitive_ft_dataset/remote-home1/sdzhang/datasets/VLABench_release/primitive/select_poker/episode_122.hdf5'
data_path = "/mnt/data/zhangkaidong/VLABench/dataset/select_toy"

files = os.listdir(data_path)
actions = []
for file in files:
    if file.endswith(".hdf5"):
        file_name = os.path.join(data_path, file)
        with h5py.File(file_name, 'r') as file:
            # 列出文件中所有的组
            # print("Keys: %s" % file.keys())
            data = file['data'][list(file['data'].keys())[0]]
            # print('data',data.keys())
            trajectory = data['trajectory'][:]
            q_state = data['observation']['q_state'][:]
            ee_state = data['observation']['ee_state'][:]
            actions = trajectory[:]
            for i in range(0, 3):
                print('i,',actions[i][:3],ee_state[i][:3])
        break

# print('q_state',np.diff(q_state[:,:3],axis=0).max(),np.diff(q_state[:,:3],axis=0).min())

# # actions = np.array(actions)
# # # 求actions的mean和std
# # mean = actions.mean(axis=0)
# # std = actions.std(axis=0)
# # max = actions.max(axis=0)
# # min = actions.min(axis=0)
# # mean[-2] = 0
# # std[-2] = 1
# # mean[-1] = 0
# # std[-1] = 1
# # print('mean',mean)
# # print('std',std)
# # print('max',max)
# # print('min',min)
# # norm_actions = (actions - mean) / std
# # output_actions = np.concatenate([actions[:,2:3],norm_actions[:,2:3]],axis=1)

# # print('actions',actions[:,:3])
# # print('actions',output_actions[:,:])
# # print('norm_actions',norm_actions[:,:3])
# # print(norm_actions[:,:3].max(),norm_actions[:,:3].min())
# # print(norm_actions[:,3:6].max(),norm_actions[:,3:6].min())
# # print(norm_actions.argmax(),norm_actions.argmin())
