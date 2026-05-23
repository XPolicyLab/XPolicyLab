# Convert the R3M-formatted Franka kitchen data to the LLaVA format
output_path = "/media/rl/HDD/data/data/franka_kitchen/frankakitchen_llava"
import pickle
import json
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
TASK = {
    'kitchen_sdoor_open-v3':"Open the sliding door", 
    'kitchen_micro_open-v3':"Open the microwave oven", 
    'kitchen_light_on-v3':"Toggle the light switch", 
    'kitchen_ldoor_open-v3':"Open the cabinet door",
    'kitchen_knob1_on-v3':"Rotate the round stovetop knob"
}

def franka_kitchen2llava_format(data_path:str, ratio:float, view):
    os.makedirs(os.path.join(output_path, view), exist_ok=True)
    os.makedirs(os.path.join(output_path, view, 'images'), exist_ok=True)
    
    pickle_path = os.listdir(data_path)
    pickle_path = [p for p in pickle_path if p.endswith('.pickle')]
    all_task_demo_paths = []
    
    t_bar = tqdm(total=1000 * ratio)
    
    try:
        with open(os.path.join(data_path, f'mean_var_{view}.json'), 'r') as f:
            mean_std = json.load(f)
    except:
        mean_std = {'action':{},'state':{}}
    
    
    data_processed = []
    eval_data = []
    train_data = []
    for p in pickle_path:
        cur_data = []

        if p.split('/')[-1] not in mean_std['action'].keys():
            mean_std['action'][p.split('/')[-1]] = {}
            mean_std['state'][p.split('/')[-1]] = {}

        action_all = []
        state_all = []
        # print(p.split('.')[0])
        max_action = [0] * 9
        min_action = [1000000] * 9
        max_state = [0] * 9
        min_state = [1000000] * 9
        
        demo_paths_loc = os.path.join(data_path, p)
        demo_paths = pickle.load(open(demo_paths_loc, 'rb'))
        all_task_demo_paths += demo_paths[:int(ratio*200)]
        # print(all_task_demo_paths[0].keys())
        # print(all_task_demo_paths[0]['actions'].shape)
        for idx, each in enumerate(demo_paths[:int(ratio*200)]): # trajectory id 
            traj_len = each['images'].shape[0]
            
            # normalize action and state
            m_a,v_a = np.array([mean_std['action'][p.split('/')[-1]]['mean']]), np.array([mean_std['action'][p.split('/')[-1]]['var']])
            m_s,v_s = np.array([mean_std['state'][p.split('/')[-1]]['mean']]), np.array([mean_std['state'][p.split('/')[-1]]['var']])
            each['actions'] = (each['actions'] - m_a) /np.sqrt(v_a)
            # print(each['observations'][:,:9].shape)
            each['observations'][:,:9] = (each['observations'][:,:9] - m_s) / np.sqrt(v_s)
            # #############################
            for i in range(traj_len): # frame id
                t = {
                    "id": "",
                    "image": "",
                    'state': [],
                    'action': [],
                    "conversations": [{"from": "human", "value": "<image>\n"}, {"from": "gpt", "value": " "}]
                }
                img_p = os.path.join(output_path, view, 'images', f'{p.split(".")[0]}_{idx}_{i}.png')
                # print(each['images'].shape)
                # break
                if not os.path.exists(img_p):
                    Image.fromarray(each['images'][i]).save(img_p)
                t['image'] = img_p
                t['id'] = img_p.split('/')[-1]
                t["conversations"][0]["value"] += TASK[p.split('.')[0]]
                t['state'] = each['observations'][i].tolist()
                t['action'] = each['actions'][i].tolist()
                # print(t['action'])
                ########################################查看一下数据范围
                # for j,a in enumerate(zip(t['action'])):
                #     if max_action[j] < a:
                #         max_action[j] = a
                #     if min_action[j] > a:
                #         min_action[j] = a
                # for j,a in enumerate(t['state'][:9]):
                #     if max_state[j] < a:
                #         max_state[j] = a
                #     if min_state[j] > a:
                #         min_state[j] = a
                action_all.append(t['action'])
                state_all.append(t['state'][:9])
                
                data_processed.append(t)
                cur_data.append(t)

            t_bar.update(1)
            #     print(t)
            #     break
            # break
        # print(p.split('/')[-1])
        # print(max_action,min_action)
        # print(max_state,min_state)
    
        mean_action = np.mean(np.array(action_all), axis=0)
        var_action = np.var(np.array(action_all), axis=0)

        mean_state = np.mean(np.array(state_all), axis=0)
        var_state = np.var(np.array(state_all), axis=0)
        
        mean_std['action'][p.split('/')[-1]]['mean'] = mean_action.tolist()
        mean_std['action'][p.split('/')[-1]]['var'] = var_action.tolist()
        
        mean_std['state'][p.split('/')[-1]]['mean'] = mean_state.tolist()
        mean_std['state'][p.split('/')[-1]]['var'] = var_state.tolist()
        eval_data += cur_data[int(200*50*0.9):]
        train_data += cur_data[:int(200*50*0.9)]
#         print("action")
#         print(mean_action, var_action)

#         print("state")
#         print(mean_state, var_state)
    # print(mean_std)
    
    # with open(f'mean_var_{view}.json', 'w') as f:
    #     json.dump(mean_std,f)
    
    # with open('action_all.txt', 'w') as f:
    #     for a in action_all:
    #         f.write(str(a) + '\n')
    # with open('state_all.txt', 'w') as f:
    #     for s in state_all:
    #         f.write(str(s) + '\n')
    
    # with open(os.path.join(output_path, view, f"std_{view}_50k.json"), "w") as f:
    #     json.dump(data_processed, f, indent=4)
    print(len(train_data), len(eval_data))
    with open(os.path.join(output_path, view, f"std_eval_{view}_50k.json"), "w") as f:
        json.dump(eval_data, f, indent=4)
    with open(os.path.join(output_path, view, f"std_train_{view}_50k.json"), "w") as f:
        json.dump(train_data, f, indent=4)
    
for view in ['default', 'left_cap2', 'right_cap2']: 
    data_path = f"/media/rl/HDD/data/data/franka_kitchen/FrankaKitchen/{view}"
    franka_kitchen2llava_format(data_path, 1, view)