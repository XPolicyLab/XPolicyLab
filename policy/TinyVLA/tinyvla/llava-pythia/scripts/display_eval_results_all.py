import os

ckpt_p = "/data/private/wenjj/llava-pythia/checkpoint_all/pythia_{msz}/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-view_channel_cat_lora2/checkpoint-{ckpt}"

model_size = ['14M', '70M', '160M', '410M', '1B']

TASK = ('kitchen_sdoor_open-v3', 'kitchen_micro_open-v3', 'kitchen_light_on-v3', 'kitchen_ldoor_open-v3','kitchen_knob1_on-v3')

task_id = 0
for task_id in range(5):
    # print(f"{TASK[task_id]}")
    with open('eval_results_all.txt', 'w') as f:
        f.write(f"eval results on Franka Kitchen {TASK[task_id]}:\n")

        for msz in model_size:
            f.write(f"###################Model size:{msz}###################\n")
            for i in range(2,6):
                ckpt = str(i*1000)
                p = ckpt_p.replace('{msz}', msz).replace('{ckpt}', ckpt)
                try:
                    with open(os.path.join(p, f"{TASK[task_id]}.txt"), 'r') as f1:
                        content = f1.read()
                        content = content.split('\n')
#                         for e in content:
#                             if e == "":
#                                 continue
#                             if '50_' not in e:
#                                 continue

#                             f.write(f"ckpt_{i*1000}:{e.strip()}\n")
                        f.write(f"ckpt_{ckpt}:{content[-1].strip()}\n")
                except Exception as e:
                    print(e)
                    pass
    with open('eval_results_all.txt', 'r') as f:
        data = f.read()
        print(data)
            
