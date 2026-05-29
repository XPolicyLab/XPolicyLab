import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Tuple
from tqdm.auto import tqdm
import math
from abc import abstractmethod
import numpy as np
import scipy.stats

from a1.torch_util import move_to_device, get_global_rank, get_world_size, is_distributed


def get_similarity(f_x1s, f_x2s, detach_f1=False):
    if detach_f1:
        f_x1s = f_x1s.detach()
    f_x1 = F.normalize(f_x1s, p=2., dim=-1, eps=1e-5)
    f_x2 = F.normalize(f_x2s, p=2., dim=-1, eps=1e-5)
    # loss = F.mse_loss(f_x1, f_x2, reduction="none").sum(-1).mean(0)
    sim = (f_x1 * f_x2).sum(-1)
    return sim


def logits_to_expected_value(logits, bin_boundaries):
    # Assuming logits is a tensor with shape (batch, num_bin)
    ndim = logits.ndim
    if ndim == 3:
        d1, d2 = logits.shape[:2]
        logits = logits.flatten(0, 1)
    batch, num_bin = logits.shape
    # Create a tensor representing the midpoints of each bin
    bin_midpoints = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    # Expand bin_midpoints to match the batch size
    bin_midpoints = bin_midpoints.unsqueeze(0).expand(batch, num_bin)
    # Convert logits to probabilities (softmax)
    probabilities = torch.softmax(logits, dim=1)
    # Calculate the expected value as the weighted sum of bin midpoints
    expected_values = torch.sum(probabilities * bin_midpoints, dim=1)
    if ndim == 3:
        expected_values = expected_values.reshape((d1, d2))
    return expected_values

def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16" or precision == "amp_bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


class BaseValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # dummy for DDP wrapper
        self.head = torch.nn.Linear(10, 10)
      
    @abstractmethod  
    def clear_hidden_state(self) -> None:
        pass
    
    @abstractmethod
    def update_memory(self):
        pass
    
    @abstractmethod
    def set_bin_boundaries(self, bin_boundaries):
        pass
           
   
 
class ActionValueNet(BaseValueNet):
    def __init__(self,  exit_list, exit_head, model, interval, threshold_type, anchor=False) -> None:
        super().__init__()
        self.exit_list = exit_list
        self.exit_head = exit_head
        self.model = model
        self.interval = interval # exit interval
        self.action_list = []
        # self.window_size = window_size
        self.threshold_type = threshold_type
        self.anchor = anchor
        
    def reset_actions(self):
        self.action_list = []
        
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    # def update_exit_hidden_state(self):
    #     """called after early exiting and executing an action in the environment"""
    #     self.exit_head.module.update_hidden_state()
        
    def get_ensemble_action(self):
        assert len(self.action_list) > 0
        actions, grippers = zip(*self.action_list[-2:])
        return torch.stack(actions, dim=0).mean(0), torch.stack(grippers, dim=0).mean(0)
        
    def forward(  # type: ignore
        self,
        feats,
        i=None,
        proprio=None,
        start_idx=0,
        end_idx=0,
        pos_offset=0,
        mode='infer',
        ):
        
        def get_delta(action1, action2):
            delta = torch.abs(action1 - action2)
            if self.threshold_type == 'mean':
                delta = delta.mean(-1)
            elif self.threshold_type == 'L2':
                delta = delta.pow(2).mean(-1).pow(0.5)
            elif self.threshold_type == 'max':
                delta = delta.max(-1)[0]
            elif self.threshold_type == 'cosine':
                delta = 1 - get_similarity(action1, action2)
            else:
                raise NotImplementedError
            return delta
            
        
        if mode == 'infer':
            assert i > 0, 'the first layer similarity is not implemented yet'
            if len(self.action_list) == 0:
                # Use previous layer's feature as pseudo previous action for the first check
                prev_layer_idx = max(0, i - 1)
                if self.model.config.action_head != 'flow_matching':
                    prev_action = self.model.predict_actions_by_hidden_states_by_idx(feats[prev_layer_idx], start_idx, end_idx)
                else:
                    prev_action = self.model.predict_actions_flow_matching(feats[:prev_layer_idx+1], proprio, pos_offset)
            elif i > 0 and i - self.interval < 0:
                if self.model.config.action_head != 'flow_matching':
                    prev_action = self.model.predict_actions_by_hidden_states_by_idx(feats[i-1], start_idx, end_idx)
                else:
                    prev_action = self.model.predict_actions_flow_matching(feats[:i], proprio, pos_offset)
            else:
                prev_action = self.action_list[-1]
            
            # action = self.exit_head(feats[i],)
            if self.model.config.action_head != 'flow_matching':
                action = self.model.predict_actions_by_hidden_states_by_idx(feats[i], start_idx, end_idx)
            else:
                action = self.model.predict_actions_flow_matching(
                    feats[:i+1], proprio, pos_offset, 
                    input_x=self.action_list[-1] if (self.anchor and len(self.action_list) > 0) else None)
            self.action_list.append(action)   
            delta = get_delta(action[0], prev_action[0])

            return delta, action
        # 这部分用于离线校准,mode='generate'
        else:
            assert 0 not in self.exit_list
            # exits_feat = [feats[i] for i in [0]+self.exit_list] # (n_exit+1, bs * action_seq_len, lang_len, d)
            
            # exits_action_list = []
            # lang_len, d = feats[0].shape[1:]
            # for seq_id in range(self.window_size//2-1, self.window_size-1):

            # (bs * action_seq_len, lang_len, d) -> (bs, seq_id, lang_len, d)
            # prev_time_feat = rand_layer_feat.reshape(-1, self.window_size, lang_len, d)[:, :seq_id, :, :]
            
            # exit_action = [] # (exit+1, bs, dim) or (exit+1, action_seq_len, dim)
            # for i in [0]+self.exit_list:
            #     # (bs * action_seq_len, lang_len, d) -> (bs, 1, lang_len, d)
            #     # last_time_feat = feats[i].reshape(-1, self.window_size, lang_len, d)[:, seq_id:seq_id+1, :, :]
            #     # combined_feat = torch.concat([prev_time_feat, last_time_feat], dim=1) # (bs, seq_id+1, lang_len, d)
        
            #     # self.exit_head.last_action = True
            #     # action = self.exit_head(combined_feat) # (bs, dim)
            #     action = self.model.predict_actions_by_hidden_states_by_idx(feats[i], proprio, start_idx, end_idx)
            #     # self.exit_head.last_action = False
            #     if action.ndim == 3:  # (batch, seq, action_dim)
            #         action = action[0]  # 取第一个batch（校准默认bs=1），此时形状为 (seq, action_dim)
            #     exit_action.append(action)   # (exit+1, seq, dim) 或 (exit+1, bs, dim)

            # 直接在 exit 维度上堆叠，不再引入额外的 size=1 维度
            # exit_action_stack = torch.stack(exit_action)  # (exit+1, seq, dim) 或 (exit+1, bs, dim)
            
            # 兼容两种输入：
            # - hidden_states: Tensor 列表 (每层形状: (B, L, D)) → 向量化堆叠与一次性前向
            # - attn_key_values (flow_matching): List[Tuple[Tensor, Tensor]] → 逐层截断 KV 并循环前向
            selected_layers = [0] + self.exit_list

            if self.model.config.action_head != 'flow_matching':
                # 向量化：将所有 exit 的隐藏态一次性送入动作头，减少前向调用次数
                layer_hiddens = [feats[i] for i in selected_layers]  # 每个形状: (B, L, D)
                # (E+1, B, L, D)
                stacked_hidden = torch.stack(layer_hiddens, dim=0)
                E_plus_1, B, L, D = stacked_hidden.shape
                # 展平到批维度，变为 ((E+1)*B, L, D)
                flat_hidden = stacked_hidden.flatten(0, 1)
                actions_flat = self.model.predict_actions_by_hidden_states_by_idx(flat_hidden, start_idx, end_idx)
                # 还原为 (E+1, B, ...)
                if actions_flat.ndim >= 3:
                    actions_stacked = actions_flat.reshape(E_plus_1, B, *actions_flat.shape[1:])
                    # 仅取第一个 batch（与原始实现一致：校准默认 bs=1）=> 形状 (E+1, seq, dim)
                    exit_action_stack = actions_stacked[:, 0]
                else:
                    exit_action_stack = actions_flat.reshape(E_plus_1, B, -1)[:, 0]
            else:
                # flow matching: feats 是按层的 (k, v) 序列
                actions_per_exit = []
                for layer_idx in selected_layers:
                    # 仅使用至第 layer_idx 层（含）的 KV 进行推理，模拟对应 exit
                    # 把上一层得到的noise action作为初始状态，传递给下一层
                    kvs_i = feats[:layer_idx + 1]
                    actions_i = self.model.predict_actions_flow_matching(
                        kvs_i, proprio, pos_offset,
                        input_x=actions_per_exit[-1] if (self.anchor and len(actions_per_exit) > 0) else None)
                    actions_per_exit.append(actions_i)
                # (E+1, B, horizon, dim)
                actions_stacked = torch.stack(actions_per_exit, dim=0)
                # 仅取第一个 batch（与原始实现一致：校准默认 bs=1）=> 形状 (E+1, horizon, dim)
                exit_action_stack = actions_stacked[:, 0]
            
            prev_actions = exit_action_stack[:-1]  # (n_exit, ...)
            last_actions = exit_action_stack[1:]   # (n_exit, ...)
            delta = get_delta(prev_actions, last_actions)
            # 将后两维展平为样本维度（兼容 (n_exit, seq) 或 (n_exit, bs*seq) 情况）
            if delta.ndim >= 3:
                delta = delta.flatten(1, 2)
            else:
                delta = delta.flatten(1)
            return delta
                    
        
class ExitController(torch.nn.Module):
    def __init__(self, value_net, exit_id_list, steps_per_stage, exit_dist='exp', leq=True, max_layer=12):
        super().__init__()
        self.value_net = value_net
        self.thresholds = None
        self.leq = leq
        self.exit_id_list = exit_id_list
        self.num_exit = len(self.exit_id_list)
        self.steps_per_stage = steps_per_stage
        self.exit_dist = exit_dist
        self.max_layer = min(max_layer - 1, self.exit_id_list[-1])
        # for debug
        # self.history_values = [[] for i in range(num_exit)]
        
    def _set_threshold_value(self, thresholds):
        real_num_exit = len([x for x in self.exit_id_list if x <= self.max_layer])
        assert len(thresholds) == real_num_exit
        self.thresholds = {self.exit_id_list[i] : thresholds[i]  for i in range(real_num_exit)}
        if get_global_rank() == 0:
            print('setting thresholds, ', thresholds)
        
        
    def set_threshold(self, args, model, dataloader, exit_ratio, model_name, values=None):  
        if values is None:  
            device_id = get_global_rank()
            if isinstance(self.value_net, ActionValueNet):
                pred_value_list, target_value_list = generate_action_values(args, model, self.value_net, dataloader, device_id=device_id)
            else:
                raise NotImplementedError

            # Gather data across ranks if distributed; otherwise, use local values
            if is_distributed():
                num_devices = get_world_size()
                pred_value_gathered = [torch.zeros_like(pred_value_list) for _ in range(num_devices)]
                torch.distributed.all_gather(pred_value_gathered, pred_value_list)
                pred_value_gathered = torch.cat(pred_value_gathered, dim=1)
            else:
                pred_value_gathered = pred_value_list
        else:
            pred_value_gathered = values
            
        n_stage, n_sample = pred_value_gathered.size() # (exit, bs * seq_length)

        _, sorted_idx = pred_value_gathered.sort(dim=1, descending=not self.leq)
        # print(f"*** sorted_idx: ",sorted_idx)
        # print(f"*** pred_value_gathered: ",pred_value_gathered)

        filtered = torch.zeros(n_sample)
        real_num_exit = len([x for x in self.exit_id_list if x <= self.max_layer])
        
        T = torch.Tensor(real_num_exit).fill_(-1e8) if self.leq else torch.Tensor(real_num_exit).fill_(1e8)
        
        
        if self.exit_dist == 'exp':
            probs = exit_ratio ** torch.arange(1, real_num_exit+1) # n (including the last exit)
    
        elif self.exit_dist == 'gauss':
            # Gaussian (normal) distribution centered around `num_exit // 2`
            center = exit_ratio
            std_dev = 1.0  # Arbitrary standard deviation to cover a significant range
            probs = torch.tensor([math.exp(-(i - center) ** 2 / (2 * std_dev ** 2)) for i in range(real_num_exit)])
        
        elif self.exit_dist == 'gamma':
            # Gamma distribution
            x = torch.arange(1, real_num_exit + 1, dtype=torch.float32)
            shape = exit_ratio
            scale = 2.0
            probs = torch.tensor([scipy.stats.gamma.pdf(val, shape, scale=scale) for val in x], dtype=torch.float32)
        
        else:
            raise ValueError("Unsupported exit distribution")
        
        if '7b' in model_name or '8b' in model_name:
            probs[0] = 0 # only enable exits from at least 4th layer for very deep model
        
        probs /= probs.sum()
        
        if get_global_rank() == 0: print('Expected early exit rate ', probs)

        for k in range(real_num_exit - 1): # not include the last exit
            count = 0
            out_n = math.floor(n_sample * probs[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = pred_value_gathered[k][ori_idx]
                        break
            if self.leq:
                filtered.add_(pred_value_gathered[k].le(T[k]).type_as(filtered))
                # filtered.add_(pred_value_gathered[k].less(T[k]).type_as(filtered))
            else:
                filtered.add_(pred_value_gathered[k].ge(T[k]).type_as(filtered))
                # filtered.add_(pred_value_gathered[k].greater(T[k]).type_as(filtered))

        if self.leq:
            T[real_num_exit - 1] = 1e8
        else:
            T[real_num_exit - 1] = -1e8

        self.thresholds = {self.exit_id_list[i] : T[i]  for i in range(real_num_exit)}
        if get_global_rank() == 0:
            print(f'Mean value for each layer:')
            for i in range(n_stage):
                print(f'{i+1} : {pred_value_gathered[i].mean():.5f}, {pred_value_gathered[i].std():.5f}, {pred_value_gathered[i].max():.5f}, {pred_value_gathered[i].min():.5f}')
            print(f'Find threshold on {n_sample} samples:')
            for i in range(real_num_exit):
                print(f'{i+1} : {T[i]:.5f}')
        return pred_value_gathered
    
    def get_threshold(self):
        assert self.thresholds is not None, "Please set thresholds before calling get_threshold"
        return self.thresholds

    def set_timestep(self, t):
        self.cur_step = t

    @torch.no_grad()  
    def forward(self, x, i, proprio, start_idx, end_idx, pos_offset):
        assert self.thresholds is not None, "Please set thresholds before calling forward"
        assert isinstance(i, int), 'index muast be integer'
        if i not in self.exit_id_list:
            return False, None
        
        # if still in a stage just use previous exit id
        if self.cur_step % self.steps_per_stage != 0:
            return i >= self.cur_exit_id, None
        
        if isinstance(self.value_net, ActionValueNet):
            value, action = self.value_net(x, i, proprio, start_idx, end_idx, pos_offset)
        else:
            raise NotImplementedError
        
        
        # print(f"*** self.thresholds: {self.thresholds}")
        thr = float(self.thresholds[i])
        value = float(value.mean().item())
        # print(f"*** value: {value}, thresholds[i]: {self.thresholds[i]}")
        if bool(value <= thr) is self.leq or i >= self.max_layer: # both be true or both be false
            self.cur_exit_id = i
            return True, action
        else:
            return False, None


@torch.no_grad()
def generate_action_values(
    args,
    model,
    value_net,
    dataloader,
    device_id,
):
    
    # 获取迭代总批次数：优先使用自定义属性，其次兼容标准 DataLoader 的 __len__
    # try:
    #     num_batches_per_epoch = calvin_loader.num_batches  # 自定义 DataLoader 可能提供该属性
    # except AttributeError:
    #     try:
    num_batches_per_epoch = len(dataloader)
        # except TypeError:
        #     num_batches_per_epoch = None  # 让 tqdm 自动推断

    # cast_dtype = get_cast_dtype(args.precision)

    model.eval()
    value_net.eval()

    # Enable TF32 for faster matmul/convolution on Ampere+ GPUs
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # loop through dataloader
    t = tqdm(
        enumerate(dataloader),
        disable=get_global_rank() != 0,
        total=num_batches_per_epoch,
        initial=0,
    )
    t.set_description(f"generate values by similarity")
    pred_value_list = []
    
    calib_max_batches = getattr(args, "calib_max_batches", None)
    for num_steps, batch in t:
        global_step = num_steps
        batch = move_to_device(batch, device_id)

        
        
        # Use inference_mode (faster than no_grad) and AMP if enabled
        with torch.inference_mode():
            # 统一使用 bfloat16 的 AMP（更稳），提升生成 KV 的数值稳定性
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=getattr(args, "amp", True)):
                # normalized_actions = model.predict_actions(**batch)
                outputs = model.forward(
                    input_ids=batch["input_ids"],
                    target_actions = None, #batch.get("action"),  ##
                    attention_mask=batch.get("attention_mask"),
                    attention_bias=batch.get("attention_bias"),
                    response_mask=(batch["loss_masks"] > 0) if "loss_masks" in batch else None,
                    images=batch.get("images"),
                    image_masks=batch.get("image_masks"),
                    image_input_idx=batch.get("image_input_idx"),
                    subsegment_ids=batch.get("subsegment_ids"),
                    position_ids=batch.get("position_ids"),
                    action_proprio=batch["proprio"],  ##
                    proprio_token_idx = batch["proprio_token_idx"],  ##
                    output_hidden_states=True if model.config.action_head != "flow_matching" else None,
                    use_cache=True if model.config.action_head == 'flow_matching' else False,
                    # Skip final LN+logits by exiting at last block to save time
                    exit_id=getattr(model.config, 'n_layers', None) - 1 if hasattr(model, 'config') else None,
                    # We don't need full logits for threshold calibration; compute minimal logits to save time
                    last_logits_only=True,
                )

            if model.config.action_head != 'flow_matching':
                start_idx, end_idx = model.get_action_idx(batch["input_ids"])
            else:
                start_idx, end_idx = 0, 0

            pos_offset = (batch["input_ids"] != -1).to(torch.int64).sum(dim=1)

        # # put images and labels on device
        # images = (batch_calvin[0].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))
        # gripper = (batch_calvin[3].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(2).unsqueeze(2))

        # # input_ids is LongTensor and does not require conversion precision
        # # repeat the input_ids to match the sequence length of the images
        # if args.fusion_mode != 'vit_concat':
        #     input_ids = batch_calvin[1][0].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        # else:
        #     input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)
        # # input_ids = batch_calvin[1][0].to(device_id, non_blocking=True)

        # # do the same to the attention mask 
        # if args.fusion_mode != 'vit_concat':
        #     attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True).unsqueeze(1).repeat(1, images.shape[1], 1)
        # else:
        #     attention_mask = batch_calvin[1][1].to(device_id, non_blocking=True)
        
        # state_tensor = batch_calvin[4].to(device_id, dtype=cast_dtype, non_blocking=True)
        # robot_obs = batch_calvin[5].to(device_id, dtype=cast_dtype, non_blocking=True)
        # if args.clip_state:
        #     state_tensor = torch.cat([state_tensor[..., :6], state_tensor[..., [-1]]], dim=-1)
        # labels = batch_calvin[2].to(device_id, dtype=cast_dtype, non_blocking=True)
        
        # state_tensor = state_tensor.unsqueeze(2).unsqueeze(2)

        # # merge the batch and the sequence dimension
        # images = images.flatten(0, 1)
        # gripper = gripper.flatten(0, 1)
        # state_tensor = state_tensor.flatten(0, 1)
        # if args.fusion_mode != 'vit_concat':
        #     input_ids = input_ids.flatten(0, 1)
        #     attention_mask = attention_mask.flatten(0, 1)

        # # [:6] is the joint position and [6:] is the gripper control, which is -1, 1, thus we need to convert it to 0, 1
        # if args.use_hist:
        #     labels = labels[:, [-1]]  # only calculate last step action
        # if args.fusion_mode == 'vit_concat':
        #     labels = labels[:, -1]
        # labels = [labels[..., :6], (labels[..., 6:] + 1) // 2]
        # print(f'{args.amp=}')
       
        # with torch.cuda.amp.autocast(enabled=args.amp), torch.no_grad():
            # if args.head_type == 'deterministic':
            #     final_output, exit_outputs, extra_exit_output, rand_layer_feat, _ = model(
            #         vision_x=images,
            #         lang_x=input_ids,
            #         attention_mask=attention_mask,
            #         # labels=labels,  # loss计算放在外面
            #         vision_gripper=gripper,
            #         state_tensor=state_tensor if (args.use_state or args.sep_lm_head) else None,
            #         with_gripper_logits=True,
            #         return_in_feat=True,
            #         only_extra_exit=True,
            #     )

        if model.config.action_head != 'flow_matching':
            feats = outputs.hidden_states # n_exit x (bs * action_seq_len, lang_len, d)
        else:
            feats = outputs.attn_key_values 

        # rand_layer_feat = rand_layer_feat # (bs * action_seq_len, lang_len, d)
        sim = value_net(feats, mode='generate', proprio=batch["proprio"], start_idx=start_idx, end_idx=end_idx, pos_offset=pos_offset)
        # print(f"*** similarity in generate_action_values: {sim}")
        if torch.isnan(sim).any() or torch.isinf(sim).any():
            raise ValueError("sim contains NaN or Inf")
        
        # record
        pred_value_list.append(sim)

        if (num_steps + 1) >= num_batches_per_epoch:
            break

        # optional early stop to accelerate calibration
        if calib_max_batches is not None and (num_steps + 1) >= calib_max_batches:
            break

    pred_value_list = torch.cat(pred_value_list, dim=1)
    # pred_value_list = pred_value_list.flatten(1, 2)
        
    # return pred_value_list, target_value_list
    return pred_value_list, None
