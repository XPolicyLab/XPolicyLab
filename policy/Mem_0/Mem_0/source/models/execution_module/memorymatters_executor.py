import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from termcolor import cprint


from source.models.execution_module.qwen3_vl_model.qwen3_vl_encapsulation import Qwen3VL_Encapsulation
from source.models.execution_module.memory_bank.memory_bank import MemoryBank
from source.models.execution_module.action_model.ActionHeader import FlowmatchingActionHead
from source.models.execution_module.classifier.subtask_classifier import SubtaskEndClassifier

def resize_images(images, target_size=(224, 224)):
    """
    recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):  # if it is a single PIL image
        return images.resize(target_size)
    elif isinstance(images, list):  # if it is a list, recursively process each element
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")

class MemoryMattersExecutor(nn.Module):
    def __init__(self, config: Optional[dict] = None, device: Optional[torch.device] = None, **kwargs):
        """
        Initialize the MemoryMatters executor.
        """
        super().__init__()
        # --------------------------------
        # ---       model config       ---
        # --------------------------------
        # get qwen_vl config
        qwenvl_config = config.execution_module.get("qwen_vl", {})
        model_path = qwenvl_config.get("model_path", "./checkpoints/Qwen3-VL-2B-Instruct")
        # get memory bank config
        memory_bank_config = config.execution_module.get("memory_bank", {})
        window_size = memory_bank_config.get("window_size", 30)
        initial_anchor_size = memory_bank_config.get("initial_anchor_size", 1)
        num_heads = memory_bank_config.get("num_heads", 8)
        memory_bank_dropout = memory_bank_config.get("dropout", 0.1)
        memory_accumulation = memory_bank_config.get("memory_accumulation", 8)
        # get action model config
        action_model_config = config.execution_module.get("action_model", {})
        self.action_horizon = action_model_config.get("action_horizon", 16)
        self.repeated_diffusion_steps = action_model_config.get("repeated_diffusion_steps", 12)
        # get classifier config
        classifier_config = config.execution_module.get("classifier", {})
        classifier_hidden_sizes = classifier_config.get("hidden_sizes", [2048])
        classifier_dropout = classifier_config.get("dropout", 0.1)
        classifier_pos_weight = classifier_config.get("pos_weight", None)
        classifier_focal_gamma = classifier_config.get("focal_gamma", 0.0)
        self.classifier_threshold = float(classifier_config.get("threshold", 0.5))
        # get loss weights config
        loss_weights = config.execution_module.get("loss_weights", {})
        self.lambda_action = loss_weights.get("lambda_action", 1.0)
        self.lambda_classifier = loss_weights.get("lambda_classifier", 0.0)

        # --------------------------------
        # ---    model definition      ---
        # --------------------------------
        # set qwen vl model
        self.qwen_model = Qwen3VL_Encapsulation(model_path, device=device)
        # set memory bank
        self.memory_bank = MemoryBank(
            hidden_dim=self.qwen_model.hidden_size,
            window_size=window_size,
            initial_anchor_size=initial_anchor_size,
            num_heads=num_heads,
            dropout=memory_bank_dropout,
            memory_accumulation=memory_accumulation,
            dtype=torch.bfloat16,
            device=device,
        )
        # set action model
        self.action_model = FlowmatchingActionHead(action_model_config, hidden_size=self.qwen_model.hidden_size)
        # set classifier
        self.classifier = SubtaskEndClassifier(
            hidden_sizes=classifier_hidden_sizes,
            dropout=classifier_dropout,
            pos_weight=classifier_pos_weight,
            focal_gamma=classifier_focal_gamma,
        )
        
    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ):
        """
        Forward pass for the MemoryMatters executor.
        """
        # ------------ Prepare Inputs ------------
        batch_images = [example["image"] for example in examples]  # [B, [PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 16]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Prepare classifier labels
        cls_labels = [ex.get("subtask_end", 0) for ex in examples]  # [B]
        
        # Extract episode_ids if available, otherwise use batch indices
        episode_ids = [example["episode_id"] for example in examples]  # [B]
        subtask_end_flags = [example.get("subtask_end", 0) for example in examples]
        
        # ------------ Forward Procedure ------------

        # Step 1: Get Latent Features from Qwen-VL
        qwen_inputs = self.qwen_model.build_qwenvl_inputs(batch_images, instructions, system_prompt=None, add_summary_token=False, add_generation_prompt=False, max_length=128) # prepare inputs
        qwenvl_forward_outputs = self.qwen_model( # get outputs with hidden states
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = qwenvl_forward_outputs.hidden_states[-1]    # (batch_size, sequence_length, hidden_size). final layer hidden state.
        image_feature, text_feature = self.qwen_model.extract_features(qwen_inputs.input_ids, last_hidden_state) # (batch_size, num_image_tokens, hidden_size), (batch_size, num_text_tokens, hidden_size)
        
        # Step 2: Memory Fusion and Update
        # Memory bank processes image features and returns fused memory feature and anchor feature
        fused_instant_memory_feature, fused_anchor_memory_feature = self.memory_bank(
            image_feature,
            episode_ids=episode_ids,
            subtask_end_flags=subtask_end_flags,
        )  # fused_instant_memory_feature: (B, 1, H), fused_anchor_memory_feature: (B, 1, H)
        
        # Concatenate three features: [fused_memory, anchor, text] for action prediction
        summary_feature = torch.cat([fused_instant_memory_feature, fused_anchor_memory_feature, text_feature], dim=1)  # (B, 3, H)

        # Step 3: Action Prediction
        # Action model uses the concatenated 3-feature summary (B, 3, H)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.stack(
                [
                    torch.as_tensor(a, device=fused_instant_memory_feature.device, dtype=fused_instant_memory_feature.dtype)
                    for a in actions
                ]
            )  # (B, T_full, action_dim)
            actions_target = actions[:, -(self.action_horizon):, :]  # (B, chunk_len, action_dim), expert actions labels

            # Repeat for multiple diffusion steps
            actions_target_repeated = actions_target.repeat(self.repeated_diffusion_steps, 1, 1)  # (B * repeated_diffusion_steps, chunk_len, action_dim)
            summary_feature_repeated = summary_feature.repeat(self.repeated_diffusion_steps, 1, 1)  # (B * repeated_diffusion_steps, 3, H)
            
            state_repeated = None
            if state is not None:
                state = torch.stack(
                    [
                        torch.as_tensor(s, device=summary_feature.device, dtype=summary_feature.dtype)
                        for s in state
                    ]
                )
                state_repeated = state.repeat(self.repeated_diffusion_steps, 1, 1)  # (B * repeated_diffusion_steps, 1, state_dim)
            
            action_loss = self.action_model(
                vl_embs=summary_feature_repeated,
                actions=actions_target_repeated,
                state=state_repeated,
            )
        
        # Step 4: Subtask Classification
        # Classifier uses mean pooling of the 3-feature summary to get (B, H) input
        # This preserves information while matching classifier's expected input shape
        summary_feature_for_classifier = summary_feature.reshape(summary_feature.shape[0], -1) # (B, 3 * hidden_size)
        with torch.autocast("cuda", dtype=torch.float32):
            cls_labels = torch.tensor(cls_labels, device=summary_feature_for_classifier.device, dtype=summary_feature_for_classifier.dtype)  # (B,)
            
            classifier_output = self.classifier(
                fused_hidden=summary_feature_for_classifier,  # (B, 3 * hidden_size)
                labels=cls_labels,
            )
            classifier_loss = classifier_output["loss"]

        # Classifier accuracy/precision/recall for logging/debug.
        with torch.no_grad():
            logits = classifier_output["logits"]
            prob = torch.sigmoid(logits)
            preds = (prob >= self.classifier_threshold).to(cls_labels.dtype)
            tp = ((preds == 1) & (cls_labels == 1)).to(summary_feature_for_classifier.dtype).sum()
            fp = ((preds == 1) & (cls_labels == 0)).to(summary_feature_for_classifier.dtype).sum()
            fn = ((preds == 0) & (cls_labels == 1)).to(summary_feature_for_classifier.dtype).sum()
            correct_sum = (preds == cls_labels).to(summary_feature_for_classifier.dtype).sum()
            denom = torch.tensor(float(cls_labels.numel()), device=cls_labels.device).clamp_min(1e-6)
            positive_count = cls_labels.sum()

            classifier_accuracy = correct_sum / denom
            precision = tp / (tp + fp).clamp_min(1e-6)
            recall = tp / (tp + fn).clamp_min(1e-6)
            f1_score = (2 * tp) / (2 * tp + fp + fn).clamp_min(1e-6)
            prate = positive_count / denom

            tp_sum = float(tp.item())
            fp_sum = float(fp.item())
            fn_sum = float(fn.item())
            correct_val = float(correct_sum.item())
            total_val = float(denom.item())
            positive_val = float(positive_count.item())

        total_loss = self.lambda_action * action_loss + self.lambda_classifier * classifier_loss
        
        return {
            "loss": total_loss,
            "action_loss": action_loss,
            "classifier_loss": classifier_loss,
            "classifier_accuracy": classifier_accuracy,
            "classifier_precision": precision,
            "classifier_recall": recall,
            "classifier_f1_score": f1_score,
            "classifier_prate": prate,
            "cls_tp": tp_sum,
            "cls_fp": fp_sum,
            "cls_fn": fn_sum,
            "cls_correct": correct_val,
            "cls_total": total_val,
            "cls_positive": positive_val,
        }
        
    @torch.inference_mode()
    def predict(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        episode_id: Optional[int] = None,
        **kwargs: str,
    ) -> dict:
        """
        Inference: regress future actions (no diffusion sampling) and predict subtask-end probability.

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          3. Memory fusion to incorporate history
          4. Predict actions and subtask-end logits

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            state: Optional state tensor aligned with samples.
            episode_id: Optional episode ids for memory bank.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim].
                classifier_logits (Tensor or None): Shape (B,), raw logits.
                classifier_prob (np.ndarray or None): Shape (B,), sigmoid probabilities.
        """
        batch_images = resize_images(batch_images, target_size=(224, 224))
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_model.build_qwenvl_inputs(batch_images, instructions, system_prompt=None, add_summary_token=False, add_generation_prompt=False, max_length=128) # prepare inputs
        qwenvl_outputs = self.qwen_model(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = qwenvl_outputs.hidden_states[-1]    # (batch_size, sequence_length, hidden_size). final layer hidden state.
        image_feature, text_feature = self.qwen_model.extract_features(qwen_inputs.input_ids, last_hidden_state) # (batch_size, 1, hidden_size), (batch_size, 1, hidden_size)
        
        # Extract episode_ids if available, otherwise use batch indices
        # For inference, we assume each sample is from the same episode (episode_id=0) by default
        episode_id = episode_id if episode_id is not None else 0
        
        # Step 2: Memory Fusion and Update
        memory_fusion_output, anchor_output, sub_end_flag = self.memory_bank.update_on_eval(image_feature, text_feature, self.classifier, episode_id=episode_id) # anchor_output: (batch_size, 1, hidden_size), memory_fusion_output: (batch_size, 1, hidden_size), sub_end_flag: bool

        summary_feature = torch.cat([memory_fusion_output, anchor_output, text_feature], dim=1) # (batch_size, 3, hidden_size)
        
        # Step 3: Action Prediction
        with torch.autocast("cuda", dtype=torch.float32):
            state = torch.from_numpy(np.array(state)).to(memory_fusion_output.device, dtype=memory_fusion_output.dtype) if state is not None else None
            pred_actions = self.action_model.predict_action(summary_feature, state)  # (B, chunk_len, action_dim)

        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
            "subtask_end_flag": sub_end_flag,  # List of bools, one per sample
        }

        
        
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from omegaconf import OmegaConf
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./source/config/execution_module_train.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()
    
    cfg = OmegaConf.load(args.config_yaml)
    executor = MemoryMattersExecutor(cfg, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # output_path = os.path.join(os.path.dirname(__file__), "memorymatters_executor_model_structure.txt")
    # with open(output_path, "w") as f:
    #     f.write(str(executor))
    
    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    batch  = [
        {
            "action": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float16), # action_chunk, action_dim
            "image": [image], # two views
            "lang": "This is a fake for testing. whatever it is",
            "state" : np.random.uniform(-1, 1, size=(1, 16)).astype(np.float16), # chunk, state_dim
            "subtask_end": 0, # binary label
            "episode_id": 0, # episode id
        },
        {
            "action": np.random.uniform(-1, 1, size=(16, 16)).astype(np.float16), # action_chunk, action_dim
            "image": [image], # two views
            "lang": "This is a fake for testing",
            "state" : np.random.uniform(-1, 1, size=(1, 16)).astype(np.float16), # chunk, state_dim
            "subtask_end": 0, # binary label
            "episode_id": 0, # episode id
        }
    ]  # batch size 2
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = executor.to(device)
    outputs = model(batch)
    print(f"Total Loss: {outputs['loss'].item()}")
    print(f"Action Loss: {outputs['action_loss'].item()}")
    print(f"Classifier Loss: {outputs['classifier_loss'].item()}")
    
    # test predict
    for example in batch:
        model.eval()
        predict_output = model.predict(batch_images=[example["image"]], instructions=[example["lang"]], state=[example["state"]], episode_ids=example["episode_id"])
        print(f"normalized action: {predict_output['normalized_actions'].shape}")
        print(f"subtask_end_flag: {predict_output['subtask_end_flag']}")
