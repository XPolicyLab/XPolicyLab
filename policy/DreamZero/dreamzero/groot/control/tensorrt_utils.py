import torch
import os
import subprocess
import tensorrt as trt
import sys
import atexit
import ctypes
import modelopt.torch.quantization as mtq
from typing import Dict, List, Tuple
import shutil

import numpy as np
import torch


FP8_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
        "*output_quantizer": {"enable": False},
        "*[qkv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

NVFP4_DEFAULT_CONFIG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*input_quantizer": {
            "num_bits": (2, 1),
            "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
            "axis": None,
            "enable": True,
        },
        "*output_quantizer": {"enable": False},
        "*[qkv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
        "*softmax_quantizer": {
            "num_bits": (4, 3),
            "axis": None,
        },
        "default": {"enable": False},
    },
    "algorithm": "max",
}

    

def wan_quantize(
    policy,
    quantization_config,
    model_type,
    forward_loop,
):
    """Quantize the VLA model using ModelOpt - simplified to use calc_mse_for_single_trajectory."""

    # Configure quantization - disable problematic layers
    if "quant_cfg" in quantization_config:
        quantization_config["quant_cfg"]["*patch_embedding*"] = {"enable": False}
    #    if model_type == "14B" or model_type == "ar_14B":
    #        # Workaround: until we understand the issue https://nvbugspro.nvidia.com/bug/5612316
    #        quantization_config["quant_cfg"]["*.self_attn.o.*"] = {"enable": False}
    #        quantization_config["quant_cfg"]["*.cross_attn.o.*"] = {"enable": False}

    policy.trained_model.action_head.model = mtq.quantize(
        policy.trained_model.action_head.model, quantization_config, forward_loop=forward_loop
    )
    mtq.print_quant_summary(policy.trained_model.action_head.model)

    return


def wan_trt_quantize_and_load_engine(
    policy,
    cfg,
    onnx_path,
    engine_path,
    model_type,
    forward_loop,
):
    if (
        os.path.exists(os.path.dirname(engine_path))
        and cfg.inference_mode == "trt_build"
    ):
        shutil.rmtree(os.path.dirname(engine_path))

    quantization_config = None
    if cfg.quantize_dtype == "fp8":
        quantization_config = FP8_DEFAULT_CONFIG.copy()
    elif cfg.quantize_dtype == "nvfp4":
        quantization_config = NVFP4_DEFAULT_CONFIG.copy()
    else:
        print(f"Quantization type {cfg.quantize_dtype} not supported. Skipping quantization.")

    if quantization_config is not None and cfg.inference_mode == "trt_build":
        #policy.trained_model.action_head.model.to(torch.float16)
        wan_quantize(
            policy,
            quantization_config,
            model_type=model_type,
            forward_loop=forward_loop,
        )

    if  cfg.inference_mode == "trt_build":
        policy.trained_model.action_head.model.to(torch.float16)

        print("Export model:", policy.trained_model.action_head.model)

        test_inputs = create_wan_test_inputs(policy, device="cuda", model_type=model_type)
        min_shape = None
        max_shape = None
        opt_shape = None

        if model_type == "ar_14B":

            policy.trained_model.action_head.model.forward = policy.trained_model.action_head.model._forward_inference_trt
            dynamic_axes = {
                "kv_cache_packed": {3: "kv_cache_len"},
            }
            min_shape = "kv_cache_packed:40x2x1x880x40x128"
            max_shape = "kv_cache_packed:40x2x1x8800x40x128"
            opt_shape = "kv_cache_packed:40x2x1x7920x40x128"
        elif model_type == "ar_14B_droid":
            policy.trained_model.action_head.model.forward = policy.trained_model.action_head.model._forward_inference_trt
            dynamic_axes = {
                "kv_cache_packed": {3: "kv_cache_len"},
            }
            min_shape = "kv_cache_packed:40x2x1x880x40x128"
            max_shape = "kv_cache_packed:40x2x1x8800x40x128"
            opt_shape = "kv_cache_packed:40x2x1x7920x40x128"
        elif model_type == "ar_5B_n6":
            policy.trained_model.action_head.model.forward = policy.trained_model.action_head.model._forward_inference_trt
            dynamic_axes = {
                "kv_cache_packed": {3: "kv_cache_len"},
            }
            min_shape = "kv_cache_packed:30x2x1x220x24x128"
            max_shape = "kv_cache_packed:30x2x1x3080x24x128"
            opt_shape = "kv_cache_packed:30x2x1x2860x24x128"
        else:
            dynamic_axes = None

        if cfg.quantize_dtype == "nvfp4":
            export_to_onnx_fp4(policy.trained_model.action_head.model, test_inputs, onnx_path, dynamic_axes=dynamic_axes)
        else:
            export_to_onnx(
                policy.trained_model.action_head.model,
                test_inputs,
                onnx_path,
                model_type=model_type,
                quantization_mode=cfg.quantize_dtype,
                dynamic_axes=dynamic_axes,
            )

        build_tensorrt_engine(onnx_path, engine_path, min_shape, max_shape, opt_shape)

    trt_wan_model = load_tensorrt_engine(engine_path, model_type=model_type)
    policy.trained_model.action_head.model = trt_wan_model

def export_to_onnx_fp4(model, test_inputs, onnx_save_path, dynamic_axes=None):
    from modelopt.torch._deploy.utils.torch_onnx import OnnxBytes
    from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata

    print("exporting to onnx fp4")
    try:
        onnx_bytes, _ = get_onnx_bytes_and_metadata(model=model, dummy_input=test_inputs, dynamic_axes=dynamic_axes)
        onnx_model = OnnxBytes.from_bytes(onnx_bytes)
    except Exception as e:
        print(f"Error exporting model to ONNX: {e}")
        return
    save_dir = os.path.dirname(os.path.abspath(onnx_save_path))
    os.makedirs(save_dir, exist_ok=True)
    for filename, file_bytes in onnx_model.onnx_model.items():
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        print(f"exported onnx to {file_path}")


def export_to_onnx(
    pytorch_model,
    test_inputs,
    onnx_path="tensorrt/wan_model.onnx",
    model_type="5B",
    quantization_mode="fp8",
    dynamic_axes=None,
):
    #
    if model_type == "5B":
        return export_to_onnx_5B(pytorch_model, test_inputs, onnx_path, dynamic_axes)
    elif model_type == "14B":
        return export_to_onnx_14B(pytorch_model, test_inputs, onnx_path, dynamic_axes)
    elif model_type == "ar_14B" or model_type == "ar_14B_droid":
        return export_to_onnx_ar_14B(pytorch_model, test_inputs, onnx_path, dynamic_axes)
    else:
        raise ValueError(f"Model type {model_type} not supported")


def export_to_onnx_ar_14B(pytorch_model, test_inputs, onnx_path="tensorrt/wan_model.onnx", dynamic_axes=None):
    """Export PyTorch model to ONNX"""
    print("Exporting AR 14B model to ONNX...", onnx_path)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    pytorch_model.eval()
    pytorch_model.to(torch.float16)

    input_names = [
        "x",
        "timestep",
        "context",
        "kv_cache_packed",
        "y",
        "clip_feature",
        "action",
        "timestep_action",
        "state",
    ]
    output_names = ["video_noise_pred", "action_noise_pred"]

    try:
        with torch.no_grad():
            torch.onnx.export(
                pytorch_model,
                test_inputs,
                onnx_path,
                export_params=True,
                opset_version=20,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        print(f"  ONNX model exported to: {onnx_path}")
        return onnx_path

    except Exception as e:
        import traceback
        print(f"  ERROR: ONNX export failed. Exception type: {type(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None



def export_to_onnx_5B(pytorch_model, test_inputs, onnx_path="tensorrt/wan_model.onnx"):
    """Export PyTorch model to ONNX"""
    print("Exporting model to ONNX...")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    pytorch_model.eval()
    pytorch_model.to(torch.float16)

    x, action, timestep, context, state, embodiment_id = test_inputs

    # Define input names for better ONNX graph
    input_names = ["x", "action", "timestep", "context", "state", "embodiment_id"]
    output_names = ["video_noise_pred", "action_noise_pred"]

    try:
        with torch.no_grad():
            torch.onnx.export(
                pytorch_model,
                (x, action, timestep, context, state, embodiment_id),
                onnx_path,
                export_params=True,
                opset_version=20,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
            )
        print(f"  ONNX model exported to: {onnx_path}")
        return onnx_path

    except Exception as e:
        import traceback
        print(f"  ERROR: ONNX export failed. Exception type: {type(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None


def export_to_onnx_14B(pytorch_model, test_inputs, onnx_path="tensorrt/wan_model.onnx"):
    """Export PyTorch model to ONNX"""
    print("Exporting model to ONNX...")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    pytorch_model.eval()
    pytorch_model.to(torch.float16)

    x, action, timestep, context, state, embodiment_id, clip_feature, y = test_inputs

    # Define input names for better ONNX graph
    input_names = [
        "x",
        "action",
        "timestep",
        "context",
        "state",
        "embodiment_id",
        "clip_feature",
        "y",
    ]
    output_names = ["video_noise_pred", "action_noise_pred"]

    try:
        with torch.no_grad():
            torch.onnx.export(
                pytorch_model,
                (x, action, timestep, context, state, embodiment_id, clip_feature, y),
                onnx_path,
                export_params=True,
                opset_version=20,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
            )
        print(f"  ONNX model exported to: {onnx_path}")
        return onnx_path

    except Exception as e:
        import traceback
        print(f"  ERROR: ONNX export failed. Exception type: {type(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None


def build_tensorrt_engine(onnx_path, engine_path="tensorrt/wan_model.trt", min_shape=None, max_shape=None, opt_shape=None):
    """Build TensorRT engine from ONNX using trtexec"""
    print("Building TensorRT engine with trtexec...")

    if not os.path.exists(onnx_path):
        print(f"  ERROR: ONNX file not found: {onnx_path}")
        return None

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    # Build engine using trtexec (much faster than torch_tensorrt)
    trtexec_bin = shutil.which("trtexec") or "/opt/tensorrt/bin/trtexec"
    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp8",
        "--fp16",
        "--bf16",
        "--separateProfileRun",
        "--profilingVerbosity=detailed",
        "--memPoolSize=workspace:65536",
        "--dumpProfile",
        "--dumpLayerInfo",
        "--useCudaGraph",
        "--verbose",
    ]

    if min_shape is not None:
        cmd.append(f"--minShapes={min_shape}")
    if max_shape is not None:
        cmd.append(f"--maxShapes={max_shape}")
    if opt_shape is not None:
        cmd.append(f"--optShapes={opt_shape}")

    # Create log file for trtexec output
    log_file = engine_path.replace(".trt", "_build.log")

    try:
        print(f"  Running: {' '.join(cmd)}")
        print(f"  Logging output to: {log_file}")

        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=600)

        if result.returncode == 0:
            print(f"  TensorRT engine built successfully: {engine_path}")
            print(f"  Build log saved to: {log_file}")
            return engine_path
        else:
            print(f"  ERROR: trtexec failed with return code {result.returncode}")
            print(f"  Check build log for details: {log_file}")
            # Print last few lines of log file for immediate feedback
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        print("  Last few lines from build log:")
                        for line in lines[-10:]:  # Show last 10 lines
                            print(f"    {line.rstrip()}")
            except:
                pass
            return None

    except subprocess.TimeoutExpired:
        print("  ERROR: trtexec timed out after 5 minutes")
        print(f"  Partial build log saved to: {log_file}")
        return None
    except Exception as e:
        print(f"  ERROR: Failed to run trtexec: {e}")
        return None


def torch_type(trt_type):
    mapping = {
        trt.float32: torch.float32,  # Added missing FLOAT mapping
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
        trt.int64: torch.int64,
    }
    if trt_type in mapping:
        return mapping[trt_type]

    raise TypeError(
        f"Could not resolve TensorRT datatype to an equivalent torch datatype. {trt_type}"
    )


class Engine(object):
    def __init__(self, file, plugins=[]):
        super().__init__()

        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")

        self.plugins = [ctypes.CDLL(plugin, ctypes.RTLD_GLOBAL) for plugin in plugins]
        self.file = file
        self.load(file)

        def destroy(self):
            del self.execution_context
            del self.handle

        atexit.register(destroy, self)
        self.print()

    def print(self):

        print("============= TRT Engine Detail =============")
        print(f"Engine file: {self.file}")
        print(f"Inputs: {len(self.in_meta)}")
        for ib, item in enumerate(self.in_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")

        print(f"Outputs: {len(self.out_meta)}")
        for ib, item in enumerate(self.out_meta):
            tensor_name, shape, dtype = item[:3]
            print(f"   {ib}. {tensor_name}: {'x'.join(map(str, shape))} [{dtype}]")
        print("=============================================")

    def load(self, file):
        runtime = trt.Runtime(self.logger)

        with open(file, "rb") as f:
            self.handle = runtime.deserialize_cuda_engine(f.read())
            assert (
                self.handle is not None
            ), f"Failed to deserialize the cuda engine from file: {file}"

        self.execution_context = self.handle.create_execution_context()
        self.meta, self.in_meta, self.out_meta = [], [], []
        for tensor_name in self.handle:
            shape = self.handle.get_tensor_shape(tensor_name)
            print(f"Tensor name: {tensor_name}, shape: {shape}")
            dtype = torch_type(self.handle.get_tensor_dtype(tensor_name))
            if self.handle.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.in_meta.append([tensor_name, shape, dtype])
            else:
                self.out_meta.append([tensor_name, shape, dtype])

    def __call__(self, *args, **inputs):
        return self.forward(*args, **inputs)

    def set_runtime_tensor_shape(self, name, shape):
        self.execution_context.set_input_shape(name, shape)

    def forward(self, *args, **kwargs):
        return_list = kwargs.pop("return_list", False)
        reference_tensors = []
        stream = torch.cuda.current_stream()
        for iarg, x in enumerate(args):
            name, shape, dtype = self.in_meta[iarg]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            assert isinstance(x, torch.Tensor), f"Unsupported tensor type: {type(x)}"
            assert runtime_shape == x.shape, f"Invalid input shape: {runtime_shape} != {x.shape}"
            assert (
                dtype == x.dtype
            ), f"Invalid tensor dtype, excepted dtype is {dtype}, but got {x.dtype}"
            assert x.is_cuda, f"Invalid tensor device, excepted device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for name, shape, dtype in self.in_meta:
            if name not in kwargs:
                continue

            runtime_shape = self.execution_context.get_tensor_shape(name)
            x = kwargs[name]
            assert isinstance(x, torch.Tensor), f"Unsupported tensor[{name}] type: {type(x)}"
            assert (
                runtime_shape == x.shape
            ), f"Invalid input[{name}] shape: {x.shape}, but the expected shape is: {runtime_shape}"
            assert (
                dtype == x.dtype
            ), f"Invalid tensor[{name}] dtype, expected dtype is {dtype}, but got {x.dtype}"
            assert (
                x.is_cuda
            ), f"Invalid tensor[{name}] device, expected device is cuda, but got {x.device}"
            x = x.cuda().contiguous()
            self.execution_context.set_tensor_address(name, x.data_ptr())
            reference_tensors.append(x)

        for item in self.out_meta:
            name = item[0]
            runtime_shape = self.execution_context.get_tensor_shape(name)
            output_tensor = torch.zeros(
                *runtime_shape, dtype=item[2], device=reference_tensors[0].device
            )
            self.execution_context.set_tensor_address(name, output_tensor.data_ptr())
            reference_tensors.append(output_tensor)

        self.execution_context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        assert len(reference_tensors) == len(self.in_meta) + len(
            self.out_meta
        ), f"Invalid input tensors. The expected I/O tensors are {len(self.in_meta) + len(self.out_meta)}, but got {len(reference_tensors)}"

        if return_list:
            return [
                reference_tensors[len(self.in_meta) + i] for i, item in enumerate(self.out_meta)
            ]
        else:
            return {
                item[0]: reference_tensors[len(self.in_meta) + i]
                for i, item in enumerate(self.out_meta)
            }


class WanTrtModel5B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
    ):

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)

        output = self.engine(
            x=x.to(torch.float16),
            action=action.to(torch.float16),
            timestep=timestep.to(torch.float16),
            context=context.to(torch.float16),
            state=state.to(torch.float16),
            embodiment_id=embodiment_id.to(torch.int32),
        )
        if "out.0" in output: # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(torch.bfloat16).contiguous()


class WanTrtModel14B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        clip_feature: torch.Tensor,
        y: torch.Tensor,
    ):

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)
        self.engine.set_runtime_tensor_shape("clip_feature", clip_feature.shape)
        self.engine.set_runtime_tensor_shape("y", y.shape)

        output = self.engine(
            x=x.to(torch.float16),
            action=action.to(torch.float16),
            timestep=timestep.to(torch.float16),
            context=context.to(torch.float16),
            state=state.to(torch.float16),
            embodiment_id=embodiment_id.to(torch.int32),
            clip_feature=clip_feature.to(torch.float16),
            y=y.to(torch.float16),
        )
        if "out.0" in output: # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(torch.bfloat16).contiguous()


class WanTrtModelAr5B(torch.nn.Module):
    """TRT wrapper for ar_5B_n6 model type - uses kv_cache but no clip_feature."""
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x,
        timestep,
        context,
        kv_cache: list[torch.Tensor],
        y=None,
        action=None,
        timestep_action=None,
        state=None,
    ):

        kv_cache_packed = torch.stack(kv_cache, dim=0)

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("timestep", timestep.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("kv_cache_packed", kv_cache_packed.shape)
        # self.engine.set_runtime_tensor_shape("y", y.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("timestep_action", timestep_action.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)


        output = self.engine(
            x.to(torch.float16),
            timestep.to(torch.float16),
            context.to(torch.float16),
            kv_cache_packed.to(torch.float16),
            # y.to(torch.float16),
            action.to(torch.float16),
            timestep_action.to(torch.float16),
            state.to(torch.float16),
        )

        if "out.0" in output: # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(torch.bfloat16).contiguous()


class WanTrtModelAr14B(torch.nn.Module):
    def __init__(self, eng_path: str):
        super().__init__()
        self.engine = Engine(eng_path)

    def forward(
        self,
        x,
        timestep,
        context,
        kv_cache: list[torch.Tensor],
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
    ):

        kv_cache_packed = torch.stack(kv_cache, dim=0)

        self.engine.set_runtime_tensor_shape("x", x.shape)
        self.engine.set_runtime_tensor_shape("timestep", timestep.shape)
        self.engine.set_runtime_tensor_shape("context", context.shape)
        self.engine.set_runtime_tensor_shape("kv_cache_packed", kv_cache_packed.shape)
        self.engine.set_runtime_tensor_shape("y", y.shape)
        self.engine.set_runtime_tensor_shape("clip_feature", clip_feature.shape)
        self.engine.set_runtime_tensor_shape("action", action.shape)
        self.engine.set_runtime_tensor_shape("timestep_action", timestep_action.shape)
        self.engine.set_runtime_tensor_shape("state", state.shape)


        output = self.engine(
            x.to(torch.float16),
            timestep.to(torch.float16),
            context.to(torch.float16),
            kv_cache_packed.to(torch.float16),
            y.to(torch.float16),
            clip_feature.to(torch.float16),
            action.to(torch.float16),
            timestep_action.to(torch.float16),
            state.to(torch.float16),
        )

        if "out.0" in output: # for nvfp4 model export through modelopt
            return output["out.0"].to(torch.bfloat16).contiguous(), output["out.1"].to(torch.bfloat16).contiguous()
        else:
            return output["video_noise_pred"].to(torch.bfloat16).contiguous(), output["action_noise_pred"].to(torch.bfloat16).contiguous()

def load_tensorrt_engine(engine_path="tensorrt/wan_model.trt", model_type="5B"):
    """Load TensorRT engine"""
    if model_type == "5B":
        trt_inference = WanTrtModel5B(engine_path)
    elif model_type == "ar_5B_n6" or model_type == "ar_5B":
        trt_inference = WanTrtModelAr5B(engine_path)
    elif model_type == "14B":
        trt_inference = WanTrtModel14B(engine_path)
    elif model_type == "ar_14B" or model_type == "ar_14B_droid":
        trt_inference = WanTrtModelAr14B(engine_path)
    else:
        raise ValueError(f"Model type {model_type} not supported")
    return trt_inference


def create_wan_test_inputs(policy, device="cuda", model_type="5B"):
    # Get dtype from model parameters
    dtype = torch.float16

    # Use hardcoded dimensions from the original working version of the script
    if model_type == "5B":
        x = torch.randn(1, 48, 13, 22, 40, dtype=dtype, device=device)
        action = torch.randn(1, 48, 32, dtype=dtype, device=device)
        timestep = torch.randn(1, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
        embodiment_id = torch.zeros(1, dtype=torch.int32, device=device)
        timestep_action = torch.randn(1, 48, dtype=dtype, device=device)
        seq_len = torch.tensor(440, dtype=torch.int32, device=device)
        return x, action, timestep, context, state, embodiment_id, timestep_action, seq_len
    elif model_type == "ar_5B_n6":
        # ar_5B_n6 uses _forward_inference_trt which requires kv_cache_packed
        # Shape from dynamic_axes: kv_cache_packed:30x2x1x220x24x128
        # Note: 5B model doesn't use clip_feature (unlike 14B), but still needs y
        x = torch.randn(1, 48, 2, 22, 40, dtype=dtype, device=device)
        timestep = torch.randn(1, 2, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        # y = torch.randn(1, 52, 2, 22, 40, dtype=dtype, device=device)  # y is required by _forward_inference_trt
        action = torch.randn(1, 48, 32, dtype=dtype, device=device)
        timestep_action = torch.randn(1, 48, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
    
        num_heads = 24
        head_dim = 128
        num_layers = 30
        B = 1
    
        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                torch.zeros([2, B, 13*220, num_heads, head_dim], dtype=dtype, device=device)
            )
    
        kv_cache_packed = torch.stack(kv_cache, dim=0)
        # Return order matches _forward_inference_trt signature: x, timestep, context, kv_cache_packed, y, action, timestep_action, state
        return (x, timestep, context, kv_cache_packed, action, timestep_action, state)
    elif model_type == "14B":
        x = torch.randn(1, 16, 13, 44, 80, dtype=dtype, device=device)
        action = torch.randn(1, 48, 32, dtype=dtype, device=device)
        timestep = torch.randn(1, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
        embodiment_id = torch.zeros(1, dtype=torch.int32, device=device)
        clip_feature = torch.randn(1, 257, 1280, dtype=dtype, device=device)
        y = torch.randn(1, 20, 13, 44, 80, dtype=dtype, device=device)
        return x, action, timestep, context, state, embodiment_id, clip_feature, y
    elif model_type == "ar_14B":
        clip_feature = torch.randn(1, 257, 1280, dtype=dtype, device=device)
        y = torch.randn(1, 20, 2, 44, 80, dtype=dtype, device=device)
        timestep_action = torch.randn(1, 48, dtype=dtype, device=device)
        x = torch.randn(1, 16, 2, 44, 80, dtype=dtype, device=device)
        timestep = torch.randn(1, 2, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        seq_len = torch.tensor(1760, dtype=torch.int32, device=device)
        action = torch.randn(1, 48, 32, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
        embodiment_id = torch.zeros(1, dtype=torch.int32, device=device)
    
        num_heads = 40 
        head_dim = 5120 // num_heads
        num_layers = 40 
        B = 1
    
        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                torch.zeros([2, B, 9*880, num_heads, head_dim], dtype=dtype, device=device)
            )
    
        crossattn_k_cache = []
        for _ in range(num_layers):
            crossattn_k_cache.append(
                torch.zeros([2, B, 9*880, num_heads, head_dim], dtype=dtype, device=device)
            )
        kv_cache_packed = torch.stack(kv_cache, dim=0)
        crossattn_packed = torch.stack(crossattn_k_cache, dim=0)
        return (x, timestep, context, kv_cache_packed, y, clip_feature, action, timestep_action, state)
    elif model_type == "ar_14B_droid":
        clip_feature = torch.randn(1, 257, 1280, dtype=dtype, device=device)
        y = torch.randn(1, 20, 2, 44, 80, dtype=dtype, device=device)
        timestep_action = torch.randn(1, 24, dtype=dtype, device=device)
        x = torch.randn(1, 16, 2, 44, 80, dtype=dtype, device=device)
        timestep = torch.randn(1, 2, dtype=dtype, device=device)
        context = torch.randn(1, 512, 4096, dtype=dtype, device=device)
        seq_len = torch.tensor(1760, dtype=torch.int32, device=device)
        action = torch.randn(1, 24, 32, dtype=dtype, device=device)
        state = torch.randn(1, 1, 64, dtype=dtype, device=device)
        embodiment_id = torch.zeros(1, dtype=torch.int32, device=device)
    
        num_heads = 40 
        head_dim = 5120 // num_heads
        num_layers = 40 
        B = 1
    
        kv_cache = []
        for _ in range(num_layers):
            kv_cache.append(
                torch.zeros([2, B, 9*880, num_heads, head_dim], dtype=dtype, device=device)
            )
    
        crossattn_k_cache = []
        for _ in range(num_layers):
            crossattn_k_cache.append(
                torch.zeros([2, B, 9*880, num_heads, head_dim], dtype=dtype, device=device)
            )
        kv_cache_packed = torch.stack(kv_cache, dim=0)
        crossattn_packed = torch.stack(crossattn_k_cache, dim=0)
        return (x, timestep, context, kv_cache_packed, y, clip_feature, action, timestep_action, state)


