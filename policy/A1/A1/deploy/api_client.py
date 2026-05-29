#!/usr/bin/env python3
"""
VLA Inference API Client
Test client for the VLA inference API server
"""

import os
import sys
import json
import base64
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from io import BytesIO
import numpy as np

class VLAAPIClient:
    """Client for VLA Inference API"""
    
    def __init__(self, base_url: str = "http://localhost:6789"):
        """Initialize client with API base URL"""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check API server health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Health check failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model_info", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Model info request failed: {e}")
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")
    
    def create_random_image_base64(self, width: int = 336, height: int = 336) -> str:
        """Create a random image and encode to base64"""
        try:
            # Create random RGB image
            random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(random_array)
            
            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to create random image: {e}")
    
    def run_inference(self, 
                     instruction: str,
                     images: Optional[List[str]] = None,
                    #  action_head: str = "l1_regression",
                    #  use_proprio: bool = True,
                    #  use_wrist_image: bool = True,
                    #  seq_len: int = 768,
                     proprio_data: Optional[List[List[float]]] = None,
                    #  previous_actions: Optional[List[List[float]]] = None
                     ) -> Dict[str, Any]:
        """Run VLA inference"""
        
        payload = {
            "instruction": instruction,
            # "action_head": action_head,
            # "use_proprio": use_proprio,
            # "use_wrist_image": use_wrist_image,
            # "seq_len": seq_len
        }
        
        if images is not None:
            payload["images"] = images
        if proprio_data is not None:
            payload["proprio_data"] = proprio_data
        # if previous_actions is not None:
            # payload["previous_actions"] = previous_actions
        
        try:
            response = self.session.post(
                f"{self.base_url}/inference", 
                json=payload, 
                timeout=60  # Longer timeout for inference
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Inference request failed: {e}")
    
    def run_mock_inference(self) -> Dict[str, Any]:
        """Run mock inference (for testing without real model)"""
        try:
            response = self.session.post(f"{self.base_url}/inference/mock", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Mock inference request failed: {e}")

def test_basic_connectivity(client: VLAAPIClient):
    """Test basic API connectivity"""
    print("🔍 Testing basic connectivity...")
    
    try:
        health = client.health_check()
        print(f"✅ Health check passed")
        print(f"   Status: {health.get('status')}")
        print(f"   Model loaded: {health.get('model_loaded')}")
        print(f"   Device: {health.get('device')}")
        print(f"   Timestamp: {health.get('timestamp')}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_info(client: VLAAPIClient):
    """Test model info endpoint"""
    print("\n🔍 Testing model info...")
    
    try:
        info = client.get_model_info()
        print(f"✅ Model info retrieved")
        print(f"   Model loaded: {info.get('model_loaded')}")
        print(f"   Device: {info.get('device')}")
        if 'config' in info:
            config = info['config']
            print(f"   Action head: {config.get('action_head')}")
            print(f"   Use proprio: {config.get('use_proprio')}")
            print(f"   Vision backbone: {config.get('vision_backbone')}")
        return True
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_mock_inference(client: VLAAPIClient):
    """Test mock inference endpoint"""
    print("\n🔍 Testing mock inference...")
    
    try:
        start_time = time.time()
        result = client.run_mock_inference()
        end_time = time.time()
        
        print(f"✅ Mock inference completed in {(end_time - start_time)*1000:.2f}ms")
        print(f"   Instruction: {result.get('instruction')[:50]}...")
        print(f"   Action head: {result.get('action_head_type')}")
        print(f"   Predicted actions shape: {np.array(result.get('predicted_actions', [])).shape}")
        print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")
        return True
    except Exception as e:
        print(f"❌ Mock inference failed: {e}")
        return False

def _read_json_list_of_lists(path: str) -> List[List[float]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or (len(data) > 0 and not isinstance(data[0], list)):
        raise ValueError("JSON 文件需为二维数组，例如 [[...], [...]]")
    return data


def _parse_csv_to_float_list(csv_str: str) -> List[float]:
    values = [x.strip() for x in csv_str.split(',') if x.strip() != ""]
    return [float(x) for x in values]


def test_real_inference_with_random_images(
    client: VLAAPIClient,
    user_images_b64: Optional[List[str]] = None,
    user_proprio: Optional[List[List[float]]] = None,
    # user_prev_actions: Optional[List[List[float]]] = None,
    user_instruction: Optional[str] = None,
):
    """Test real inference, prefer user-provided inputs, otherwise fallback to random."""
    print("\n🔍 Testing inference...")

    try:
        # Images
        if user_images_b64 and len(user_images_b64) > 0:
            images_b64 = user_images_b64
        else:
            main_image_b64 = client.create_random_image_base64()
            wrist_image_b64 = client.create_random_image_base64()
            images_b64 = [main_image_b64, wrist_image_b64]

        # Proprio and previous actions (shape: 1 x D)
        proprio_data = user_proprio if user_proprio is not None else np.random.randn(1, 7).tolist()
        # previous_actions = user_prev_actions if user_prev_actions is not None else np.random.randn(1, 7).tolist()

        # Instruction(s)
        if user_instruction:
            instructions = [user_instruction]
        else:
            instructions = [
                "Pick up the red block and place it on the blue block",
                "Open the drawer and put the object inside",
                "Grasp the cup and move it to the table",
            ]

        for i, instruction in enumerate(instructions):
            print(f"\n--- Test {i+1}: {instruction[:30]}... ---")

            start_time = time.time()
            result = client.run_inference(
                instruction=instruction,
                images=images_b64,
                proprio_data=proprio_data,
                # previous_actions=previous_actions,
            )
            end_time = time.time()

            print(f"✅ Inference completed in {(end_time - start_time)*1000:.2f}ms")
            print(f"   Action head: {result.get('action_head_type')}")

            actions = np.array(result.get('predicted_actions', []))
            print(f"   Predicted actions shape: {actions.shape}")
            print(f"   Predicted actions: {actions.flatten()[:7]}")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")

        return True
    except Exception as e:
        print(f"❌ Real inference failed: {e}")
        return False

def test_different_configurations(client: VLAAPIClient):
    """Test different inference configurations"""
    print("\n🔍 Testing different configurations...")
    
    instruction = "Pick up the object and move it to the target location"
    main_image_b64 = client.create_random_image_base64()
    wrist_image_b64 = client.create_random_image_base64()
    
    configurations = [
        {"action_head": "l1_regression", "use_proprio": True, "use_wrist_image": True},
        {"action_head": "l1_regression", "use_proprio": False, "use_wrist_image": True},
        {"action_head": "l1_regression", "use_proprio": True, "use_wrist_image": False},
        {"action_head": "diffusion", "use_proprio": True, "use_wrist_image": True},
    ]
    
    for i, config in enumerate(configurations):
        print(f"\n--- Configuration {i+1}: {config} ---")
        
        try:
            result = client.run_inference(
                instruction=instruction,
                images=[main_image_b64, wrist_image_b64] if config["use_wrist_image"] else [main_image_b64],
                **config
            )
            
            print(f"✅ Configuration test passed")
            print(f"   Action head: {result.get('action_head_type')}")
            actions = np.array(result.get('predicted_actions', []))
            print(f"   Actions shape: {actions.shape}")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f}ms")
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            continue
    
    return True

def benchmark_inference(client: VLAAPIClient, num_requests: int = 10):
    """Benchmark inference performance"""
    print(f"\n🔍 Benchmarking inference performance ({num_requests} requests)...")
    
    instruction = "Pick up the red block and place it on the blue block"
    main_image_b64 = client.create_random_image_base64()
    wrist_image_b64 = client.create_random_image_base64()
    
    times = []
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            result = client.run_inference(
                instruction=instruction,
                images=[main_image_b64, wrist_image_b64],
                # action_head="l1_regression"
            )
            end_time = time.time()
            
            request_time = (end_time - start_time) * 1000
            times.append(request_time)
            
            print(f"  Request {i+1}/{num_requests}: {request_time:.2f}ms")
            
        except Exception as e:
            print(f"  Request {i+1}/{num_requests} failed: {e}")
    
    if times:
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Average time: {avg_time:.2f}ms")
        print(f"   Min time: {min_time:.2f}ms")
        print(f"   Max time: {max_time:.2f}ms")
        print(f"   Std deviation: {std_time:.2f}ms")
        print(f"   Throughput: {1000/avg_time:.2f} requests/second")
        
        return True
    else:
        print("❌ No successful requests for benchmarking")
        return False

def main():
    parser = argparse.ArgumentParser(description="VLA API Client Test Suite")
    parser.add_argument("--url", default="http://localhost:6789", 
                       help="API server URL (default: http://localhost:6789)")
    parser.add_argument("--test", choices=[
        "connectivity", "model_info", "mock", "inference", "config", "benchmark", "all"
    ], default="all", help="Test to run")
    parser.add_argument("--benchmark_requests", type=int, default=10,
                       help="Number of requests for benchmark test")
    # Custom input options for inference
    parser.add_argument("--instruction", type=str, default=None, help="Instruction text for inference")
    parser.add_argument("--image", type=str, default=None, help="Path to main image file")
    parser.add_argument("--wrist_image", type=str, default=None, help="Path to wrist image file")
    parser.add_argument("--images", nargs='+', default=None, help="List of image file paths (overrides --image/--wrist_image)")
    parser.add_argument("--proprio_csv", type=str, default=None, help="Comma-separated proprio values, e.g. 'v1,v2,...' -> [[...]]")
    parser.add_argument("--proprio_json", type=str, default=None, help="Path to JSON file containing [[...], ...]")
    # parser.add_argument("--prev_csv", type=str, default=None, help="Comma-separated previous action values -> [[...]]")
    parser.add_argument("--prev_json", type=str, default=None, help="Path to JSON file containing [[...], ...]")
    
    args = parser.parse_args()
    
    print("🚀 VLA API Client Test Suite")
    print("="*50)
    print(f"API Server: {args.url}")
    print("")
    
    # Initialize client
    client = VLAAPIClient(args.url)
    
    # Run tests based on selection
    tests_passed = 0
    total_tests = 0
    
    if args.test in ["connectivity", "all"]:
        total_tests += 1
        if test_basic_connectivity(client):
            tests_passed += 1
    
    if args.test in ["model_info", "all"]:
        total_tests += 1
        if test_model_info(client):
            tests_passed += 1
    
    if args.test in ["mock", "all"]:
        total_tests += 1
        if test_mock_inference(client):
            tests_passed += 1
    
    if args.test in ["inference", "all"]:
        total_tests += 1
        # Build user-provided inputs if any
        user_images_b64: Optional[List[str]] = None
        if args.images:
            user_images_b64 = []
            for p in args.images:
                user_images_b64.append(client.encode_image_to_base64(p))
        elif args.image or args.wrist_image:
            user_images_b64 = []
            if args.image:
                user_images_b64.append(client.encode_image_to_base64(args.image))
            if args.wrist_image:
                user_images_b64.append(client.encode_image_to_base64(args.wrist_image))

        user_proprio = None
        if args.proprio_json:
            user_proprio = _read_json_list_of_lists(args.proprio_json)
        elif args.proprio_csv:
            user_proprio = [_parse_csv_to_float_list(args.proprio_csv)]

        # user_prev = None
        # if args.prev_json:
        #     user_prev = _read_json_list_of_lists(args.prev_json)
        # elif args.prev_csv:
        #     user_prev = [_parse_csv_to_float_list(args.prev_csv)]

        if test_real_inference_with_random_images(
            client,
            user_images_b64=user_images_b64,
            user_proprio=user_proprio,
            # user_prev_actions=user_prev,
            user_instruction=args.instruction,
        ):
            tests_passed += 1
    
    if args.test in ["config", "all"]:
        total_tests += 1
        if test_different_configurations(client):
            tests_passed += 1
    
    if args.test in ["benchmark", "all"]:
        total_tests += 1
        if benchmark_inference(client, args.benchmark_requests):
            tests_passed += 1
    
    # Print summary
    print("\n" + "="*50)
    print("📊 Test Summary")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    exit(main())