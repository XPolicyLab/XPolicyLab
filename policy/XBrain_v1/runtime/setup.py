from setuptools import find_namespace_packages, setup


setup(
    name="xbrain-v1-inference-runtime",
    version="0.1.0",
    description="Inference-only runtime required by XBrain-v1 RoboDojo policies",
    license="Apache-2.0",
    packages=find_namespace_packages(
        include=("giga_models*", "xbrain_v1_runtime*")
    ),
    install_requires=[
        "accelerate==1.13.0",
        "diffusers==0.34.0",
        "einops==0.8.0",
        "numpy<2",
        "Pillow==10.4.0",
        "peft>=0.15.0",
        "safetensors==0.5.2",
        "timm>=1.0.19",
        "torch",
        "torchvision",
        "transformers==4.54.1",
    ],
    python_requires=">=3.11",
)
