<p align="center">
<img width="55%" alt="GigaTrain" src="./docs/source/imgs/logo.png?raw=true">
</p>
<h3 align="center">
An Efficient and Scalable Training Framework for AI Models
</h3>
<p align="center">
    | <a href="#-installation">Quick Start</a>
    | <a href="#-contributing">Contributing</a>
    | <a href="#-license">License</a>
    | <a href="#-citation">Citation</a> |
</p>

## ✨ Introduction

GigaTrain is an efficient and scalable training framework engineered to accelerate the development of large AI models. It provides optimized performance and streamlined training workflows, allowing researchers and developers to easily experiment with various models.

<details open>
<summary>Major features</summary>

- 🔍 **Unified distributed training**: Seamless multi-GPU/multi-node execution; supports DeepSpeed ZeRO (0/1/2/3), FSDP/FSDP2, DDP, etc.
- 🔧 **Flexible and reproducible configs**: Clean PY/YAML/JSON configuration and a registry-driven, modular design with pluggable optimizers, schedulers, samplers, transforms, etc.
- 📈 **Performance and memory efficiency**: Mixed precision (FP16/BF16/FP8), gradient accumulation, gradient checkpointing, EMA, etc.
- 📊 **Built-in monitoring and checkpointing**: Integrated experiment logging and robust checkpointing for reliable long runs and resumability.
- ⚡ **Lightweight and Easy to Use**: Simple pip/source install; developers can focus solely on implementing the key algorithm, as the framework handles repetitive, tedious, and error-prone things like backprop, logging, checkpointing, resuming, EMA, and multi-node/multi-GPU execution.

</details>

## ⚡ Installation

GigaTrain can be installed from PyPi and has to be installed in a virtual environment (venv or conda for instance):

```bash
pip3 install giga-train
```

or you can install directly from source for the latest updates:

```bash
conda create -n giga_train python=3.11.10
conda activate giga_train
git clone https://github.com/open-gigaai/giga-train.git
cd giga-train
pip3 install -e .
```

## 🚀 Getting Started

- We provide a step-by-step [example](./examples/wan/README.md) to teach you how to fine-tune a model using GigaTrain.
- Refer to [projects](https://github.com/open-gigaai/giga-models/tree/main/projects) for more examples.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

```bibtex
@misc{gigaai2025gigatrain,
    author = {GigaAI},
    title = {GigaTrain: An Efficient and Scalable Training Framework for AI Models},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/open-gigaai/giga-train}}
}
```
