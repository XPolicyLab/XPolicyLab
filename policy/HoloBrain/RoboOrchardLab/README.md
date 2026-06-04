# RoboOrchardLab

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-green.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/HorizonRobotics/robo_orchard_lab/blob/master/LICENSE)
[![Documentation](https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online)](https://horizonrobotics.github.io/robot_lab/robo_orchard/lab/index.html)

**RoboOrchardLab** is a python package that provides a modular and extensible framework for training and evaluating embodied AI algorithms.
It is part of the **RoboOrchard** project, which aims to support the full lifecycle of robotics research and development.

The goal of **RoboOrchardLab** is to create a unified and flexible architecture that allows researchers and developers to easily compose and customize their training pipelines, while also providing model components, datasets, and evaluation metrics that are specifically designed for embodied AI tasks and research.

## Core Features

- **Modular Training Framework**: We provide a highly modular training framework that allows you to easily compose and customize your training pipelines. For example, you can easily add or change the implementation of model updates, training monitors or other components to suit your specific research needs without changing the overall training pipeline.
- **Efficient Distributed Training**: This project is designed to work seamlessly with the Hugging Face ecosystem (Accelerate, Datasets, etc), enabling efficient distributed training across multiple GPUs and nodes. This allows you to scale your experiments and leverage the power of modern hardware.
- **SOTA Embodied AI Algorithms**: Access an expanding suite of state-of-the-art (SOTA) algorithms. Initial support focuses on advanced perception, with intelligent grasping policies and end-to-end Vision-Language Action (VLA) models being actively developed and planned for near-future releases. Longer-term, we aim to integrate broader manipulation and whole-body control capabilities.
- **Open Community**. **RoboOrchardLab** is an open-source project, and we warmly welcome developers and researchers from academia and industry to join us. Contribute code, share models, improve documentation, and let's advance embodied intelligence technology together.

## License

**RoboOrchardLab** is open-source and licensed under the [Apache License 2.0](https://github.com/HorizonRobotics/robo_orchard_lab/blob/master/LICENSE). If you are interested in contributing, please reach out to us.
