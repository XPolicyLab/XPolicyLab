# 🌏 RISE

<div id="top" align="left">
  <a href="https://opendrivelab.com/rise/"><img src="https://img.shields.io/badge/Proj_Page-blue" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2602.11075"><img src="https://img.shields.io/badge/arXiv-2602.11075-b31b1b" alt="arXiv"></a>
  <a href="https://x.com/jiazhi_yang2024/status/2022247675060797636"><img src="https://img.shields.io/badge/summary-000000?logo=x&logoColor=white" alt="X"></a>
</div>


<img src="rise_assets/teaser.gif" alt="RISE demo" width="100%">
</img>

## 🔥 Highlights

<!-- RISE is a self-improving robot policy framework that turns world models into a practical learning environment for real-world manipulation. In short, we make the following three key contributions: -->

- **A compositional world model.**
A principled design that combines a controllable multi-view dynamics model with a progress value model, yielding informative advantages for robust policy improvement.
- **RL in imagination.**
A scalable self-improving framework that bootstraps robot policies through imaginary rollouts, avoiding the hardware cost and laborious reset of real-world interactions.
- **Real-world manipulation gains.**
Non-trivial performance improvements on challenging dexterous tasks, including +35% on dynamic brick sorting, +45% on backpack packing, and +35% on box closing.

## 🗺️ Overview

RISE repository is structured with three major parts:

- `policy_and_value/policy_offline_and_value`: OpenPI-based offline policy and value training. They are put together since they share similar model architecture and training pipeline.
- `dynamics/dynamics_model`: Action-conditioned dynamics model.
- `policy_and_value/policy_online`: Online RL for policy improvement.


## 🧑‍💻 Getting Started
- [💻 Installation](docs/installation.md)
- [🏆 Offline Policy Training & Value Model](docs/offline_learning.md)
- [🔮 Dynamics Model](docs/dynamics_model.md)
- [🦾 Online Policy Training](docs/online_training.md)
- [🛠️ Deployment on Piper](docs/deploy.md)



## ❤️ Acknowledgement

We thank the following projects for their open-source contributions: [OpenPI Pi0 / Pi05](https://github.com/openpi/openpi), [Genie Envisioner](https://github.com/AgibotTech/Genie-Envisioner), [RLinf](https://github.com/RLinf/RLinf), [LTX-Video](https://github.com/Lightricks/LTX-Video).

## 📢 News
- [2026/04/27] 🚀 RISE got accepted to RSS 2026.
- [2026/04/22] Training code and pre-trained dynamics model are released.
- [2026/02/11] Paper released on [arXiv](https://arxiv.org/abs/2602.11075).

## 📄 License and Citation

All assets and code in this repository are under the Apache 2.0 license unless specified otherwise. The data and checkpoint are under CC BY-NC-SA 4.0. Other modules inherit their own distribution licenses.

```bibtex
@article{rise2026,
  title={RISE: Self-Improving Robot Policy with Compositional World Model},
  author={Yang, Jiazhi and Lin, Kunyang and Li, Jinwei and Zhang, Wencong and Lin, Tianwei and Wu, Longyan and Su, Zhizhong and Zhao, Hao and Zhang, Ya-Qin and Chen, Li and Luo, Ping and Yue, Xiangyu and Li, Hongyang},
  journal={arXiv preprint arXiv:2602.11075},
  year={2026}
}
```

[![Star History Chart](https://api.star-history.com/svg?repos=OpenDriveLab/RISE&type=Date)](https://star-history.com/#OpenDriveLab/RISE&Date)
