# WorldScape Policy — RoboDojo runtime

This source is bundled with the RoboDojo-only WorldScape Policy 2 adapter.
Use the [adapter guide](../README.md) for installation, pretrained initialization,
Flash / Memory training, hot loading and deployment. The [Chinese guide](../README_zh.md)
contains the same commands.

The native implementation retains shared conditioning, visual memory, Wan2.2,
checkpoint serialization and data primitives. Public policy entrypoints target
RoboDojo ARX-X5 absolute joint actions and the `mosaic_2x2` camera layout.
Both training recipes use z-score; inference preserves each checkpoint's own
normalization, including published q99 Memory weights.

- [Native training](docs/native_training.md)
- [Post-training](docs/posttraining.md)
- [Data hot loading](docs/robodojo_hotload.md)
- [Evaluation](docs/evaluation.md)

Based on the WorldScape Team's [WorldScape Policy](https://github.com/manifoldai-research/WorldScape-Policy).
Retain the bundled LICENSE and third-party notices when redistributing.

## 📝 Citation

```bibtex
@article{su2026worldscape_policy_2,
  title={WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory},
  author={Su, Haisheng and Liu, Zongdai and Jin, Xin and Dou, Haoxuan and Hu, Chengming and Li, Baorun and Liu, Zhanwang and Xu, Ruiyan and Fang, Jianjie and Zhang, Xin and Yang, Zhenjie and Yang, Xue and Gao, Chen and Yan, Junchi and Li, Yong and Wu, Wei},
  journal={arXiv preprint arXiv:2607.18840},
  year={2026}
}

@article{su2026worldscape_policy,
  title={WorldScape Policy: Generalizable Robotic Learning via a Foundation World Model},
  author={Su, Haisheng and Shang, Yu and Zhang, Xin and Dou, Haoxuan and Jin, Xin and Zhang, Hongling and Gao, Chen and Li, Yong and Wu, Wei},
  year={2026}
}
```

