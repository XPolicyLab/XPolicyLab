"""Model namespace for flexpi.

  flexpi.py / backbone.py          the FlexPi model and the backbone it extends
  dino_encoder.py                  frozen DINOv3 tokenizer
  pointmap_encoder.py              GT-depth → normalized-XYZ tokenizer
  mot.py / action_dit.py           Mixture-of-Transformers core, action expert
  wan22.py / wan_video_*.py        Wan2.2 TI2V-5B pipeline, DiT, VAE, T5

  helpers/         checkpoint loading + stateless DINO / flex-joint helpers
  schedulers/      continuous flow-match scheduler
  inference_opt/   opt-in deploy-time speed adapters (TRT, CUDA graphs)

Deliberately re-exports nothing: importing ``models.helpers.io`` should not
pull the 4k-line model in with it.
"""
