huggingface-cli download Lightricks/LTX-Video --include "tokenizer/*" --local-dir checkpoints
huggingface-cli download Lightricks/LTX-Video --include "text_encoder/*" --local-dir checkpoints
huggingface-cli download Lightricks/LTX-Video --include "vae/*" --local-dir checkpoints
huggingface-cli download OpenDriveLab-org/RISE_Assets --include "dynamics_model/*" --local-dir checkpoints