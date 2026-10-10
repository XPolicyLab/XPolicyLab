import argparse
import os
from itertools import chain
from typing import Any, Dict

import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, ProcessorMixin

from llava.data.builder import DATASETS, parse_mixture
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger
from llava.utils.media import extract_media


class ImageDataset(Dataset):
    def __init__(self, name: str, processor: ProcessorMixin) -> None:
        super().__init__()
        self.name = name
        self.processor = processor
        self.instances = []
        dataset = instantiate(DATASETS[name], _partial_=True)(tokenizer=None, data_args=None)
        for index, instance in enumerate(dataset.instances):
            messages = dataset.process(instance)
            media = extract_media(messages, draft=True)
            for k, image in enumerate(media.get("image", [])):
                self.instances.append({"uid": f"{name}/{index}-{k}", "image": image})

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]
        uid, path = instance["uid"], instance["image"].path
        image = Image.open(path).convert("RGB")
        image = self.processor(images=[image], return_tensors="pt").pixel_values[0]
        return {"uid": uid, "path": path, "image": image}

    def __len__(self) -> int:
        return len(self.instances)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmdb-dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, default=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    torch.cuda.set_device(dist.local_rank())

    # Load model and processor
    model = AutoModel.from_pretrained(args.model_name_or_path).cuda()
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    for name in tqdm(parse_mixture(args.dataset), disable=not dist.is_main()):
        if os.path.exists(os.path.join(args.mmdb_dir, name + ".jsonl")):
            logger.warning(f"Skipping '{name}' as it already exists.")
            continue

        # Set up image dataset and data loader
        dataset = ImageDataset(name, processor)
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=False,
            drop_last=False,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )

        # Extract image features and metainfos
        features, metainfos = [], []
        for batch in tqdm(data_loader, disable=not dist.is_main(), leave=False):
            image = batch["image"].cuda(non_blocking=True)
            with torch.inference_mode(), torch.cuda.amp.autocast():
                feature = model.get_image_features(image)
            features.append(feature.cpu())
            for uid, path in zip(batch["uid"], batch["path"]):
                metainfos.append({"uid": uid, "path": path})

        # Gather image features and metainfos from all ranks
        features = dist.gather(features, dst=0)
        metainfos = dist.gather(metainfos, dst=0)
        if not dist.is_main():
            continue
        features = list(chain(*features))
        metainfos = list(chain(*metainfos))

        # Save image features and metainfos
        if features and metainfos:
            io.save(os.path.join(args.mmdb_dir, name + ".pt"), torch.cat(features))
            io.save(os.path.join(args.mmdb_dir, name + ".jsonl"), metainfos)
        else:
            logger.warning(f"Skipping '{name}' as no image features are extracted.")

    # Save configuration
    if dist.is_main():
        config = {"model_name_or_path": args.model_name_or_path}
        io.save(os.path.join(args.mmdb_dir, "config.json"), config, indent=4)


if __name__ == "__main__":
    main()
