import dataclasses
import logging
from typing import List, Optional, Any, Dict, Tuple, TYPE_CHECKING
from threading import Thread
from queue import Queue, Empty

import numpy as np
import torch
from torch.utils.data import Sampler
from PIL import Image, ImageEnhance
from a1.data.dataset import DeterministicDataset
from a1.torch_util import get_world_size, get_global_rank

log = logging.getLogger(__name__)

def random_erasing_numpy(image: np.ndarray, rng: np.random.RandomState, 
                         num_patches: Optional[int] = None, 
                         patch_size_range: Tuple[int, int] = (20, 80)) -> np.ndarray:
    """
    随机遮挡/擦除：在图像上随机添加黑色方块（numpy数组版本）
    
    Args:
        image: numpy数组格式的图像 (H, W, C) 或 (H, W)，uint8类型
        rng: 随机数生成器
        num_patches: 遮挡块的数量，如果为None则随机选择1-3个
        patch_size_range: 遮挡块大小的范围（最小，最大）
    
    Returns:
        处理后的图像（numpy数组）
    """
    img_array = np.copy(image)
    h, w = img_array.shape[:2]
    
    if num_patches is None:
        num_patches = rng.randint(1, 4)  # 1-3个
    
    for _ in range(num_patches):
        # 随机选择遮挡块大小
        patch_size = rng.randint(patch_size_range[0], patch_size_range[1] + 1)
        
        # 随机选择位置（确保不超出边界）
        x = rng.randint(0, max(1, w - patch_size + 1))
        y = rng.randint(0, max(1, h - patch_size + 1))
        
        # 绘制黑色方块
        if len(img_array.shape) == 3:
            img_array[y:y+patch_size, x:x+patch_size] = [0, 0, 0]
        else:
            img_array[y:y+patch_size, x:x+patch_size] = 0
    
    return img_array


def sharpen_image_numpy(image: np.ndarray, rng: np.random.RandomState, 
                       factor: Optional[float] = None) -> np.ndarray:
    """
    边缘增强与锐化（numpy数组版本）
    
    Args:
        image: numpy数组格式的图像 (H, W, C)，uint8类型
        rng: 随机数生成器
        factor: 锐化强度，如果为None则随机选择2.0-3.5
    
    Returns:
        处理后的图像（numpy数组）
    """
    if factor is None:
        factor = rng.uniform(2.0, 3.5)
    
    # 转换为PIL Image
    if len(image.shape) == 3:
        pil_image = Image.fromarray(image, mode='RGB')
    else:
        pil_image = Image.fromarray(image, mode='L')
    
    # 使用PIL的锐化滤镜
    enhancer = ImageEnhance.Sharpness(pil_image)
    sharpened = enhancer.enhance(factor)
    
    # 转换回numpy数组
    return np.array(sharpened)


def apply_image_augmentation(item: Dict[str, Any], rng: np.random.RandomState,
                             enable_random_erasing: bool = True,
                             enable_sharpening: bool = True,
                             augmentation_prob: float = 0.5) -> Dict[str, Any]:
    """
    对数据项中的图像应用增强
    
    Args:
        item: 数据字典，可能包含 'images' 或 'image' 字段
        rng: 随机数生成器
        enable_random_erasing: 是否启用随机遮挡
        enable_sharpening: 是否启用锐化
        augmentation_prob: 应用增强的概率
    
    Returns:
        处理后的数据字典
    """
    if rng.random() > augmentation_prob:
        return item
    
    # 决定应用哪种增强
    effects = []
    if enable_random_erasing:
        effects.append('erasing')
    if enable_sharpening:
        effects.append('sharpening')
    
    if not effects:
        return item
    
    # 随机选择一种或多种增强效果
    effect = rng.choice(effects)
    
    # 处理 'images' 字段（列表）
    if 'images' in item and isinstance(item['images'], list):
        augmented_images = []
        for img in item['images']:
            if isinstance(img, np.ndarray) and len(img.shape) >= 2:
                if effect == 'erasing':
                    img = random_erasing_numpy(img, rng)
                elif effect == 'sharpening':
                    img = sharpen_image_numpy(img, rng)
                # 如果同时启用两种效果，随机决定是否都应用
                if len(effects) > 1 and rng.random() < 0.3:  # 30%概率同时应用两种效果
                    if effect == 'erasing' and enable_sharpening:
                        img = sharpen_image_numpy(img, rng)
                    elif effect == 'sharpening' and enable_random_erasing:
                        img = random_erasing_numpy(img, rng)
            augmented_images.append(img)
        item['images'] = augmented_images
    
    # 处理 'image' 字段（列表或单个图像）
    if 'image' in item:
        if isinstance(item['image'], list):
            augmented_images = []
            for img in item['image']:
                if isinstance(img, np.ndarray) and len(img.shape) >= 2:
                    if effect == 'erasing':
                        img = random_erasing_numpy(img, rng)
                    elif effect == 'sharpening':
                        img = sharpen_image_numpy(img, rng)
                    if len(effects) > 1 and rng.random() < 0.3:
                        if effect == 'erasing' and enable_sharpening:
                            img = sharpen_image_numpy(img, rng)
                        elif effect == 'sharpening' and enable_random_erasing:
                            img = random_erasing_numpy(img, rng)
                augmented_images.append(img)
            item['image'] = augmented_images
        elif isinstance(item['image'], np.ndarray) and len(item['image'].shape) >= 2:
            if effect == 'erasing':
                item['image'] = random_erasing_numpy(item['image'], rng)
            elif effect == 'sharpening':
                item['image'] = sharpen_image_numpy(item['image'], rng)
            if len(effects) > 1 and rng.random() < 0.3:
                if effect == 'erasing' and enable_sharpening:
                    item['image'] = sharpen_image_numpy(item['image'], rng)
                elif effect == 'sharpening' and enable_random_erasing:
                    item['image'] = random_erasing_numpy(item['image'], rng)
    
    return item
class IterableDatasetMixture(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """Infinitely iterates over a mixture of datasets"""

    def __init__(
        self,
        datasets: List[DeterministicDataset],
        global_batch_size: int,
        mixture_rates: List[float]=None,
        seed: int = 0,
        start_index: int = 0,
        shuffle: bool = True,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        stratify: bool = False,
        worker_info=None
    ):
        self.datasets = list(datasets)
        if mixture_rates:
            self.mixture_rates = np.array(mixture_rates, dtype=np.float32)
        else:
            self.mixture_rates = None

        self.seed = seed
        assert seed is not None
        self.start_index = start_index
        self.shuffle = shuffle
        self.world_size = world_size if world_size is not None else get_world_size()
        self.rank = rank if rank is not None else get_global_rank()
        self.global_batch_size = global_batch_size
        assert self.global_batch_size % self.world_size == 0
        self.device_batch_size = global_batch_size // self.world_size
        self.stratify = stratify
        self.worker_info = worker_info  # For testing

    def _get_next_sources(self, rng, counts):
        if len(self.datasets) == 1:
            return np.zeros(self.global_batch_size, dtype=np.int32)
        if self.stratify:
            out = []
            counts = np.copy(counts)
            total = counts.sum()
            for _ in range(self.global_batch_size):
                # Sample the most under-represented dataset
                ix = np.argmax(np.abs(counts/total - self.mixture_rates))
                out.append(ix)
                counts[ix] += 1
                total += 1
            return np.array(out)
        else:
            return rng.choice(
                len(self.datasets),             # if 3
                size=self.global_batch_size,    # if 4
                p=self.mixture_rates            # if [0.5, 0.25, 0.25]
            )                                   # return [0, 1, 2, 0]

    def __iter__(self):
        worker_info = self.worker_info or torch.utils.data.get_worker_info()
        batch_ix = 0
        rng = np.random.RandomState(self.seed)

        # How often each dataset has been sampled globally across all devices/workers
        counts = np.zeros(len(self.datasets), dtype=np.int64)
        if self.start_index != 0:
            assert self.start_index % self.global_batch_size == 0
            start_batch = self.start_index // self.global_batch_size
            if worker_info is None:
                log.info(f"Fast forwarding instance {self.start_index}, batch {start_batch}...")
            for i in range(start_batch):
                ix = self._get_next_sources(rng, counts)
                batch_ix += 1
                np.add.at(counts, ix, 1)
            if worker_info is None:
                log.info(f"Done")
        shuffled_ixs = [(None, None) for _ in self.datasets]

        while True:
            ix = self._get_next_sources(rng, counts)
            if worker_info and batch_ix % worker_info.num_workers != worker_info.id:
                # Workers participate in every num_workers-th batch, `DataLoader` collects complete
                # batches from individual workers one-by-one so this ensures the number
                # of workers does not affect the order of the data
                np.add.at(counts, ix, 1)
                batch_ix += 1
                continue

            batch_ix += 1
            for i, dataset_ix in enumerate(ix):
                count = counts[dataset_ix]
                counts[dataset_ix] += 1

                if (i + self.rank) % self.world_size != 0:
                    continue
                device_ix = (i + self.rank) // self.world_size

                dataset = self.datasets[dataset_ix]
                epoch = count // len(dataset)

                shuffled_for, shuffled_order = shuffled_ixs[dataset_ix]
                if epoch != shuffled_for:
                    shuffle_seed = self.seed + epoch * 1771
                    shuffled_order = np.arange(len(dataset), dtype=np.int32)
                    np.random.RandomState(shuffle_seed).shuffle(shuffled_order)
                    shuffled_ixs[dataset_ix] = (epoch, shuffled_order)

                yield dataset.get(int(shuffled_order[count % len(dataset)]), epoch)

### jian
class MultiSourceIterableDataset(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """Mix multiple iterable datasets by sampling a source per sample using given weights.

    This class expects each source to be an iterable dataset (e.g., RLDSDataset (iterable mode) or
    IterableDatasetWrapper). Each iteration yields one item from one of the sources, where the source
    index is sampled according to normalized `weights`.

    Notes:
    - Each worker/device will maintain its own iterator for every source to avoid cross-worker
      interference. Iterators are recreated lazily when exhausted.
    - We keep the interface simple: infinite stream controlled by outer DataLoader epoching/steps.
    """

    def __init__(self, datasets: List[torch.utils.data.IterableDataset], weights: List[float], seed: int = 0,
                 prefetch_per_source: int = 32,
                 source_block_size: int = 1,
                 enable_steal: bool = True):
        assert len(datasets) > 0, "At least one dataset is required"
        assert len(datasets) == len(weights), "Datasets and weights must be the same length"
        self.datasets = datasets
        weights = np.array(weights, dtype=np.float64)
        assert np.all(weights >= 0), "Weights must be non-negative"
        total = float(weights.sum())
        assert total > 0, "At least one weight must be positive"
        self.probs = (weights / total).astype(np.float64)
        self.seed = seed
        # Prefetch and chunking controls
        self.prefetch_per_source = max(1, int(prefetch_per_source))
        self.source_block_size = max(1, int(source_block_size))
        self.enable_steal = bool(enable_steal)
        # rank-aware sharding across processes
        self.world_size = get_world_size()
        

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        self.rank = get_global_rank()
        rng = np.random.RandomState(self.seed + 12345 + worker_id + self.rank*1000)

        # Create one iterator per dataset for this worker
        source_iters = [iter(ds) for ds in self.datasets]

        # One bounded queue per source and a background prefetch thread
        queues: List[Queue] = [Queue(maxsize=self.prefetch_per_source * 2) for _ in self.datasets]
        stop_flags = [False for _ in self.datasets]

        def _producer_loop(src_i: int):
            q = queues[src_i]
            it = source_iters[src_i]
            produced = 0  # count elements to implement simple modulo sharding across ranks
            while not stop_flags[src_i]:
                try:
                    item = next(it)
                except StopIteration:
                    # Recreate exhausted iterator and continue
                    it = iter(self.datasets[src_i])
                    source_iters[src_i] = it
                    continue
                # Simple rank-based sharding: only keep items where produced % world_size == rank
                if (produced % max(1, self.world_size)) == self.rank:
                    # Block if queue is full
                    q.put(item)
                produced += 1

        threads: List[Thread] = []
        for i in range(len(self.datasets)):
            t = Thread(target=_producer_loop, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        # Chunked sampling to reduce cross-source switching
        current_src = None
        remaining_in_block = 0

        while True:
            if remaining_in_block <= 0 or current_src is None:
                current_src = int(rng.choice(len(self.datasets), p=self.probs))
                remaining_in_block = self.source_block_size

            # Try non-blocking get from the current source first
            try:
                item = queues[current_src].get_nowait()
                remaining_in_block -= 1
                yield item
                continue
            except Empty:
                pass

            # Optionally steal from other ready sources to avoid stall
            if self.enable_steal:
                stolen = False
                for alt in rng.permutation(len(self.datasets)):
                    if alt == current_src:
                        continue
                    try:
                        item = queues[alt].get_nowait()
                        # Reset block to the stolen source to leverage locality
                        current_src = alt
                        remaining_in_block = self.source_block_size - 1
                        yield item
                        stolen = True
                        break
                    except Empty:
                        continue
                if stolen:
                    continue

            # If nothing available immediately, block on current source
            item = queues[current_src].get()
            remaining_in_block -= 1
            yield item

    def __len__(self):
        """Return a finite proxy length for logging/scheduling purposes.

        For a mixture of iterable datasets, we define the length as the sum of
        lengths of the underlying datasets when available. This aligns with the
        intuition of total available samples across sources and is sufficient
        for informative logging (e.g., printing dataset size).
        """
        return int(sum(len(ds) for ds in self.datasets))


class SimpleMultiSourceIterableDataset(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """Mix multiple iterable datasets by sampling a source per sample using given weights.
    
    Simplified version without multi-threading. All datasets are loaded in the same process.
    This class expects each source to be an iterable dataset (e.g., RLDSDataset (iterable mode) or
    IterableDatasetWrapper). Each iteration yields one item from one of the sources, where the source
    index is sampled according to normalized `weights`.

    Notes:
    - Each worker/device will maintain its own iterator for every source to avoid cross-worker
      interference. Iterators are recreated lazily when exhausted.
    - No multi-threading: all data loading happens synchronously in the main process.
    - We keep the interface simple: infinite stream controlled by outer DataLoader epoching/steps.
    """

    def __init__(self, datasets: List[torch.utils.data.IterableDataset], weights: List[float], seed: int = 0,
                 source_block_size: int = 1,
                 enable_image_augmentation: bool = True,
                enable_random_erasing: bool = True,
                enable_sharpening: bool = True,
                augmentation_prob: float = 0.5,
                augmentation_seed: Optional[int] = None
        ):
        assert len(datasets) > 0, "At least one dataset is required"
        assert len(datasets) == len(weights), "Datasets and weights must be the same length"
        self.datasets = datasets
        weights = np.array(weights, dtype=np.float64)
        assert np.all(weights >= 0), "Weights must be non-negative"
        total = float(weights.sum())
        assert total > 0, "At least one weight must be positive"
        self.probs = (weights / total).astype(np.float64)
        self.seed = seed
        # Chunking control: number of consecutive samples from the same source
        self.source_block_size = max(1, int(source_block_size))

        # 图像增强相关参数
        self.enable_image_augmentation = enable_image_augmentation
        self.enable_random_erasing = enable_random_erasing
        self.enable_sharpening = enable_sharpening
        self.augmentation_prob = augmentation_prob
        self.augmentation_seed = augmentation_seed if augmentation_seed is not None else seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        self.rank = get_global_rank()
        rng = np.random.RandomState(self.seed + 12345 + worker_id + self.rank*1000)
        augmentation_rng = np.random.RandomState(
            self.augmentation_seed + worker_id + self.rank * 1000
        )
        # Create one iterator per dataset for this worker
        source_iters = [iter(ds) for ds in self.datasets]

        # Chunked sampling to reduce cross-source switching
        current_src = None
        remaining_in_block = 0

        while True:
            # Select a new source if needed
            if remaining_in_block <= 0 or current_src is None:
                current_src = int(rng.choice(len(self.datasets), p=self.probs))
                remaining_in_block = self.source_block_size

            # Get next item from current source
            try:
                item = next(source_iters[current_src])
                # 应用图像增强
                if self.enable_image_augmentation:
                    item = apply_image_augmentation(
                        item, 
                        augmentation_rng,
                        enable_random_erasing=self.enable_random_erasing,
                        enable_sharpening=self.enable_sharpening,
                        augmentation_prob=self.augmentation_prob
                    )
                remaining_in_block -= 1
                yield item
            except StopIteration:
                # Recreate exhausted iterator and continue
                source_iters[current_src] = iter(self.datasets[current_src])
                continue

    def __len__(self):
        """Return a finite proxy length for logging/scheduling purposes.

        For a mixture of iterable datasets, we define the length as the sum of
        lengths of the underlying datasets when available. This aligns with the
        intuition of total available samples across sources and is sufficient
        for informative logging (e.g., printing dataset size).
        """
        return int(sum(len(ds) for ds in self.datasets))