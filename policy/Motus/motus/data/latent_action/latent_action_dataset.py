import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.data as data
import logging
from tqdm import tqdm
import pickle
import uuid

from transformers import AutoProcessor

from data.utils.image_utils import (
    load_video_frames,
    load_first_frame,
    get_video_frame_count,
    tensor_to_pil,
)
from utils.vlm_utils import preprocess_vlm_messages

'''
# Import normalization functions
from data.utils.norm import (
    load_quantile_stats,
    normalize_actions_with_quantiles,
)
'''


logger = logging.getLogger(__name__)


def _has_triplet_subdirs(directory: Path) -> bool:
    return all((directory / sub).exists() for sub in ["videos", "umt5_wan", "latent_action_dim14"])


def _find_leaf_dataset_dirs(root: Path) -> List[Path]:
    """Recursively find directories that contain videos/umt5_wan/latent_action subfolders."""
    results: List[Path] = []
    try:
        if _has_triplet_subdirs(root):
            results.append(root)
            return results
        for file in os.listdir(root):
            current = Path(root, file)
            results += _find_leaf_dataset_dirs(current)
    except Exception as e:
        logger.warning(f"Failed scanning {root}: {e}")
    return results


class LatentActionDataset(data.Dataset):
    """
    Multi-source pretraining dataset using latent_action as action supervision.

    Directory patterns:
      - Standard: <root>/{videos, umt5_wan, latent_action}
      - Special: recurse into subfolders (e.g., robotwin2_copy/clean) until a leaf folder containing all three exists

    Alignment rules:
      - Use video basename as episode id; require a one-to-one match across videos, umt5_wan and latent_action
      - Sampling: given condition_frame_idx and global_downsample_rate step
        video_indices = cond + (i+1)*step, i=0..num_video_frames-1
        action_indices = cond + i*step,    i=0..num_video_frames-1   (start frame for adjacent-frame latent action)
    """

    def __init__(
        self,
        dataset_dir: List[str] = None,
        *,
        global_downsample_rate: int = 6,
        num_video_frames: int = 8,
        video_size: Tuple[int, int] = (384, 320),  # (H, W)
        image_aug: bool = False,  # reserved; not used currently
        vlm_checkpoint_path: Optional[str] = None,
        max_episodes: Optional[int] = None,
        val: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.dataset_dir: List[Path] = [Path(p) for p in dataset_dir]
        self.global_downsample_rate = int(global_downsample_rate)
        self.num_video_frames = int(num_video_frames)
        self.video_size = video_size
        self.image_aug = image_aug and not val
        self.max_episodes = max_episodes

        # VLM processor (optional)
        self.vlm_processor = None
        if vlm_checkpoint_path:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                logger.info(f"Loaded VLM processor from {vlm_checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to load VLM processor: {e}")

        self.episodes: List[Dict[str, Any]] = self._scan_all_episodes()

        if self.max_episodes is not None and self.max_episodes > 0:
            self.episodes = self.episodes[: min(self.max_episodes, len(self.episodes))]

        '''
        # Load quantile statistics for latent actions (relative path)
        current_dir = Path(__file__).parent.parent  # Go up to data directory
        q_path = current_dir / "utils" / "latent_action_q01_q99.json"
        self.action_q01, self.action_q99 = load_quantile_stats(str(q_path), 'latent_action', lower_key='q01', upper_key='q99')
        logger.info(f"Loaded quantile stats (q01/q99) from {q_path}")
        '''

        logger.info(
            f"LatentActionDataset initialized with {len(self.episodes)} episodes from {len(self.dataset_dir)} roots"
        )

    def _scan_all_episodes(self) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        for root in self.dataset_dir:
            # Use versioned cache file to avoid stale/incompatible old caches
            cache_file = root / "cached_episodes.v2.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    cached_episodes = pickle.load(f)
                    logger.info(f"Loaded {len(cached_episodes)} cached episodes from {cache_file}")
                    episodes.extend(cached_episodes)
                    continue
            cur_episodes: List[Dict[str, Any]] = []
            leaf_dirs = _find_leaf_dataset_dirs(root)
            if not leaf_dirs:
                logger.warning(f"No valid leaf dataset dirs under {root}")
                continue
            pbar = tqdm(leaf_dirs)
            for d in pbar:
                pbar.set_description(f"Scanning {d}")
                videos_dir = d / "videos"
                umt5_dir = d / "umt5_wan"
                la_dir = d / "latent_action_dim14"
                metas_dir = d / "metas"

                video_stems = {Path(f).stem for f in os.listdir(videos_dir) if f.endswith(".mp4")}
                umt5_pt_stems = {Path(f).stem for f in os.listdir(umt5_dir) if f.endswith(".pt")}
                la_stems = {Path(f).stem for f in os.listdir(la_dir) if f.endswith(".pt")}
                txt_stems = {Path(f).stem for f in os.listdir(metas_dir) if f.endswith(".txt")}

                common_stems = video_stems & umt5_pt_stems & la_stems & txt_stems

                for stem in sorted(common_stems):
                    cur_episodes.append(
                        {
                            "video_path": str(videos_dir / f"{stem}.mp4"),
                            "lang_path": str(umt5_dir / f"{stem}.pt"),
                            "latent_action_path": str(la_dir / f"{stem}.pt"),
                            "text_path": str(metas_dir / f"{stem}.txt"),
                            "root": str(d),
                            "episode_name": stem,
                        }
                    )
            try:
                tmp_path = root / f"cached_episodes.{uuid.uuid4().hex}.pkl"
                with open(tmp_path, "wb") as f:
                    pickle.dump(cur_episodes, f)
                    os.replace(tmp_path, root / "cached_episodes.v2.pkl")
                    logger.info(f"Cached {len(cur_episodes)} episodes to {root / 'cached_episodes.v2.pkl'}")
            except Exception as e:
                logger.warning(f"Failed to cache episodes for {root}: {e}")
        return episodes

    def __len__(self) -> int:
        return len(self.episodes) * 100

    def _select_indices(self, total_frames: int) -> Tuple[int, List[int], List[int]]:
        step = self.global_downsample_rate
        # constraint: cond + num_video_frames*step <= total_frames - 1
        max_cond = total_frames - 1 - self.num_video_frames * step
        if max_cond < 0:
            condition_idx = 0
        else:
            condition_idx = random.randint(0, max_cond)

        video_indices = [condition_idx + (i + 1) * step for i in range(self.num_video_frames)]
        action_indices = [condition_idx + i * step for i in range(self.num_video_frames)]
        # clamp to valid ranges
        video_indices = [min(i, total_frames - 1) for i in video_indices]
        action_indices = [min(i, total_frames - 2) for i in action_indices]  # avoid indexing the last frame for actions
        return condition_idx, video_indices, action_indices

    def _load_language_embedding(self, lang_path: str) -> Tuple[torch.Tensor, int]:
        """Keep the same format and behavior as AC-One: list => random choice; tensor => use directly."""
        embedding_data = torch.load(lang_path, map_location='cpu')

        if isinstance(embedding_data, list):
            selected_idx = random.randint(0, len(embedding_data) - 1)
            embeddings = embedding_data[selected_idx]
        else:
            embeddings = embedding_data
            selected_idx = 0

        if isinstance(embeddings, torch.Tensor) and embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"Language embedding must be a Tensor or list of Tensors: {lang_path}")

        return embeddings, selected_idx

    def _load_text_instruction(self, episode: Dict[str, Any], selected_idx: int | None = None) -> str:
        """Load raw text instruction strictly.

        Rule:
        - If a 'metas' directory exists under the episode root, load <root>/metas/<stem>.txt
        - Else, require a text file with the same stem next to the umt5_wan .pt file
        """
        if "text_path" not in episode:
            raise ValueError("Episode missing text_path for instruction loading")
        txt_path = Path(episode["text_path"]) 

        if not txt_path.exists():
            raise FileNotFoundError(f"Instruction text not found: {txt_path}")
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Support multi-line metas: pick one line aligned with selected_idx (if provided), otherwise first non-empty line
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"Empty instruction text: {txt_path}")
        if selected_idx is None:
            return lines[0]
        # clamp index to valid range
        idx = max(0, min(selected_idx, len(lines) - 1))
        return lines[idx]

    def _load_latent_action(self, la_path: str) -> torch.Tensor:
        data = torch.load(la_path, map_location="cpu")
        if isinstance(data, torch.Tensor):
            return data.float()
        if isinstance(data, dict):
            if 'latent_action' in data and isinstance(data['latent_action'], torch.Tensor):
                return data['latent_action'].float()
            raise ValueError(f"latent_action dict must contain a Tensor under key 'latent_action': {la_path}")
        if isinstance(data, list):
            # latent_action must be unique; list is not allowed
            raise ValueError(f"latent_action must be a single Tensor, got list at {la_path}")
        raise TypeError(f"Unsupported latent_action format at {la_path}")

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self.episodes:
            return None
        episode = random.choice(self.episodes)

        try:
            # Latent action sequence
            la_full = self._load_latent_action(episode["latent_action_path"])  # [T - 1, D]
            if la_full.dim() == 1:
                la_full = la_full.unsqueeze(0)

            total_frames = la_full.shape[0] + 1
            if total_frames < 2:
                return None

            cond_idx, video_indices, action_indices = self._select_indices(total_frames)

            first_frame = load_video_frames(episode["video_path"], [cond_idx] + video_indices, self.video_size)  # [1 + T', C, H, W]
            video_frames = first_frame[1:]  # [T', C, H, W]
            first_frame = first_frame[0]    # [C, H, W]

            max_valid_idx = la_full.shape[0] - 1
            if max_valid_idx <= 0:
                return None
            act_idxs = [min(i, max_valid_idx) for i in action_indices]
            action_sequence = la_full[act_idxs].float()

            # Normalize latent actions using q01/q99 quantile scaling to [0, 1]
            # normalized_actions = normalize_actions_with_quantiles(action_sequence, self.action_q01, self.action_q99, clip=True)
            normalized_actions = action_sequence

            # Language embedding (WAN features)
            language_embedding, selected_lang_idx = self._load_language_embedding(episode["lang_path"])

            # VLM inputs
            vlm_tokens = None

            # Build VLM tokens only if processor is provided
            if self.vlm_processor is not None:
                # Align metas text with the selected language embedding index (if multiple)
                text_instruction = self._load_text_instruction(episode, selected_idx=selected_lang_idx)  # strict: must exist
                first_frame_pil = tensor_to_pil(first_frame)
                vlm_tokens = preprocess_vlm_messages(text_instruction, first_frame_pil, self.vlm_processor)

            return {
                "first_frame": first_frame,               # [C, H, W]
                "video_frames": video_frames,             # [F, C, H, W]
                "action_sequence": normalized_actions,    # [F, D] - normalized
                "language_embedding": language_embedding, # [L, E]
                "vlm_inputs": vlm_tokens,                 # may be None
                # Note: no initial_state
            }
        except Exception as e:
            logger.error(
                f"Error loading episode {episode.get('episode_name', 'unknown')} from {episode.get('root', '')}: {e}"
            )
            return None