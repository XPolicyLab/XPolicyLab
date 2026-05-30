"""
PIL & Matplotlib helper utilities.

Features:
- save_plot_and_numpy: Save a matplotlib figure/pyplot, PIL image(s), or numpy array to image and its rendered numpy array.
- compare_images_and_mask: Compare two images; if different, export originals and a masked shadow overlay highlighting differences.

Usage examples are provided in the __main__ block.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

# If set to "1", generate an additional contrast-stretched visualization PNG
import os

DEBUG_NORMALIZE = os.getenv("PIL_TOOLS_DEBUG_NORMALIZE", "0") == "1"


def _ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _rgb_stats(arr: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    """Compute per-channel min/max/mean assuming arr is RGB uint8."""
    if arr.ndim != 3 or arr.shape[2] < 3:
        return {"min": (float(arr.min()),) * 3, "max": (float(arr.max()),) * 3, "mean": (float(arr.mean()),) * 3}
    mins = arr.min(axis=(0, 1))[:3]
    maxs = arr.max(axis=(0, 1))[:3]
    means = arr.mean(axis=(0, 1))[:3]
    return {
        "min": (float(mins[0]), float(mins[1]), float(mins[2])),
        "max": (float(maxs[0]), float(maxs[1]), float(maxs[2])),
        "mean": (float(means[0]), float(means[1]), float(means[2])),
    }


def _pil_to_uint8_rgb(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to uint8 RGB, scaling float data in [0,1] up to [0,255]."""
    # Convert to RGB first to drop alpha and standardize channels
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb)

    arr = np.clip(arr, 0, 255)
    # If values are very small (<=1), scale up assuming [0,1] float or 0/1 mask
    if arr.max() <= 1.0:
        arr = arr * 255.0
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    return arr


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-channel min-max stretch to full [0,255] for visualization."""
    arr = arr.astype(np.float32)
    mins = arr.min(axis=(0, 1), keepdims=True)
    maxs = arr.max(axis=(0, 1), keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    stretched = (arr - mins) / denom * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def save_plot_and_numpy(
    plt_or_fig: object,
    save_path: Union[str, Path],
    dpi: int = 200,
    close: bool = True,
) -> Dict[str, str]:
    """
    Save an input as an image and also export its pixel buffer as a numpy array (.npy).

    Supported inputs:
    - matplotlib.pyplot module (commonly named `plt`) or a matplotlib Figure instance
    - PIL.Image.Image
    - list/tuple of PIL.Image.Image (concatenated horizontally)
    - numpy array (H,W[,3])

    Args:
        plt_or_fig: The input to render and save.
        save_path: Target path. If ends with ".png/.jpg/.jpeg" it is used directly,
                   and .npy is derived from the same stem. Otherwise, we append
                   ".png" and ".npy" to the provided base path.
        dpi: DPI when saving matplotlib figures.
        close: Whether to close the matplotlib figure after saving.

    Returns:
        dict with keys:
            - image: saved image path
            - numpy: saved numpy path
    """
    save_path = Path(save_path)
    if save_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image_path = save_path
        numpy_path = image_path.with_suffix(".npy")
    else:
        image_path = save_path.with_suffix(".png")
        numpy_path = save_path.with_suffix(".npy")

    _ensure_dir(image_path.parent)

    # Branch 1: PIL.Image.Image
    if isinstance(plt_or_fig, Image.Image):
        img_arr = _pil_to_uint8_rgb(plt_or_fig)
        img = Image.fromarray(img_arr, mode="RGB")
        img.save(str(image_path))
        if DEBUG_NORMALIZE:
            vis_arr = _normalize_uint8(img_arr)
            Image.fromarray(vis_arr, mode="RGB").save(str(image_path.with_stem(image_path.stem + "_vis")))
        np.save(str(numpy_path), img_arr)
        saved_arr = np.array(Image.open(image_path).convert("RGB"))
        return {
            "image": str(image_path),
            "numpy": str(numpy_path),
            "stats": _rgb_stats(img_arr),
            "saved_stats": _rgb_stats(saved_arr),
        }

    # Branch 2: list/tuple of PIL images -> concatenate horizontally
    if isinstance(plt_or_fig, (list, tuple)) and len(plt_or_fig) > 0 and all(isinstance(x, Image.Image) for x in plt_or_fig):
        imgs = [Image.fromarray(_pil_to_uint8_rgb(x), mode="RGB") for x in plt_or_fig]
        base_w, base_h = imgs[0].size
        imgs = [im.resize((base_w, base_h), Image.BILINEAR) if im.size != (base_w, base_h) else im for im in imgs]
        total_w = base_w * len(imgs)
        concat = Image.new("RGB", (total_w, base_h))
        x_off = 0
        for im in imgs:
            concat.paste(im, (x_off, 0))
            x_off += base_w
        concat_arr = np.array(concat)
        concat.save(str(image_path))
        if DEBUG_NORMALIZE:
            vis_arr = _normalize_uint8(concat_arr)
            Image.fromarray(vis_arr, mode="RGB").save(str(image_path.with_stem(image_path.stem + "_vis")))
        np.save(str(numpy_path), concat_arr)
        saved_arr = np.array(Image.open(image_path).convert("RGB"))
        return {"image": str(image_path), "numpy": str(numpy_path), "stats": _rgb_stats(concat_arr), "saved_stats": _rgb_stats(saved_arr)}

    # Branch 3: numpy array
    if isinstance(plt_or_fig, np.ndarray):
        arr = plt_or_fig
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255)
            if arr.max() <= 1.0:
                arr = (arr * 255.0)
            arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img_arr = np.array(img)
        img.save(str(image_path))
        if DEBUG_NORMALIZE:
            vis_arr = _normalize_uint8(img_arr)
            Image.fromarray(vis_arr, mode="RGB").save(str(image_path.with_stem(image_path.stem + "_vis")))
        np.save(str(numpy_path), img_arr)
        saved_arr = np.array(Image.open(image_path).convert("RGB"))
        return {
            "image": str(image_path),
            "numpy": str(numpy_path),
            "stats": _rgb_stats(img_arr),
            "saved_stats": _rgb_stats(saved_arr),
        }

    # Branch 4: matplotlib pyplot or figure
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        raise TypeError(
            "plt_or_fig must be matplotlib Figure/pyplot, PIL.Image, list of PIL.Images, or numpy array"
        )

    fig = None
    if hasattr(plt_or_fig, "savefig") and hasattr(plt_or_fig, "canvas"):
        fig = plt_or_fig  # Figure
    elif hasattr(plt_or_fig, "gcf"):
        fig = plt_or_fig.gcf()  # pyplot
    else:
        raise TypeError(
            "plt_or_fig must be matplotlib Figure/pyplot, PIL.Image, list of PIL.Images, or numpy array"
        )

    fig.canvas.draw()
    fig.savefig(str(image_path), dpi=dpi, bbox_inches="tight")
    img_arr = np.array(Image.open(image_path).convert("RGB"))
    if DEBUG_NORMALIZE:
        vis_arr = _normalize_uint8(img_arr)
        Image.fromarray(vis_arr, mode="RGB").save(str(image_path.with_stem(image_path.stem + "_vis")))
    np.save(str(numpy_path), img_arr)

    if close:
        try:
            plt.close(fig)
        except Exception:
            pass

    saved_arr = np.array(Image.open(image_path).convert("RGB"))
    return {
        "image": str(image_path),
        "numpy": str(numpy_path),
        "stats": _rgb_stats(img_arr),
        "saved_stats": _rgb_stats(saved_arr),
    }


def compare_images_and_mask(
    img_path_a: Union[str, Path],
    img_path_b: Union[str, Path],
    out_dir: Union[str, Path] = "images/pil_results",
    tolerance: float = 0.0,
    shadow_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.6,
) -> Dict[str, Union[bool, Dict[str, str]]]:
    """
    Compare two images; if different, export A/B originals and a masked version of A highlighting differences vs B.

    The mask overlays a semi-transparent shadow on pixels where A and B differ beyond `tolerance`.

    Args:
        img_path_a: Path to the first image.
        img_path_b: Path to the second image.
        out_dir: Output directory to write results; default is "images/pil_results".
        tolerance: Per-pixel average absolute difference threshold in [0, 255]. 0 means any change.
        shadow_color: RGB color used for shadow overlay.
        alpha: Alpha for shadow overlay in [0.0, 1.0].

    Returns:
        dict with keys:
            - is_same (bool): True if images are identical within tolerance; False otherwise.
            - outputs (dict): When different, contains paths for 'A', 'B', and 'A_masked'.
    """
    img_path_a = Path(img_path_a)
    img_path_b = Path(img_path_b)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    a = Image.open(img_path_a).convert("RGB")
    b = Image.open(img_path_b).convert("RGB")

    if a.size != b.size:
        b = b.resize(a.size, Image.BILINEAR)

    a_np = np.array(a, dtype=np.int16)
    b_np = np.array(b, dtype=np.int16)

    mad = np.mean(np.abs(a_np - b_np), axis=2)
    diff_mask = mad > tolerance

    is_same = not bool(np.any(diff_mask))
    outputs: Dict[str, str] = {}

    if is_same:
        return {"is_same": True, "outputs": outputs}

    stem_a = img_path_a.stem
    stem_b = img_path_b.stem
    out_a = out_dir / f"{stem_a}_A.png"
    out_b = out_dir / f"{stem_b}_B.png"
    a.save(out_a)
    b.save(out_b)

    h, w = diff_mask.shape
    shadow_np = np.zeros((h, w, 3), dtype=np.uint8)
    shadow_np[..., 0] = shadow_color[0]
    shadow_np[..., 1] = shadow_color[1]
    shadow_np[..., 2] = shadow_color[2]

    a_float = np.array(a, dtype=np.float32)
    shadow_float = shadow_np.astype(np.float32)

    blended = a_float.copy()
    blended[diff_mask] = (1.0 - alpha) * blended[diff_mask] + alpha * shadow_float[diff_mask]
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    masked_img = Image.fromarray(blended, mode="RGB")

    out_masked = out_dir / f"{stem_a}_A_masked.png"
    masked_img.save(out_masked)

    outputs = {"A": str(out_a), "B": str(out_b), "A_masked": str(out_masked)}
    return {"is_same": False, "outputs": outputs}


if __name__ == "__main__":
    # Quick demo: generate a simple plot and save
    try:
        import matplotlib.pyplot as plt
        x = np.linspace(0, 2 * np.pi, 200)
        y = np.sin(x)
        plt.figure(figsize=(4, 3))
        plt.plot(x, y, label="sin(x)")
        plt.title("Demo Plot")
        plt.legend()
        out = save_plot_and_numpy(plt, Path("images/pil_results/demo_plot"))
        print("Saved:", out)
    except Exception as e:
        print("Matplotlib demo skipped:", e)

    # Demo: save PIL images (single and list)
    try:
        img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        out_single = save_plot_and_numpy(img1, Path("images/pil_results/demo_pil_single"))
        print("Saved single PIL:", out_single)
        out_list = save_plot_and_numpy([img1, img2], Path("images/pil_results/demo_pil_concat"))
        print("Saved list PIL:", out_list)
    except Exception as e:
        print("PIL demo skipped:", e)

    # Quick demo: compare two images (if available)
    a_path = Path("images/test/a.png")
    b_path = Path("images/test/b.png")
    if a_path.exists() and b_path.exists():
        result = compare_images_and_mask(a_path, b_path, out_dir=Path("images/pil_results"))
        print("Compare result:", result)
    else:
        print("Image compare demo skipped: images/test/a.png or b.png not found.")
