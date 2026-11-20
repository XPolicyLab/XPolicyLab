import numpy as np


def patch_metrics(source_high, destination_high, valid):
    source_high, destination_high, valid = map(np.asarray, (source_high, destination_high, valid))
    if (source_high.shape != (240, 320) or destination_high.shape != source_high.shape
            or valid.shape != source_high.shape or valid.dtype != bool
            or not np.isfinite(source_high).all() or not np.isfinite(destination_high).all()):
        raise ValueError('Match only bounded current public high-frequency image fields and their valid overlap')
    def patches(value):
        return value.reshape(15, 16, 20, 16).transpose(0, 2, 1, 3)
    selected = patches(valid)
    first, second = patches(source_high), patches(destination_high)
    counts = selected.sum(axis=(2, 3))
    denominators = np.maximum(counts, 1)
    first_mean = (first * selected).sum(axis=(2, 3)) / denominators
    second_mean = (second * selected).sum(axis=(2, 3)) / denominators
    first = (first - first_mean[..., None, None]) * selected
    second = (second - second_mean[..., None, None]) * selected
    first_energy = (first * first).sum(axis=(2, 3))
    second_energy = (second * second).sum(axis=(2, 3))
    scale = np.sqrt(first_energy * second_energy)
    correlation = (first * second).sum(axis=(2, 3)) / np.maximum(scale, 1e-12)
    eligible = ((counts >= 128) & (np.sqrt(first_energy / denominators) >= .8)
        & (np.sqrt(second_energy / denominators) >= .8))
    inlier = eligible & (correlation >= .60)
    rows, columns = np.where(inlier)
    pixels = int(counts[inlier].sum())
    if pixels == 0:
        return None
    agreement = float((correlation[inlier] * counts[inlier]).sum() / pixels)
    overlap = int(counts[eligible].sum())
    qualified = (pixels >= 6000 and np.ptp(rows) >= 4 and np.ptp(columns) >= 4
        and len(set(zip(rows // 5, columns // 7))) >= 4)
    metrics = {'correlation': agreement * min(1., pixels / 12000.),
        'current_matching_pixels': pixels, 'textured_overlap_pixels': overlap,
        'selection_score': agreement * pixels / overlap}
    return metrics, bool(qualified)


def rank(source_high, destination_high, valid):
    result = patch_metrics(source_high, destination_high, valid)
    return None if result is None else result[0]


def measure(source_high, destination_high, valid):
    result = patch_metrics(source_high, destination_high, valid)
    return None if result is None or not result[1] else result[0]


def run(source_high, destination_high, valid):
    result = measure(source_high, destination_high, valid)
    return None if result is None else (result['correlation'], result['current_matching_pixels'])
