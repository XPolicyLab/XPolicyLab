"""Instance masks from text or point prompts (segmentation server)."""

from harness_policy.runtime.session import Observation


class Segmentation:
    def segment_text(self, observation, prompt, camera="cam_head", threshold=None):
        """Instance masks for `prompt` in observation.images[camera], best first.
        By default keeps instances scoring at least half the best score."""
        if self.segmenter is None:
            raise RuntimeError("No segmentation server is configured for this harness")
        return self.segmenter.segment(observation.images[camera], prompt, threshold)

    def segment_point(self, observation, xy, camera="cam_head", labels=None):
        """Masks of the object under pixel xy (or a list of pixels) in `camera`, best
        first. labels: 1 = on the object (default), 0 = background, one per pixel."""
        if self.segmenter is None:
            raise RuntimeError("No segmentation server is configured for this harness")
        points = [xy] if len(xy) == 2 and not hasattr(xy[0], "__len__") else list(xy)
        image = observation.images[camera] if isinstance(observation, Observation) else observation
        return self.segmenter.segment_point(image, points, labels)
