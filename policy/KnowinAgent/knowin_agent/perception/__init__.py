"""Perception: text grounding plus RGB-D geometry in the world frame.

`Perceiver` is the only entry point the runtime uses. Segmentation and depth are
pluggable services so the same geometry runs on simulator depth, a learned
depth model, or a physical RGB-D camera.
"""

from knowin_agent.perception.types import (  # noqa: F401
    AxisTipEstimate,
    DepthSource,
    Mask,
    ObjectEstimate,
    OpeningEstimate,
    PerceptionError,
    Segmenter,
    SupportPlaneEstimate,
)
from knowin_agent.perception.perceiver import Perceiver  # noqa: F401
from knowin_agent.perception.services import (  # noqa: F401
    CachedDepth,
    CachedSegmenter,
    HttpPointDepth,
    HttpSegmenter,
    NativeDepth,
)
