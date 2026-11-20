"""RoboDojo adapter without the estimate capability.

A measurement is agp_core.rgb_measure.measure: RGB images, the public camera model and the height matcher, no depth file.
"""
from migration.adapter import RoboDojoAdapter


class OursAdapter(RoboDojoAdapter):
    supports_estimate = False

    def measure(self, capture, query):
        from agp_core import rgb_measure
        return rgb_measure.measure(self.session, capture, query)
