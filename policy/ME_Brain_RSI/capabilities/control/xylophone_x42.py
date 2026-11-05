"""X42: horizontal stroke route first; tilted hexashell routes only as fallback.

Zero-friction mallet: with the -60 deg hexashell tilt gravity drags the shaft
through the pinch and the head sinks during the sequence (contact plateaus from
key 3-5, sometimes from key 0). A horizontal hold has no axial gravity component.
X42 therefore plans the endpoint-IK and clearance checked horizontal route (the
X34 vertical-stroke functional route, which keeps the post-grasp orientation)
before any tilted candidate. With a horizontal shaft both beat points sit at the
head-centre height, so the head is placed at .806 m: inside the beat band
[hit_z+.010, hit_z+.036] = [.789,.815] with 8 mm between the head bottom and the
key top. Tilted routes and the X41 contact tracking remain the fallback.
"""
from .xylophone_x41 import XylophonePolicy as X41

HORIZONTAL_HIT_Z=.806


class XylophonePolicy(X41):
    FALLBACK_HIT_Z=HORIZONTAL_HIT_Z

    def reset(self):
        super().reset()
        self.program_notes.update(candidate='X42_horizontal_route_first')

    def _choose_strokes(self,side,rotation,anchor,prefix=(),reserve=60):
        route=self._endpoint_checked_strokes(side,anchor,prefix)
        if route is not None:
            self.program_notes['x42_route']=dict(selected='horizontal_endpoint_route',hit_head_z=HORIZONTAL_HIT_Z)
            return route,dict(degrees=0.,heading_degrees=0.,orientation='x42_horizontal_endpoint_route')
        self.program_notes['x42_route']=dict(selected='tilted_fallback',reason=(self.program_notes.get('x34_fallback_route') or {}).get('reason'))
        return super()._choose_strokes(side,rotation,anchor,prefix,reserve)
