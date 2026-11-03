"""Project relay candidates onto support measured in the current observation."""
import numpy as np
from . import Camera, GeometryUnavailable
from .semantic_support import observe_semantic_plane
from .camera_projection import world_to_pixel_on_height


class SupportProjection:
    def __init__(self, camera, plane, visible):
        self.camera = camera
        self.plane = np.asarray(plane, float).copy()
        self.visible = np.asarray(visible, bool).copy()
        if self.plane.shape != (3,) or not np.isfinite(self.plane).all():
            raise GeometryUnavailable('Measured finite support plane required')
        if self.visible.shape != camera.image.shape[:2] or not self.visible.any():
            raise GeometryUnavailable('Current observed support extent required')

    def height(self, x, y):
        if not np.isfinite([x,y]).all():
            raise GeometryUnavailable('Finite support query required')
        a,b,c = self.plane
        return float(a*x+b*y+c)

    def project(self, xy):
        x,y = xy
        return world_to_pixel_on_height(xy, world_z=self.height(x,y), camera=self.camera)

    def unproject(self, pixel):
        uv = np.asarray(pixel, float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise GeometryUnavailable('Finite support pixel required')
        u,v = np.rint(uv).astype(int)
        h,w = self.visible.shape
        if not (0<=u<w and 0<=v<h and self.visible[v,u]):
            raise GeometryUnavailable('Relay target is outside observed support')
        origin = self.camera.transform[:3,3]
        ray = self.camera.unproject([uv],[1.])[0]-origin
        a,b,c = self.plane
        denominator = ray[2]-a*ray[0]-b*ray[1]
        if abs(denominator)<1e-9:
            raise GeometryUnavailable('Relay ray parallel to support')
        distance = (c-origin[2]+a*origin[0]+b*origin[1])/denominator
        if distance<=0:
            raise GeometryUnavailable('Relay support behind camera')
        return tuple(float(value) for value in (origin+distance*ray)[:2])


def observed_language_support(episode):
    cached = getattr(episode, '_observed_relay_support', None)
    if cached is not None and cached[0] is episode.obs:
        return cached[1]
    prior = prior_language_support_plane(episode)
    kwargs = {} if prior is None else {'prior_plane': prior}
    plane,visible = observe_semantic_plane(episode.obs, episode.inventory.sam.segment_many, **kwargs)
    support = SupportProjection(Camera.from_observation(episode.obs), plane, visible)
    episode._observed_relay_support = (episode.obs, support)
    return support


def prior_language_support_plane(episode):
    """Return a private copy of this episode's previous measurement as a seed."""
    cached = getattr(episode, '_observed_relay_support', None)
    return None if cached is None else cached[1].plane.copy()


def cache_language_support(episode, camera, plane, visible):
    """Reuse this observation's already validated depth-support measurement."""
    support = SupportProjection(camera, plane, visible)
    episode._observed_relay_support = (episode.obs, support)


def observed_contact_height(surface_z, support_z, retry_adjustment=0.):
    """Keep public X5 fingertip clearance relative to the measured support."""
    if not np.isfinite([surface_z,support_z,retry_adjustment]).all():
        raise GeometryUnavailable('Finite measured grasp and support heights required')
    # Public X5 joint7/8 origin X plus the finger STL maximum X.
    fingertip_distance = .15757000232458113
    # Original generic grasp inset from the fingertip, not an object dimension.
    contact_distance = fingertip_distance - .025
    floor = support_z + fingertip_distance + .003
    return float(max(floor, max(floor, surface_z + contact_distance - .008) + retry_adjustment))
