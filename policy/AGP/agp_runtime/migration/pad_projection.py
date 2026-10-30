"""Selected-point projection against robot pad geometry; no object/contact inference."""
import numpy as np
from migration.robot_geometry import PAD_ZERO

def _pad_projection(point_tcp, pad_vertices_tcp, numerical_epsilon_m=1e-12):
    """Convex nominal pad only. Signed distance positive inside, zero on edge."""
    point = np.asarray(point_tcp, dtype=float)
    vertices = np.asarray(pad_vertices_tcp, dtype=float)
    if point.shape != (3,) or vertices.shape != (4, 3):
        raise ValueError('Expected one point and four pad vertices')
    if not np.isfinite(point).all() or not np.isfinite(vertices).all():
        raise ValueError('Geometry must be finite')
    if not np.isfinite(numerical_epsilon_m) or numerical_epsilon_m < 0:
        raise ValueError('Invalid numerical epsilon')
    if np.ptp(vertices[:, 0]) > 1e-10:
        raise ValueError('Expected nominal pad plane perpendicular to TCP X')
    polygon = vertices[:, 1:]
    center = polygon.mean(axis=0)
    polygon = polygon[np.argsort(np.arctan2(polygon[:, 1]-center[1], polygon[:, 0]-center[0]))]
    edges = np.roll(polygon, -1, axis=0)-polygon
    lengths = np.linalg.norm(edges, axis=1)
    turns = edges[:, 0]*np.roll(edges, -1, axis=0)[:, 1]-edges[:, 1]*np.roll(edges, -1, axis=0)[:, 0]
    if np.any(lengths <= 1e-12) or np.any(turns <= 1e-16):
        raise ValueError('Expected nondegenerate strictly convex pad')
    delta = point[1:]-polygon
    fractions = np.clip(np.sum(delta*edges, axis=1)/(lengths**2), 0., 1.)
    distances = np.linalg.norm(delta-fractions[:, None]*edges, axis=1)
    side_distances = (edges[:, 0]*delta[:, 1]-edges[:, 1]*delta[:, 0])/lengths
    minimum = float(distances.min())
    inside = bool(np.all(side_distances >= 0))
    signed = minimum if inside else -minimum
    relation = 'boundary' if minimum <= numerical_epsilon_m else ('inside' if inside else 'outside')
    if relation == 'boundary':
        signed = 0.
    z_min, z_max = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    return {'projection_relation': relation, 'signed_edge_distance_m': signed,
            'axial_z_interval_m': [z_min, z_max],
            'signed_axial_margin_m': min(float(point[2])-z_min, z_max-float(point[2]))}



def selected_point_pad_projection(point_tcp, point_source):
    if point_source not in {'assumed_plane_intersection'}:
        raise ValueError('Unknown point source')
    return {'point_source':point_source,
            'pads':{'link'+side:_pad_projection(point_tcp,vertices) for side,vertices in PAD_ZERO.items()},
            'numerical_epsilon_m':1e-12,
            'scope':'Selected point projected onto nominal pad YZ boundaries. Ignores closing-axis X and does not establish whole-object overlap, contact, opposing surfaces or possession. Numerical epsilon is not physical uncertainty.'}
