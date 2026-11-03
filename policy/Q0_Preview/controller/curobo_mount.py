"""Allow the fixed robot base's mounting contact with the table, only in world queries.

CuRobo's fitted base spheres extend below the mounting plane. Disabling the link
would also remove self-collision protection. Instead, each world query owns a copy;
the original kinematics spheres used by self-collision remain unchanged.
"""


class TableMountCollisionFilter:
    def __init__(self, checker, base_indices):
        if not base_indices.numel():
            raise ValueError('Fixed base has no collision spheres')
        self._checker = checker
        self._base_indices = base_indices.clone()
        # Resolve static indices before CUDA capture; scalar fills avoid the
        # tensor advanced-index assignment path inside the captured graph.
        self._base_slots = tuple(int(i) for i in base_indices.detach().cpu().tolist())

    def __getattr__(self, name):
        return getattr(self._checker, name)

    def _query(self, method, query_sphere, *args, **kwargs):
        spheres = query_sphere.clone()
        # CuRobo convention: a negative radius disables this sphere in the query.
        for index in self._base_slots:
            spheres[..., index, 3].fill_(-100.)
        return getattr(self._checker, method)(*args, query_sphere=spheres, **kwargs)

    def get_sphere_distance(self, query_sphere, *args, **kwargs):
        return self._query('get_sphere_distance', query_sphere, *args, **kwargs)

    def get_sphere_collision(self, query_sphere, *args, **kwargs):
        return self._query('get_sphere_collision', query_sphere, *args, **kwargs)

    def get_swept_sphere_distance(self, query_sphere, *args, **kwargs):
        return self._query('get_swept_sphere_distance', query_sphere, *args, **kwargs)

    def get_swept_sphere_collision(self, query_sphere, *args, **kwargs):
        return self._query('get_swept_sphere_collision', query_sphere, *args, **kwargs)


def allow_fixed_table_mount(scene_collision, kinematics, scene_model):
    """Install before warmup/CUDA capture; only valid for this table-only scene."""
    if set(scene_model) != {'cuboid'} or set(scene_model['cuboid']) != {'table'}:
        raise ValueError('Table mount exemption requires a table-only collision scene')
    if isinstance(scene_collision.checker, TableMountCollisionFilter):
        return
    indices = kinematics.config.kinematics_config.get_sphere_index_from_link_name('base_link')
    scene_collision.checker = TableMountCollisionFilter(scene_collision.checker, indices)
