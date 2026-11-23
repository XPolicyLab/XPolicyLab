"""Empirical defaults from directed self-play.

Each value matches docs/self_play_estimates/reports/manifest.json.
These are task-scoped operating choices, not universal bounds.
"""

# organize_table.footprint_grid
FOOTPRINT_CELL_M = 0.004

# organize_table.footprint_elongation
FOOTPRINT_ELONGATION = 1.5

# organize_table.planar_center_agreement
PLANAR_CENTER_AGREEMENT_M = 0.004

# organize_table.planar_extent_quantiles
PLANAR_EXTENT_QUANTILES = (0.01, 0.99)

# sweep_blocks.pan_curve_bound
PAN_CURVE_UPPER = 2.5

# sweep_blocks.sweep_cube_plane
SWEEP_CUBE_PLANE_M = 0.0096

# sweep_blocks.ribbon_simplification
SWEEP_RIBBON_EPSILON_PX = 1.5

# sweep_blocks.pan_projection_plane
PAN_PROJECTION_PLANE_M = 0.035

# sweep_blocks.pan_size_bounds
PAN_SIZE_INITIAL_BOUNDS_M = (0.11625, 0.22225)

# sweep_blocks.sweep_initial_dimensions
SWEEP_INITIAL_RATIOS = {'pan': (1.100308, 0.23813), 'brush_head': (0.449116, 0.318794)}

# sweep_blocks.sweep_initial_dimensions
SWEEP_ROD_DIAMETERS_M = (0.0252, 0.02512)

# sweep_blocks.sweep_initial_dimensions
SWEEP_BLOCK_SPAN_RATIO = 0.499413

# sweep_blocks.sweep_shape_bounds
SWEEP_INITIAL_CENTER_HEIGHTS_M = {'pan': (0.012973, 0.03072), 'rod': (0.008, 0.023), 'brush_head': (0.01046, 0.02646)}

# make_toast.toast_height_range, make_toast.toaster_slot_plane_height
TOASTER_SLOT_HEIGHT_M = 0.9254

# make_toast.toast_height_range
TOASTER_RIM_HEIGHT_BOUNDS_M = (0.918320534, 0.928830765)

# make_toast.toast_divider_bounds
TOASTER_DIVIDER_OFFSET_BOUNDS_M = (-0.0018, 0.0016)

# make_toast.toast_divider_bounds
TOASTER_DIVIDER_WIDTH_BOUNDS_M = (0.014437603, 0.0293)

# make_toast.toast_divider_bounds
TOASTER_DIVIDER_INITIAL_WIDTH_M = 0.026496508

# make_toast.bread_opening_clearance
BREAD_CLEARANCE_PER_SIDE_M = 0.0045

# make_toast.bread_release_opening
BREAD_RELEASE_OPENING = 0.38903991256786846

# insert_key.key_shaft_width
KEY_SHAFT_INITIAL_FLOOR_MM = 4.107446031

# insert_key.key_bow_join
KEY_BOW_JOIN_CLIP = 0.304508832

# insert_key.key_plane
KEY_PLANE_OFFSET_M = 0.004851724

# hang_mugs.mug_radius_bounds
MUG_RADIUS_INITIAL_BOUNDS_M = (0.0188, 0.06)

# hang_mugs.mug_height_seed
MUG_HEIGHT_SEEDS_M = (0.045, 0.068, 0.09)

# hang_mugs.mug_pregrasp
MUG_PREGRASP_HEIGHT_M = 0.08

# hang_mugs.mug_grasp_clearance
MUG_GRASP_CLEARANCE_M = 0.01

# hang_mugs.peg_radius_bounds
PEG_RADIUS_INITIAL_BOUNDS_M = (0.003, 0.007)

# hang_mugs.peg_axial_sampling
PEG_AXIAL_SAMPLE_FRACTIONS = (0.18, 0.9)

# hang_mugs.peg_crop_extent
PEG_CROP_UPPER_FRACTION = 1.08

# align_blocks.cube_iou
CUBE_MIN_IOU = 0.85

# align_blocks.ridge_inlier_distance
RIDGE_INLIER_DISTANCE_PX = 3.5

# align_blocks.ridge_quantiles
RIDGE_SEARCH_QUANTILES = (0.01, 0.99)

# align_blocks.wedge_scan_quantile
WEDGE_SCAN_QUANTILES = (0.06, 0.84)

# align_blocks.wedge_ridge_extent
WEDGE_RIDGE_FRACTIONS = (0.03, 0.9789)

# build_tower.shape_size_floor
SHAPE_SIZE_FLOOR_M = 0.0

# build_tower.shape_initial_dimensions
SLAB_INITIAL_WIDTH_RATIO = 0.2651

# build_tower.shape_initial_dimensions
SLAB_INITIAL_THICKNESS_M = 0.019455

# build_tower.shape_initial_dimensions
BOX_INITIAL_RATIOS = (0.618, 0.627)

# solve_equation.token_area_ratio
TOKEN_AREA_RATIO_BOUNDS = (0.9899803997695769, 1.0170663556939687)

# solve_equation.token_plane
TOKEN_PROJECTION_OFFSET_M = 0.006552497
