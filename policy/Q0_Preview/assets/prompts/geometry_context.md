## Fixed robot geometry

Offsets from the grasp reference, in local `(tool, jaw, tool cross jaw)` axes and metres:
- front-tip midpoint: `[0.01257, -0.000002, -0.00014012]`;
- distal-pad bounding-box midpoint: `[0.00507, -0.000002, -0.00144044]`.
Use `geom.grasp_reference_for` with `region='front_tip'` or
`region='distal_pad_bbox_center'` for the intended working region. These are fixed
geometric references; actual pad contact depends on object shape and alignment.
The front edge is about 10 mm wide and 1.5 mm thick; the rear finger structure is wider.
Full-open inner gap is about 88.8 mm; empty closed jaws can retain about 0.8 mm.

The fixed table plane is world `z=0.765` m. An object's upper surface is above it
by the object's thickness. Pixel scale depends on depth; use calibrated projection
or plane-scale helpers for the actual working point. For triangulation, match the
same identifiable feature and check reprojection; a small ray residual alone does
not establish a correct match.

Detailed dimensions are in `/work/geometry_reference.md`; reference measurements are in
`/work/geometry_evidence.md`. Historical tracking errors and settling times are
examples, not operating bounds. Robot pose accuracy does not establish object pose
or contact-point accuracy.
