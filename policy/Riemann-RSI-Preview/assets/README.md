# Runtime assets

This directory contains four small runtime inputs (about 1.25 MB), grouped by
purpose. Model weights, environment installations and evaluation outputs are not
stored here.

| Location | Used for |
| --- | --- |
| `robots/x5/X5A.urdf` | Public own-robot kinematics and camera mounting geometry |
| `robots/x5/curobo.yml` | The same robot's numerical planner/collision configuration |
| `perception/garments/geometry.npz` | Public garment template vertices and topology |
| `perception/garments/metadata.json` | Matching public local landmarks, units and source identities |

The garment templates are static shape priors; current pose and correspondence
come from public RGB. They are not saved episode layouts or measured scene state.

Model/dependency declarations live in `configs/models/`. Public resource locations
use `configs/resource-roots.json`; `RIEMANN_RESOURCE_ROOTS` explicitly relocates
external model/asset roots. SHA checks remain in the corresponding resource
contracts, rather than in filenames.

The original supplied MIT notice is consolidated in the
[third-party documentation](../docs/third-party-provenance.md#original-mit-notice).
See [source attribution](../docs/third-party-provenance.md) for provenance scope.

External model mirrors and the single deduplicated template cache are downloaded
with `download_checkpoints.sh --prepare`. The installer derives the required
official static asset subset from current task declarations and copies verified
ordinary files into the external `official_assets/` cache. This supports the
official installer's linked geometry without adding robot meshes to this source
repository or changing actor-side content checks. Fixed calibration/defaults for
insert-key, coin and general pickup live in the corresponding task skills.
