# Source and artifact provenance

WorldScape Policy combines original policy, conditioning, memory, checkpoint,
and evaluation interfaces with compatibility code derived from upstream
open-source research implementations. Required copyright and attribution
information is centralized in `NOTICE` and retained source headers.

## Source lineage

- **Legacy checkpoint compatibility:** versioned key mappings and conversion
  utilities are retained only for importing older checkpoint layouts.
- **Wan:** video-backbone modules retain their upstream notices in source files.
- **Other incorporated utilities:** source-level notices for DeepMind tree
  utilities and termcolor are retained beside those files.
- **WorldScape additions:** the `worldscape_policy` package, launchers, native
  checkpoint format, evaluation adapters, and release documentation.

This document supplements, and does not replace, `LICENSE`, `NOTICE`, source
headers, or third-party license files.

## Checkpoints and datasets

No model weights or datasets are conveyed merely by installing this Python
package. Referenced Hugging Face repositories, Wan base models, robot datasets,
and simulator assets are independently distributed artifacts. Users must
review their access conditions, licenses, and intended-use restrictions.

Native checkpoint conversion records the source checkpoint hash and source
revision when callers provide them. Preserve the generated
`checkpoint_manifest.json` with redistributed converted artifacts.

## Reproducibility record

For a release or experiment, record:

1. the WorldScape Policy version and source commit;
2. the source checkpoint repository, revision, and checksum;
3. dataset names, versions, filtering, and conversion commands;
4. selected dependency extra and resolved environment;
5. hardware, CUDA, and simulator versions;
6. launch command, configuration overrides, and random seeds.

The repository cannot attest to the provenance of locally supplied weights,
data, simulator assets, or converted checkpoints.
