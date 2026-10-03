# XBrain-v1 release 0.1.0

This is the current reproducible release candidate for RoboDojo real-robot
policy integration. It binds the inference runtime to the three checkpoint
resource contracts recorded in `release_manifest.json`.

The release does not prevent future checkpoint changes. A different checkpoint
must be published as a new release after its `config.json`,
`inference_config.json`, norm statistics, tokenizer compatibility, and
`reset/update_obs/get_action` behavior have been revalidated.

Before publishing this release, replace logical resource identifiers with the
public checkpoint and tokenizer download locations. Do not put private local
filesystem paths in the public release.
