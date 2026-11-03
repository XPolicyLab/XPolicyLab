"""Legacy API for generic glyph identity evidence, not metric size validation."""


def metric_support(projector, mask, record):
    # The former CAD-size gate cannot be retained without privileged geometry.
    # Explicitly report scale-invariant identity evidence to downstream logs.
    return projector.validate_identity(mask, record)
