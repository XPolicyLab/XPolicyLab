"""Legacy import name for generic font identity matching; no CAD projection."""
from physicalrsi_observed_geometry.font_identity import FontDigitMatcher

# Existing controller call sites retain their API, but no renderer, camera
# constants, mesh loader, or metric object dimensions remain in this module.
ProjectedDigits = FontDigitMatcher
