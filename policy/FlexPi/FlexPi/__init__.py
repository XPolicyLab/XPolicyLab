# Vendored flex-pi (https://github.com/geyan21/flex-pi), core subset:
# src/ configs/ experiments/robotwin/ scripts/. third_party/, docs/, assets/
# and the libero/yam experiments are omitted — nothing on the RoboDojo eval
# path imports them.
#
# This file only marks the directory as a package. The adapter reaches the
# upstream policy through experiments.robotwin.flexpi_policy.deploy_policy,
# which it imports after putting this directory and its src/ on sys.path.
