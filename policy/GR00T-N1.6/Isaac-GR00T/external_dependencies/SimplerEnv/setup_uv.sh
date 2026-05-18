rm -rf /mnt/amlfs-01/shared/sim-resources/SimplerEnv_uv
mkdir -p /mnt/amlfs-01/shared/sim-resources/SimplerEnv_uv
uv init /mnt/amlfs-01/shared/sim-resources/SimplerEnv_uv --python 3.10
uv add --project /mnt/amlfs-01/shared/sim-resources/SimplerEnv_uv --editable /mnt/amlfs-01/shared/sim-resources/SimplerEnv/ManiSkill2_real2sim
uv add --project /mnt/amlfs-01/shared/sim-resources/SimplerEnv_uv --editable /mnt/amlfs-01/shared/sim-resources/SimplerEnv
uv add --project /mnt/amlfs-01/shared/sim-resources/SimplerEnv_uv json-numpy
uv add --project /mnt/amlfs-01/shared/sim-resources/SimplerEnv_uv numpy==1.26.4 ray==2.48.0 gymnasium==0.29.1
uv add --project /mnt/amlfs-01/shared/sim-resources/SimplerEnv_uv --upgrade setuptools

