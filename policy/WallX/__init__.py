"""XPolicyLab policy adapter for Wall-X checkpoints.

``model.py`` is the server side (loads the checkpoint, answers RPCs) and
``deploy.py`` is the client side (drives an episode). The split mirrors every
other policy in this repository; what is specific to Wall-X is that the
checkpoint keeps a keyframe memory, whose bookkeeping lives on the *client* and
travels with each request as ``kfmem_client_state_v1``.
"""
