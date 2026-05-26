from groot.vla.data.schema import EmbodimentTag

RAW_HDF5_FILENAME = "state_action.hdf5"
RAW_METADATA_FILENAME = "metadata.json"
RAW_VIDEO_FILENAME = "ego_view.mp4"
RAW_ANNOTATION_FILENAME = "annotation.json"

TRAINABLE_HDF5_FILENAME = "dataset.hdf5"
TRAINABLE_METADATA_FILENAME = "metadata.json"

RAW_DATA_CONTROL_FREQUENCY = 20

INITIAL_ACTIONS_FILENAME = "initial_actions.npz"

EMBODIMENT_TAG_TO_ANNOTATED_VERSION = {
    EmbodimentTag.REAL_GR1_ARMS_ONLY: EmbodimentTag.REAL_GR1_ARMS_ONLY_ANNOTATED,
    EmbodimentTag.REAL_GR1_ARMS_WAIST: EmbodimentTag.REAL_GR1_ARMS_WAIST_ANNOTATED,
    # Special for 5DC-S
    EmbodimentTag.ROBOCASA_GR1_ARMS_ONLY_FOURIER_HANDS: EmbodimentTag.REAL_GR1_ARMS_ONLY_ANNOTATED,
    EmbodimentTag.ROBOCASA_GR1_ARMS_WAIST_FOURIER_HANDS: EmbodimentTag.REAL_GR1_ARMS_WAIST_ANNOTATED,
}

# For chopped data
EPISODE_LENGTH_FILENAME = "episode_length.json"

# Processed
PROCESSED_VIDEO_FILENAME = "ego_view_pad_res224_freq20.mp4"
