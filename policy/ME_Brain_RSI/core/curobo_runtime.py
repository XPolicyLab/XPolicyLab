"""Check the shared CUDA and CuRobo v2 runtime before creating a solver."""
from importlib.metadata import version


def require_curobo(*, check_gpu=True):
    """Verify the installed CuRobo API and, on a policy GPU, CUDA availability."""
    try:
        import torch
        from cuda.core import Device, Program, ProgramOptions
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f'CuRobo v2 CUDA dependencies are unavailable: {error}. Run bash install.sh curobo.'
        ) from error
    if torch.version.cuda is None:
        raise RuntimeError('CuRobo tasks require CUDA-enabled PyTorch. Run bash install.sh curobo.')
    if check_gpu and not torch.cuda.is_available():
        raise RuntimeError('CuRobo tasks require a visible NVIDIA GPU. Check CUDA_VISIBLE_DEVICES and the driver.')
    if not callable(InverseKinematicsCfg.create) or not callable(InverseKinematics.solve_pose):
        raise RuntimeError('The installed CuRobo is missing the v2 IK API. Run bash install.sh curobo.')
    if not all((Device, Program, ProgramOptions, DeviceCfg, GoalToolPose, JointState, Pose)):
        raise RuntimeError('The installed CuRobo or cuda-core is missing required API symbols.')
    return version('nvidia-curobo')


if __name__ == '__main__':
    print(f'CuRobo v2 imports available (nvidia-curobo {require_curobo(check_gpu=False)})')
