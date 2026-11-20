"""Public policy views; the evaluator owns the underlying environment and clock.

These are engineering interfaces in the same trusted Unix process, not a
security sandbox against introspection or malicious Python.
"""
from types import MappingProxyType
import numpy as np



class PolicyActionClock:
    __slots__ = ('__clock', 'interval', 'frequency')

    def __init__(self, bridge):
        self.__clock = bridge
        self.interval = bridge.interval
        self.frequency = bridge.frequency

    def execute(self, controls, phase):
        return self.__clock.execute(controls, phase)

    def execute_normalized(self, command, phase):
        return self.__clock.execute_normalized(command, phase)

    @property
    def time_scale(self):
        return self.__clock.time_scale

    @time_scale.setter
    def time_scale(self,value):
        value=float(value)
        if not np.isfinite(value) or value<=0:raise ValueError('Motion time scale must be positive and finite')
        self.__clock.time_scale=value

    @property
    def after_action(self):
        return self.__clock.after_action

    @after_action.setter
    def after_action(self, callback):
        if callback is not None and not callable(callback):
            raise TypeError('after_action must be callable or None')
        self.__clock.after_action = callback

    @property
    def action_count(self):
        return len(self.__clock.normalized_actions)

    @property
    def normalized_actions(self):
        """Own issued-command history used by the mature retreat/recovery path."""
        return tuple(MappingProxyType({key:tuple(value) for key,value in action.items()})
                     for action in self.__clock.normalized_actions)


class PublicRGBRecorder:
    __slots__ = ('__recorder', '__frames')

    def __init__(self, recorder):
        self.__recorder = recorder
        self.__frames = {}

    @property
    def images(self):
        frames = {}
        for name, values in self.__recorder.images.items():
            copies = []
            cached = self.__frames.setdefault(name, [])
            for index, value in enumerate(values):
                if index < len(cached) and cached[index][0] is value:
                    image = cached[index][1]
                else:
                    image = np.array(value, copy=True)
                    image.setflags(write=False)
                    if index < len(cached):
                        cached[index] = (value, image)
                    else:
                        cached.append((value, image))
                copies.append(image)
            del cached[len(values):]
            frames[name] = tuple(copies)
        return MappingProxyType(frames)

    @property
    def output_root(self):
        return self.__recorder.output_root

    @property
    def _sim_step(self):
        return self.__recorder._sim_step

    @property
    def frame_interval(self):
        return self.__recorder.frame_interval

    def _state(self,name):
        return tuple(np.array(value,copy=True) for value in getattr(self.__recorder,name))

    @property
    def state_left_arm(self):return self._state('state_left_arm')

    @property
    def state_right_arm(self):return self._state('state_right_arm')

    @property
    def state_left_ee(self):return self._state('state_left_ee')

    @property
    def state_right_ee(self):return self._state('state_right_ee')

    @property
    def state_left_gripper(self):return self._state('state_left_gripper')

    @property
    def state_right_gripper(self):return self._state('state_right_gripper')

    @property
    def phases(self):
        # RGB histories include the reset observation at index zero.
        return ('reset', *self.__recorder.phases)
