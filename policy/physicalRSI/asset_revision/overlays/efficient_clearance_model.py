"""Unqualified development: reduce repeated-approach cost in measured clearance.
First contact stays at1mm. Repeated approaches use only the current episode's
proprioceptively adapted surface, never stored benchmark object coordinates.
"""
import copy
from .clearance_press_model import Model as ClearanceModel
from .bounded_contact_press_model import bounded_action
from rdj_master.tasks.press_by_number import cap as baseline

def descent_step(current_z,target_z,initial_surface_z,current_surface_z):
    adapted=initial_surface_z is not None and abs(current_surface_z-initial_surface_z)>.003
    gap=current_z-target_z
    if not adapted:return .001
    if gap>.015:return .005
    return .003 if gap>.008 else .001

class Model(ClearanceModel):
    def get_action_batch(self,env_idx_list=None):
        requested=self._observed_indices if env_idx_list is None else tuple(env_idx_list)
        # Every inner contact/clearance wrapper requires an initialized slot.
        indices=[index for index in requested if index in self._core._slots]
        actions=super().get_action_batch(indices) if indices else []
        for i,env_idx in enumerate(indices):
            slot=self._core._slots[env_idx];label=slot.session.current_label
            if label is None or slot.session.phase not in {'down','retry_down'}:continue
            arm=baseline.BUTTON_ARM_ASSIGNMENTS[label];key=arm+'_ee_pose';anchor=slot.anchors[label];target=slot.session._down_target()
            learned=(env_idx,label) in getattr(self._core,'_contact_endpoints',{})
            gap=slot.latest_robot[key][2]-target[2]
            step=(.005 if gap>.015 else .003 if gap>.008 else .001) if learned else .001
            action=actions[i][0];action[key][2]=target[2]
            actions[i][0]=bounded_action(action,slot.latest_robot,arm,step_m=step)
        active=dict(zip(indices,actions,strict=True))
        return [active[index] if index in active else
                [copy.deepcopy(self._localization_wait[index]['robot'])]
                for index in requested]
