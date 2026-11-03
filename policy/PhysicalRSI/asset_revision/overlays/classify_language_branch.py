"""Public X5 numerical continuation for the two parallel-jaw grasp symmetries.

Candidate qualification only: official EE IK still owns the executed branch.
"""
from pathlib import Path
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


class BranchSelector:
    def __init__(self):
        self.model = pin.buildModelFromUrdf(str(Path(__file__).parent/'assets/X5A.urdf'))
        self.data = self.model.createData()
        self.frame = self.model.getFrameId('link6')
        self.observed_bases = {}
        # URDF has +/-10 for some revolute axes; the observed EE solver switches
        # the principal wrist branch at +/-pi. This is a branch, not a joint limit.
        self.lo = np.maximum(self.model.lowerPositionLimit[:6], -np.pi)
        self.hi = np.minimum(self.model.upperPositionLimit[:6], np.pi)

    def fk(self, q, arm):
        full = pin.neutral(self.model); full[:6] = q
        pin.framesForwardKinematics(self.model, self.data, full)
        if arm not in self.observed_bases:
            raise ValueError('Bind current robot observation before branch kinematics')
        return self.observed_bases[arm] * self.data.oMf[self.frame]

    def bind_state(self, state):
        bases = {}
        for arm in ('left', 'right'):
            joints = np.asarray(state[arm+'_arm_joint_state'], dtype=float)
            observed = np.asarray(state[arm+'_ee_pose'], dtype=float)
            if joints.shape != (6,) or observed.shape != (7,) or not (
                    np.isfinite(joints).all() and np.isfinite(observed).all()):
                raise ValueError('Finite current robot joints and world EE pose required')
            full = pin.neutral(self.model); full[:6] = joints
            pin.framesForwardKinematics(self.model, self.data, full)
            world_ee = pin.SE3(Rotation.from_quat(observed[[4,5,6,3]]).as_matrix(), observed[:3])
            bases[arm] = world_ee * self.data.oMf[self.frame].inverse()
        self.observed_bases = bases
        # A cached solve is valid only for its original robot observation.
        self.solve_cache = None

    def begin_cache(self):
        self.solve_cache={};self.cache_hits=0;self.cache_misses=0

    def end_cache(self):
        stats={'hits':self.cache_hits,'misses':self.cache_misses,'scope':'exact same observation planning call'}
        self.solve_cache=None
        return stats

    def solve(self, arm, pose, previous):
        cache=getattr(self,'solve_cache',None)
        key=(arm,tuple(np.asarray(pose,float)),tuple(np.asarray(previous,float)))
        if cache is not None and key in cache:
            self.cache_hits+=1
            q,pe,ae=cache[key]
            return q.copy(),pe,ae
        q,pe,ae=self._solve_uncached(arm,pose,previous)
        if cache is not None:
            self.cache_misses+=1;cache[key]=(q.copy(),pe,ae)
        return q,pe,ae

    def _solve_uncached(self, arm, pose, previous):
        target_rotation = Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix()
        def residual(q):
            actual = self.fk(q, arm)
            return np.r_[actual.translation-pose[:3], pin.log3(actual.rotation.T@target_rotation)]
        initial = np.asarray(previous, float).copy()
        initial[np.abs(initial)<1e-6] = 0.  # numerical home noise must not set a 1e-12 trust radius
        result = least_squares(residual, np.clip(initial, self.lo+1e-7, self.hi-1e-7),
                               bounds=(self.lo,self.hi), max_nfev=100,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        r = residual(result.x)
        return result.x, float(np.linalg.norm(r[:3])), float(np.linalg.norm(r[3:]))

    def select(self, arm, plan, state):
        original = Rotation.from_quat(np.asarray(plan[0]['pose'])[[4,5,6,3]])
        records=[]
        for symmetry in (0, 1):
            rotation = original * Rotation.from_rotvec([symmetry*np.pi, 0., 0.])
            xyzw=rotation.as_quat(); quat=xyzw[[3,0,1,2]].tolist()
            previous = np.asarray(state[arm+'_arm_joint_state'],float)
            stations=[]; valid=True; last_point=None; last_rotation=None
            # Both exact same grasp position and closure line; fingers exchange sides.
            for segment in plan:
                if segment['phase'] in ('close','release','retreat'): continue
                end=np.asarray(segment['pose'][:3],float)
                destination_rotation=Rotation.from_quat(np.asarray(segment['pose'])[[4,5,6,3]]) * Rotation.from_rotvec([symmetry*np.pi,0.,0.])
                fractions=np.linspace(0.,1.,{'approach':14,'descend':14,'lift':12,'transport':20,'lower':10}[segment['phase']]+1)[1:]
                fractions=fractions*fractions*(3-2*fractions)
                points=[end] if last_point is None else [last_point+(end-last_point)*a for a in fractions]
                for index,point in enumerate(points):
                    rr=destination_rotation if last_rotation is None else Slerp([0,1],Rotation.concatenate([last_rotation,destination_rotation]))(float(fractions[index]))
                    qq=rr.as_quat()[[3,0,1,2]]
                    q,pe,ae=self.solve(arm,np.r_[point,qq],previous)
                    jump=float(np.max(np.abs(q-previous))) if stations else 0.
                    margin=float(np.min(np.minimum(q-self.lo,self.hi-q)))
                    passed=pe<.002 and ae<.03 and margin>.15 and jump<.35
                    stations.append({'phase':segment['phase'],'point':point.tolist(),'q':q.tolist(),
                                     'position_error_m':pe,'rotation_error_rad':ae,
                                     'margin_rad':margin,'jump_rad':jump,'passed':bool(passed)})
                    previous=q
                    if not passed:
                        valid=False; break
                if not valid: break
                last_point=end;last_rotation=destination_rotation
            records.append({'symmetry':symmetry,'quat':quat,'valid':valid,
                            'carry_quat':(Rotation.from_quat(np.asarray(plan[3]['pose'])[[4,5,6,3]]) * Rotation.from_rotvec([symmetry*np.pi,0.,0.])).as_quat()[[3,0,1,2]].tolist(),
                            'minimum_margin_rad':min(s['margin_rad'] for s in stations),
                            'stations':stations})
        eligible=[r for r in records if r['valid']]
        # Preserve an already admissible registered grasp; rotate only the unsafe branch.
        chosen=min(eligible,key=lambda r:r['symmetry']) if eligible else None
        return chosen, records
