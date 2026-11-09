import itertools
import heapq
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import quat_matrix, matrix_quat, bounded_pose
from .contracts import restrict_observation, validate_observation, validate_action
from .vision import HeadVision, WristVision, TABLE_Z, grasp_from_profile, plane
from . import wristcam
ARMS = ('left', 'right')
BASES = {'left': np.array([-0.3, -0.45]), 'right': np.array([0.3, -0.45])}
TCP = np.array([0.145, 0.0, 0.0])
STEP_LIMIT = 1050
GRIP_OPENING = 0.088
PAD_HALF = 0.0125
PLACE_GAP = 0.004
Z = np.array([0.0, 0.0, 1.0])
ARM_GAP = 0.22

def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-09)

def pose_from(R, p):
    return np.r_[np.asarray(p, float), matrix_quat(R)]

def vertical_rotation(phi):
    return Rotation.from_euler('z', phi).as_matrix() @ Rotation.from_euler('y', np.pi / 2).as_matrix()

def ee_from_tcp(R, tcp):
    return pose_from(R, np.asarray(tcp, float) - R @ TCP)

def tcp_of(pose):
    R = quat_matrix(pose[3:])
    return np.asarray(pose[:3]) + R @ TCP

def vertical_pose(xy, z_tcp, phi):
    return ee_from_tcp(vertical_rotation(phi), np.r_[np.asarray(xy, float), z_tcp])

def seg_dist(p, a, b):
    (p, a, b) = map(np.asarray, (p, a, b))
    d = b - a
    t = np.clip(np.dot(p - a, d) / max(np.dot(d, d), 1e-09), 0, 1)
    return float(np.linalg.norm(p - (a + t * d)))

class Policy:

    def __init__(self, task, calibration, vision=None):
        if task != 'sort_nesting_dolls_by_size':
            raise ValueError('Nesting candidate only')
        self.task = task
        self.calibration = calibration
        self.head = vision or HeadVision(calibration)
        self.wrist = WristVision()
        self.step = 0
        self.age = 0
        self.dwell = 0
        self.phase = 'observe'
        self.reason = 'initializing'
        self.finished = False
        self.home = None
        self.grips = {s: 1.0 for s in ARMS}
        self.motion = {}
        self.events = {}
        self.log = {}
        self.objects = None
        self.kind = None
        self.frames_to_save = []
        self.detections = []
        self.hover_z = TABLE_Z + 0.2
        self.clear_tcp = {}
        self.rest_tcp = {}
        self.grasp = {}
        self.survey_index = 0
        self.survey_age = 0
        self.survey_results = {}
        self.tasks = []
        self.plan_log = None
        self.detect_fail = 0
        self.arm = {}
        self.done = set()
        self.active = {}
        self.positions = {}

    def stop(self, reason):
        self.finished = True
        self.reason = reason

    def advance(self, phase):
        self.events.setdefault(phase, self.step)
        self.phase = phase
        self.age = 0
        self.dwell = 0

    def move(self, action, obs, side, goal, grip, translation=0.015, rotation=0.12, tol=0.008, rot_tol=0.1, dwell=2):
        current = obs['state'][side + '_ee_pose']
        err = float(np.linalg.norm(current[:3] - goal[:3]))
        ang = float(Rotation.from_matrix(quat_matrix(current[3:]).T @ quat_matrix(goal[3:])).magnitude())
        action[side + '_ee_pose'] = bounded_pose(current, goal, translation, rotation)
        action[side + '_ee_joint_state'] = np.array([grip])
        self.grips[side] = grip
        self.dwell = self.dwell + 1 if err < tol and ang < rot_tol else 0
        self.motion = dict(side=side, err=round(err, 4), ang=round(ang, 4), age=self.age, goal=np.asarray(goal).round(4).tolist(), ee=np.asarray(current).round(5).tolist())
        return self.dwell >= dwell

    def goto(self, action, obs, side, goal, grip=None, translation=0.02, rotation=0.15, tol=0.015):
        cur = obs['state'][side + '_ee_pose']
        action[side + '_ee_pose'] = bounded_pose(cur, goal, translation, rotation)
        g = self.grips[side] if grip is None else grip
        action[side + '_ee_joint_state'] = np.array([g])
        self.grips[side] = g
        ang = float(Rotation.from_matrix(quat_matrix(cur[3:]).T @ quat_matrix(goal[3:])).magnitude())
        return float(np.linalg.norm(cur[:3] - goal[:3])) < tol and ang < 0.1

    def park_goal(self, side, dz=0.0):
        goal = self.home[side].copy()
        goal[2] += dz
        return goal

    def rest(self, action, obs, side):
        return self.goto(action, obs, side, self.working_rest(side) if self.phase in ('execute','prepare_execute') else self.clear_goal(side), 1.0)

    def working_rest(self,side):
        y=(BASES[side][1]+float(np.mean([o['xy'][1] for o in self.objects])))/2
        return vertical_pose([BASES[side][0],y],self.hover_z,np.pi/2)

    def clear_goal(self, side):
        sign=-1 if side=='left' else 1
        xy=BASES[side]+np.array([sign*GRIP_OPENING*1.5,GRIP_OPENING/2])
        return vertical_pose(xy,self.home[side][2]+4*PAD_HALF,np.pi/2)

    def other(self, side):
        return 'right' if side == 'left' else 'left'

    def detect(self, obs, tag):
        rgb = obs['vision']['cam_head']['color']
        det = self.head.detect(rgb, {s: obs['state'][s + '_ee_pose'] for s in ARMS})
        self.log.setdefault('head', {})[tag] = dict(det, table=None)
        self.frames_to_save.append((tag, ('cam_head',)))
        if 'objects' in det:
            self.detections.append(det)
        else:
            self.detect_fail += 1
        return det

    def radius(self, rank):
        return self.objects[rank]['radius']

    def plan(self):
        xy = {o['rank']: np.asarray(o['xy'], float) for o in self.objects}
        radii = {r: self.radius(r) for r in xy}
        clearance = max(2 * PAD_HALF, GRIP_OPENING / 2) + max((o['uncertainty'] for o in self.objects))
        ordered = list(reversed(range(5)))
        slots = np.zeros(5)
        for i in range(1, 5):
            slots[i] = slots[i - 1] + radii[ordered[i - 1]] + radii[ordered[i]] + clearance
        slots -= np.mean(slots)
        center = np.mean(list(xy.values()), axis=0)
        workspace = np.asarray(self.calibration['workspace'])
        step = max(radii.values()) + clearance
        shared_points = [np.array([x, y]) for x in np.arange(workspace[0] + step, workspace[1] - step, step / 2) for y in np.arange(workspace[2] + step / 2, workspace[3], step / 2) if all(np.linalg.norm(np.array([x,y])-BASES[a]) < .42 for a in ARMS)]
        candidates = []
        for cy in np.arange(workspace[2]+step, workspace[3]-step, step/2):
            for cx in np.arange(workspace[0]-slots.min()+step/2, workspace[1]-slots.max()-step/2, step/2):
                targets = {r: np.array([cx + slots[i], cy]) for (i, r) in enumerate(ordered)}
                if any((not (workspace[0] + radii[r] < t[0] < workspace[1] - radii[r] and workspace[2] + radii[r] < t[1] < workspace[3] - radii[r]) for (r, t) in targets.items())):
                    continue
                arms = {r: 'left' if t[0] < 0 else 'right' for (r, t) in targets.items()}
                if any((np.linalg.norm(t - BASES[arms[r]]) > 0.43 for (r, t) in targets.items())):
                    continue
                if any((abs(t[0]) < PAD_HALF * 2 for t in targets.values())):
                    continue
                relay_required = any((xy[r][0] < 0) != (targets[r][0] < 0) for r in xy)
                if relay_required and not any(all(np.linalg.norm(h-q) >= max(radii.values()) + radii[j] + clearance for j,q in list(xy.items()) + list(targets.items())) for h in shared_points):
                    continue
                occupied = sum((np.linalg.norm(targets[r] - xy[j]) < radii[r] + radii[j] + clearance for r in xy for j in xy if r != j))
                distance = sum((np.linalg.norm(targets[r] - xy[r]) for r in xy))
                relays=sum((xy[r][0]<0)!=(targets[r][0]<0) for r in xy)
                candidates.append((occupied, relays, distance, targets, arms))
        if not candidates:
            raise ValueError('no observed collision-free row fits reachable table')
        (_, _, _, targets, arms) = min(candidates, key=lambda c: c[:3])
        xs = np.arange(workspace[0] + step, workspace[1] - step, step / 2)
        ys = np.arange(workspace[2] + step / 2, min(workspace[3], max((p[1] for p in xy.values())) + step), step / 2)
        free = [np.array([x, y]) for x in xs for y in ys]

        def margin(p, r, positions, goals=True):
            ds = [np.linalg.norm(p - q) - radii[r] - radii[j] for (j, q) in positions.items() if j != r]
            if goals:
                ds.extend((np.linalg.norm(p - q) - radii[r] - radii[j] for (j, q) in targets.items()))
            return min(ds) if ds else 1.0
        remaining = set(xy)
        self.tasks = []
        state = dict(xy)
        for _ in range(15):
            if not remaining:
                break
            choices = []
            for r in remaining:
                if all((np.linalg.norm(targets[r] - q) >= radii[r] + radii[j] + clearance - 1e-6 for (j, q) in state.items() if j != r)):
                    choices.append(r)
            if not choices:
                r = min(remaining, key=lambda j: np.linalg.norm(state[j] - targets[j]))
                side = 'left' if state[r][0] < 0 else 'right'
                spots = [p for p in free if (p[0] < 0) == (side == 'left') and np.linalg.norm(p - BASES[side]) < 0.42 and (margin(p, r, state) > clearance)]
                if not spots:
                    raise ValueError('no visible temporary placement region')
                p = min(spots, key=lambda p: np.linalg.norm(p - state[r]))
                self.tasks.append(dict(rank=r, side=side, pick=state[r].tolist(), place=p.tolist(), stage='temp'))
                state[r] = p
                continue
            r = min(choices, key=lambda j: np.linalg.norm(state[j] - targets[j]))
            src = 'left' if state[r][0] < 0 else 'right'
            dst = arms[r]
            if src != dst:
                both = [p for p in free if all((np.linalg.norm(p - BASES[a]) < 0.42 for a in ARMS)) and margin(p, r, state, goals=False) > clearance]
                if not both:
                    raise ValueError('no clear reachable handoff')
                h = min(both, key=lambda p: (max(np.linalg.norm(p-BASES[a]) for a in ARMS), np.linalg.norm(p-state[r])+np.linalg.norm(p-targets[r])))
                self.tasks.append(dict(rank=r, side=src, pick=state[r].tolist(), place=h.tolist(), stage='relay_out'))
                self.tasks.append(dict(rank=r, side=dst, pick=h.tolist(), place=targets[r].tolist(), stage='relay_in'))
            else:
                self.tasks.append(dict(rank=r, side=src, pick=state[r].tolist(), place=targets[r].tolist(), stage='direct'))
            state[r] = targets[r]
            remaining.remove(r)
        if remaining:
            raise ValueError('placement dependencies unresolved')
        for t in self.tasks:
            t.update(zg=self.objects[t['rank']]['grasp_height'], w=self.objects[t['rank']]['grasp_width'])
        self.plan_log = dict(targets={r: p.tolist() for (r, p) in targets.items()}, source='observed_dimensions_and_free_space')
        self.positions = {r: p.copy() for (r, p) in xy.items()}

    def transport_route(self, pick, place, rank, side):
        obstacles=[(np.asarray(p),self.radius(k)+self.radius(rank)+PAD_HALF) for k,p in self.positions.items() if k!=rank]
        def clear(a,b):
            return all(seg_dist(p,a,b)>rr for p,rr in obstacles)
        if clear(pick,place):return [np.asarray(place)]
        w=np.asarray(self.calibration['workspace']);resolution=2*PAD_HALF
        xs=np.arange(w[0],w[1]+resolution/2,resolution);ys=np.arange(w[2],w[3]+resolution/2,resolution)
        nodes=[np.asarray(pick),np.asarray(place)]
        nodes.extend(np.array([x,y]) for x in xs for y in ys if np.linalg.norm(np.array([x,y])-BASES[side])<.46 and all(np.linalg.norm(np.array([x,y])-p)>rr for p,rr in obstacles))
        points=np.asarray(nodes);cost={0:0.};prev={};queue=[(float(np.linalg.norm(place-pick)),0)]
        while queue:
            _,i=heapq.heappop(queue)
            if i==1:
                path=[]
                while i!=0:path.append(points[i]);i=prev[i]
                path=path[::-1];short=[];anchor=np.asarray(pick)
                while path:
                    n=max(j for j,q in enumerate(path) if clear(anchor,q))
                    anchor=path[n];short.append(anchor);path=path[n+1:]
                return short
            distances=np.linalg.norm(points-points[i],axis=1)
            adjacent=np.flatnonzero((distances<resolution*2.5)&(distances>1e-7))
            if clear(points[i],place):adjacent=np.unique(np.r_[adjacent,1])
            for j in adjacent:
                if not clear(points[i],points[j]):continue
                nc=cost[i]+float(distances[j])
                if nc<cost.get(int(j),float('inf')):
                    cost[int(j)]=nc;prev[int(j)]=i
                    heapq.heappush(queue,(nc+float(np.linalg.norm(points[j]-place)),int(j)))
        return None

    def prepare_task(self, t):
        r = t['rank']
        pick = np.asarray(t['pick'])
        place = np.asarray(t['place'])
        obj = self.objects[r]
        t['hover_z']=TABLE_Z+obj['height']+2*PAD_HALF+.005
        t['phi'] = obj['phi']
        offset=np.asarray(obj.get('grasp_offset',[0.,0.]))
        t['pick_tcp']=(pick+offset).tolist()
        t['place_tcp']=(place+offset).tolist()
        t['pre_open'] = min(1.0, (t['w'] + 2 * PAD_HALF) / GRIP_OPENING)
        top = max([self.objects[k]['height'] for (k, p) in self.positions.items() if k != r and seg_dist(p, pick, place) < self.radius(k) + self.radius(r) + 2 * PAD_HALF] + [0.0])
        route=self.transport_route(pick,place,r,t['side'])
        t['carry_points']=[(q+offset).tolist() for q in route] if route else [(place+offset).tolist()]
        t['carry_index']=0
        t['carry_z'] = TABLE_Z + t['zg'] + (2*PAD_HALF if route else top+2*PAD_HALF)
        t['grasp_z'] = TABLE_Z + t['zg']
        t['place_z'] = TABLE_Z + t['zg'] + PLACE_GAP
        return t

    def dependencies(self):
        n = len(self.tasks)

        def near(a, b, ra, rb):
            return np.linalg.norm(np.asarray(a) - np.asarray(b)) < self.radius(ra) + self.radius(rb) + 0.03
        for (i, t) in enumerate(self.tasks):
            deps = set()
            for k in range(i):
                u = self.tasks[k]
                if u['rank'] == t['rank']:
                    deps.add(k)
                    continue
                if near(u['pick'], t['place'], u['rank'], t['rank']) or near(u['place'], t['pick'], u['rank'], t['rank']) or near(u['pick'], t['pick'], u['rank'], t['rank']):
                    deps.add(k)
                if near(u['place'], t['place'], u['rank'], t['rank']):
                    deps.add(k)
                    movers = [m for m in range(k + 1, n) if self.tasks[m]['rank'] == u['rank']]
                    if movers:
                        deps.add(movers[0])
            t['deps'] = sorted((d for d in deps if d < i))
            t['waits_for_later'] = sorted((d for d in deps if d >= i))
        self.positions = {o['rank']: np.asarray(o['xy'], float) for o in self.objects}

    def conflicts(self, t, side, obs):
        o = self.other(side)
        pts = [np.asarray(t['pick']), np.asarray(t['place'])]
        ot = self.arm[o]['task']
        opts = [tcp_of(obs['state'][o + '_ee_pose'])[:2]] + ([np.asarray(ot['pick']), np.asarray(ot['place'])] if ot else [])
        for a in pts:
            for b in opts:
                if abs(a[0] - b[0]) < ARM_GAP and abs(a[1] - b[1]) < 0.3:
                    return True
        return False

    def schedule(self, side, obs):
        for (i, t) in enumerate(self.tasks):
            if i in self.done or i in self.active.values() or t['side'] != side:
                continue
            if any((k not in self.done for k in t['deps'] + t.get('waits_for_later', []))):
                return None
            if self.conflicts(t, side, obs):
                return None
            return i
        return None

    def act(self, raw):
        obs = restrict_observation(raw)
        validate_observation(obs)
        self.step += 1
        self.age += 1
        if self.home is None:
            self.home = {s: np.asarray(obs['state'][s + '_ee_pose'], float).copy() for s in ARMS}
            self.clear_tcp = {s: tcp_of(self.home[s]) for s in ARMS}
        action = {}
        for side in ARMS:
            action[side + '_ee_pose'] = np.asarray(obs['state'][side + '_ee_pose'], float).copy()
            action[side + '_ee_joint_state'] = np.array([self.grips[side]])
        if self.step >= STEP_LIMIT - 60 and self.phase not in ('home', 'park', 'settle', 'finished') and (not self.finished):
            self.events['budget_guard'] = self.step
            self.reason='operation_budget_exhausted'
            self.advance('home')
        try:
            self.run(action, obs)
        except Exception as exc:
            self.events.setdefault('exception', f'{type(exc).__name__}: {exc}'[:300])
            self.reason='perception_or_planning_error'
            self.advance('home')
        validate_action(action)
        return action

    def run(self, action, obs):
        p = self.phase
        if p == 'observe':
            self.advance('clear')
            return
        if p == 'clear':
            done = True
            for side in ARMS:
                goal = self.clear_goal(side)
                done = self.goto(action, obs, side, goal, 1.0) and done
            if done or self.age > 45:
                det = self.detect(obs, 'clear')
                if 'objects' not in det:
                    self.stop('head_detection_failed')
                    self.advance('home')
                    return
                self.kind = det['kind']
                self.objects = det['objects']
                self.hover_z = TABLE_Z + max((o['height'] for o in self.objects)) + 0.065
                self.advance('survey')
            return
        if p == 'survey':
            if self.survey_index >= len(self.objects):
                self.objects = sorted(self.survey_results.values(), key=lambda o: -o.get('volume',o['height']*o['radius']**2))
                for (rank, o) in enumerate(self.objects):
                    o['rank'] = rank
                self.head.objects = self.objects
                self.hover_z=TABLE_Z+max(o['height']+o['uncertainty'] for o in self.objects)+3*PAD_HALF
                self.plan()
                self.dependencies()
                self.arm = {s: dict(task=None, ti=None, phase='idle', age=0, dwell=0, motion={}) for s in ARMS}
                self.done = set()
                self.active = {}
                self.advance('prepare_execute')
                return
            obj = self.objects[self.survey_index]
            side = 'left' if obj['xy'][0] < 0 else 'right'
            self.rest(action, obs, self.other(side))
            survey_z=TABLE_Z+obj['height']+max(.035,2*obj['uncertainty'])+2*PAD_HALF
            goal = vertical_pose(obj['xy'], survey_z, np.pi / 2)
            if getattr(self,'survey_attempt',0):
                choices=[]
                for phi in [np.pi/2,0.,-np.pi/2,np.pi]:
                    g=vertical_pose(obj['xy'],survey_z,phi)
                    cam=type(self.head.cam)(dict(K=wristcam.K,camera_to_world_opengl=wristcam.camera_to_world(g),resolution=wristcam.RESOLUTION))
                    center=plane(cam,[[wristcam.RESOLUTION[0]/2,wristcam.RESOLUTION[1]/2]],TABLE_Z+obj['height']/2)[0,:2]
                    g[:2]+=np.asarray(obj['xy'])-center
                    choices.append((np.linalg.norm(g[:2]-BASES[side]),g))
                goal=sorted(choices,key=lambda x:x[0])[min(self.survey_attempt-1,len(choices)-1)][1]
            reached = self.move(action, obs, side, goal, 1.0, translation=0.025, rotation=0.2, tol=0.008, dwell=2)
            if reached or self.age > 60:
                measured = self.head.refine(obs, side, obj['rank'], obj['xy'])
                self.frames_to_save.append((f'survey{self.survey_index}', ('cam_head', f'cam_{side}_wrist')))
                if measured is None and getattr(self,'survey_attempt',0)<2:
                    self.survey_attempt=getattr(self,'survey_attempt',0)+1
                    self.age=0;self.dwell=0
                    return
                if measured is None:
                    self.events['measurement_failed'] = self.survey_index
                    self.stop('multiview_measurement_failed')
                    self.advance('home')
                    return
                self.survey_results[self.survey_index] = measured
                self.survey_attempt=0
                self.survey_index += 1
                self.age = 0
                self.dwell = 0
            return
        if p == 'prepare_execute':
            ready=[self.rest(action,obs,side) for side in ARMS]
            if all(ready) or self.age>45:self.advance('execute')
            return
        if p == 'execute':
            if len(self.done) >= len(self.tasks):
                self.advance('home')
                return
            for side in ARMS:
                self.run_arm(side, action, obs)
            if all((self.arm[s]['task'] is None for s in ARMS)) and len(self.done) < len(self.tasks):
                self.idle_steps = getattr(self, 'idle_steps', 0) + 1
                if self.idle_steps > 80:
                    self.events['deadlock'] = self.step
                    self.advance('home')
            else:
                self.idle_steps = 0
            return
        if p == 'home':
            done = True
            for s in ARMS:
                if not self.goto(action, obs, s, self.park_goal(s, 0.06), 1.0, translation=0.025, rotation=0.2, tol=0.02):
                    done = False
            if done or self.age > 70:
                self.advance('park')
            return
        if p == 'park':
            done = [self.goto(action, obs, s, self.park_goal(s, 0.06), 1.0, translation=0.02, rotation=0.15, tol=0.015) for s in ARMS]
            if all(done) or self.age > 60:
                self.advance('settle')
            return
        if p == 'settle':
            done = [self.goto(action, obs, s, self.park_goal(s), 1.0, translation=0.02, rotation=0.15, tol=0.01) for s in ARMS]
            if all(done) and (not self.finished):
                self.stop('program_complete' if len(self.done)==len(self.tasks) and self.tasks and self.reason=='initializing' else self.reason)
            return

    def amove(self, a, action, obs, side, goal, grip, translation=0.015, rotation=0.12, tol=0.008, rot_tol=0.1, dwell=2):
        current = obs['state'][side + '_ee_pose']
        err = float(np.linalg.norm(current[:3] - goal[:3]))
        ang = float(Rotation.from_matrix(quat_matrix(current[3:]).T @ quat_matrix(goal[3:])).magnitude())
        action[side + '_ee_pose'] = bounded_pose(current, goal, translation, rotation)
        action[side + '_ee_joint_state'] = np.array([grip])
        self.grips[side] = grip
        a['dwell'] = a['dwell'] + 1 if err < tol and ang < rot_tol else 0
        a['motion'] = dict(err=round(err, 4), ang=round(ang, 4), age=a['age'], ee=np.asarray(current).round(4).tolist())
        return a['dwell'] >= dwell

    def aadvance(self, a, side, phase):
        self.events.setdefault(f"{side[0]}{a['ti']}_{phase}", self.step)
        a['phase'] = phase
        a['age'] = 0
        a['dwell'] = 0

    def run_arm(self, side, action, obs):
        a = self.arm[side]
        a['age'] += 1
        if a['task'] is None:
            i = self.schedule(side, obs)
            if i is None:
                self.rest(action, obs, side)
                return
            a['task'] = self.prepare_task(dict(self.tasks[i]))
            a['ti'] = i
            self.active[side] = i
            xy = self.head.refit(obs['vision']['cam_head']['color'], {s: obs['state'][s + '_ee_pose'] for s in ARMS}, self.kind, a['task']['rank'], a['task']['pick'])
            self.log.setdefault('refit', []).append(dict(ti=i, **self.head.log.get('refit', {})))
            if xy is not None:
                a['task']['pick'] = xy.tolist()
            self.log.setdefault('tasks', []).append(dict(a['task'], step=self.step, ti=i))
            self.aadvance(a, side, 'travel_up')
        t = a['task']
        p = a['phase']
        if p == 'travel_up':
            if 'travel_xy' not in t:t['travel_xy']=tcp_of(obs['state'][side+'_ee_pose'])[:2].tolist()
            goal=vertical_pose(t['travel_xy'],max(self.hover_z,t['hover_z']),t['phi'])
            if self.amove(a,action,obs,side,goal,1.0,translation=.02,tol=.008) or a['age']>45:
                self.aadvance(a,side,'hover')
            return
        if p == 'hover':
            goal = vertical_pose(t['pick_tcp'], max(self.hover_z,t['hover_z']), t['phi'])
            if self.amove(a, action, obs, side, goal, t['pre_open'], translation=0.025, rotation=0.2, tol=0.008, rot_tol=0.05, dwell=2) or a['age'] > 60:
                self.aadvance(a, side, 'view')
            return
        if p == 'view':
            goal=vertical_pose(t['pick_tcp'],t['hover_z'],t['phi'])
            if self.amove(a,action,obs,side,goal,t['pre_open'],translation=.012,tol=.005) or a['age']>30:
                self.frames_to_save.append((f"hover{a['ti']}",(f'cam_{side}_wrist','cam_head')))
                self.aadvance(a,side,'refine')
            return
        if p == 'refine':
            goal = vertical_pose(t['pick_tcp'], t['hover_z'], t['phi'])
            self.amove(a, action, obs, side, goal, t['pre_open'])
            if a['age'] >= 2:
                measured = self.head.refine(obs, side, t['rank'], t['pick'])
                if measured is not None:
                    self.objects[t['rank']] = measured
                    t.update(pick=measured['xy'], zg=measured['grasp_height'], w=measured['grasp_width'])
                    self.prepare_task(t)
                self.aadvance(a, side, 'descend')
            return
        if p == 'descend':
            goal = vertical_pose(t['pick_tcp'], t['grasp_z'], t['phi'])
            if self.amove(a, action, obs, side, goal, t['pre_open'], translation=0.015, tol=0.004, rot_tol=0.05, dwell=2) or a['age'] > 45:
                self.log.setdefault('grasp_reach', []).append(dict(ti=a['ti'], err=a['motion']['err'], age=a['age']))
                self.aadvance(a, side, 'close')
            return
        if p == 'close':
            goal = vertical_pose(t['pick_tcp'], t['grasp_z'], t['phi'])
            self.amove(a, action, obs, side, goal, 0.0, translation=0.004)
            if a['age'] >= 8:
                opening=float(np.asarray(obs['state'][side+'_ee_joint_state']).reshape(-1)[0])
                if opening<max(.02,.2*t['w']/GRIP_OPENING):
                    n=t.get('grasp_retries',0)
                    self.events.setdefault('empty_grasps',[]).append(dict(step=self.step,task=a['ti'],attempt=n))
                    if n<2 and t['zg']>2*PAD_HALF:
                        t['grasp_retries']=n+1
                        delta=min(PAD_HALF/2,t['zg']/8)
                        for key in ['zg','grasp_z','place_z','carry_z']:t[key]-=delta
                        self.aadvance(a,side,'retry_open')
                    else:
                        self.reason='grasp_not_confirmed';self.advance('home')
                    return
                self.frames_to_save.append((f"closed{a['ti']}", (f'cam_{side}_wrist',)))
                self.aadvance(a, side, 'lift')
            return
        if p == 'retry_open':
            goal=vertical_pose(t['pick_tcp'],t['grasp_z']+PAD_HALF,t['phi'])
            self.amove(a,action,obs,side,goal,1.0,translation=.006)
            if a['age']>=6:self.aadvance(a,side,'descend')
            return
        if p == 'lift':
            goal = vertical_pose(t['pick_tcp'], t['carry_z'], t['phi'])
            if self.amove(a, action, obs, side, goal, 0.0, translation=0.015, tol=0.008) or a['age'] > 40:
                self.aadvance(a, side, 'carry')
            return
        if p == 'carry':
            goal = vertical_pose(t['carry_points'][t['carry_index']], t['carry_z'], t['phi'])
            if self.amove(a, action, obs, side, goal, 0.0, translation=0.018, tol=0.006, dwell=3) or a['age'] > 70:
                t['carry_index']+=1
                if t['carry_index']>=len(t['carry_points']):self.aadvance(a, side, 'lower')
                else:a['age']=0;a['dwell']=0
            return
        if p == 'lower':
            goal = vertical_pose(t['place_tcp'], t['place_z'], t['phi'])
            if self.amove(a, action, obs, side, goal, 0.0, translation=0.006, tol=0.003, dwell=3) or a['age'] > 45:
                self.frames_to_save.append((f"place{a['ti']}", (f'cam_{side}_wrist', 'cam_head')))
                self.aadvance(a, side, 'release')
            return
        if p == 'release':
            goal = vertical_pose(t['place_tcp'], t['place_z'], t['phi'])
            self.amove(a, action, obs, side, goal, 1.0, translation=0.004)
            if a['age'] >= 10:
                self.positions[t['rank']] = np.asarray(t['place'], float)
                self.aadvance(a, side, 'retreat')
            return
        if p == 'retreat':
            goal = vertical_pose(t['place_tcp'], max(self.hover_z,t['hover_z']), t['phi'])
            if self.amove(a, action, obs, side, goal, 1.0, translation=0.02, tol=0.01) or a['age'] > 30:
                self.done.add(a['ti'])
                self.active.pop(side, None)
                a['task'] = None
                a['phase'] = 'idle'
            return

    def diagnostics(self):
        arms = {s: dict(ti=a['ti'], phase=a['phase'], age=a['age'], motion=a['motion'], task=None if a['task'] is None else {k: a['task'][k] for k in ('rank', 'stage', 'pick', 'place')}) for (s, a) in getattr(self, 'arm', {}).items()}
        return dict(step=self.step, phase=self.phase, age=self.age, reason=self.reason, finished=self.finished, candidate='sort_nesting_dolls_by_size', kind=self.kind, objects=self.objects, done=sorted(getattr(self, 'done', [])), ntasks=len(self.tasks), arms=arms, plan=self.plan_log, events=self.events, log=self.log, grips=self.grips)
