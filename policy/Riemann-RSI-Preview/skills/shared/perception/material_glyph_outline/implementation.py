"""Original v23 digit segmentation, independent template ranking and current-view identity verification."""
from __future__ import annotations
import math
import cv2
import numpy as np
from skills.shared.geometry.head_wrist_camera_calibration.implementation import CameraRig, finger_occlusion_mask
from skills.shared.geometry.x5_public_frame_model.implementation import CAMERAS, rotation
from skills.shared.perception.glyph_template_bank.implementation import RASTER_SIZE, RASTER_SCALE

class RecognitionNeedsObservation(RuntimeError):
    def __init__(self, message, report, scene=None):
        super().__init__(message)
        self.report, self.scene = report, scene


def components(mask, min_area=20):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    found = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area:
            continue
        rows, columns = np.where(labels == index)
        found.append(dict(mask=labels == index, pixels=np.c_[columns, rows],
                          bbox=[int(x), int(y), int(w), int(h)], area=int(area), centroid=centers[index]))
    return found


def material_mask(image, extended=False):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = np.moveaxis(hsv, -1, 0)
    # The public blue material has pale blue texture patches (e.g. RGB
    # 188,198,247). Their hue stays blue while saturation falls below 70.
    # Retain those measured pixels rather than filling geometric holes.
    blue = (hue >= 60) & (hue <= 140) & (saturation >= 35) & (value >= 30)
    purple = (hue >= 135) & (hue <= 179) & (saturation >= 40) & (value >= 130)
    red = ((hue <= 11) & (saturation >= 150) & (value >= 220)) | (
        (hue <= 18) & (saturation >= 50) & (value >= 248))
    salmon = (hue <= 19) & (saturation >= 30) & (saturation < 150) & (value >= 245)
    orange = ((hue >= 23) & (hue <= 36) & (saturation >= 30) & (value >= 245)) | (
        (hue >= 23) & (hue <= 36) & (saturation >= 50) & (value >= 235)) | (
        (hue >= 20) & (hue <= 22) & (saturation >= 160) & (value >= 230))
    mask = blue | red | orange
    if extended:
        mask |= purple | salmon
    return mask.astype(np.uint8), hsv


def hole_areas(mask):
    """Enclosed background regions, measured in original camera pixels."""
    background = (~mask.astype(bool)).astype(np.uint8)
    flooded = np.pad(background, 1, constant_values=1)
    cv2.floodFill(flooded, None, (0, 0), 2)
    holes = (flooded[1:-1, 1:-1] == 1)
    return sorted((item['area'] for item in components(holes, 3)), reverse=True)


class DigitTools:
    MIN_SIMILARITY = .885
    MIN_MARGIN = .012
    MAX_OCCLUSION = .02

    def __init__(self, bank, rig=None, material_profile='fixed'):
        self.bank, self.rig = bank, rig or CameraRig()
        if material_profile not in ('fixed', 'random'):
            raise ValueError('Use the official fixed or random public material preset')
        self.material_profile = material_profile
        # The five public material families have identical meshes per number.
        # Verify this before sharing their shape bank, rather than assuming it.
        self.records = {}
        for record in bank.records:
            digit = int(record['digit'])
            if digit in self.records:
                canonical = self.records[digit]
                if not np.array_equal(bank.masks[canonical['model_id']], bank.masks[record['model_id']]):
                    raise ValueError('Public material-family geometry differs; recognition bank must be rebuilt')
            else:
                self.records[digit] = record
        if set(self.records) != set(range(10)):
            raise ValueError('Recognition requires all ten public digit shapes')
        self._templates = {}
        self.edge_proposals=[]
        self.recovery_mode=False

    def segment(self, image, camera, frame, state, *, placed=False):
        if self.material_profile == 'random' or self.recovery_mode:
            return self.random_segment(image,camera,frame,state,placed=placed)
        mask, hsv = material_mask(image, extended=False)
        return self.segment_mask(mask,hsv,camera,frame,state,placed=placed)

    def random_segment(self,image,camera,frame,state,placed=False):
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        hue,saturation,value=np.moveaxis(hsv,-1,0)
        rgb=image.astype(float)
        masks=[]
        # Public salmon and purple materials under randomized illumination.
        # Several independent color thresholds avoid joining a glyph to a
        # nearly matching table stripe; their shape hypotheses are compared.
        for delta in (-2.,0.,3.,6.,9.):
            masks.append(((rgb[:,:,0]-rgb[:,:,1])>delta)&
                (rgb[:,:,0]>rgb[:,:,2]*.90)&(value>45)&(value<245))
        masks.append((hue>=125)&(hue<=179)&(saturation>=18)&(value>40))
        # Bright glyphs can exceed V=245 while their purple tabletop has
        # the same hue. Preserve their complete contours and holes using
        # independent lower brightness bands, without a highlight ceiling.
        for floor in (140,180,220):
            masks.append((hue>=125)&(saturation>=60)&(value>=floor))
            masks.append(((hue<=10)|(hue>=170))&(saturation>=60)&(value>=floor))
        # Recovery also includes the standard public blue/pale materials.
        # This runs only after uncertainty on the standard branch.
        if self.recovery_mode:
            masks.append(material_mask(image,extended=False)[0].astype(bool))
        found=[]
        for mask in masks:
            for item in self.segment_mask(mask.astype(np.uint8),hsv,camera,frame,state,placed=placed):
                if any(np.count_nonzero(item['mask']&other['mask'])/max(1,
                       np.count_nonzero(item['mask']|other['mask']))>.94 for other in found):continue
                found.append(item)
        return found

    def segment_mask(self,mask,hsv,camera,frame,state,placed=False):
        plane_z = frame.mounting_z + (.019 if placed else .016)
        rows, columns = np.where(mask)
        points = camera.pixels(np.c_[columns, rows], plane_z)
        inside = (np.isfinite(points).all(axis=1) & (np.abs(points[:, 0]) <= .47)
                  & (points[:, 1] >= -.40) & (points[:, 1] <= .04))
        mask[rows[~inside], columns[~inside]] = 0
        # Remove the blue wrist camera housing using only measured robot state.
        # Its projection into its own camera is not a valid forward pixel.
        if camera.name == 'cam_head':
            for arm in ('left', 'right'):
                ee = state[arm + '_ee_pose']
                center = np.asarray(ee[:3]) + rotation(ee[3:])[:, 0] * .077
                if center[2] < frame.mounting_z + .05:
                    continue
                uv = camera.project(center[None])[0]
                if np.isfinite(uv).all():
                    cv2.circle(mask, tuple(np.rint(uv).astype(int)), 16, 0, -1)
        found = []
        head = camera.name == 'cam_head'
        for item in components(mask, 45 if head else 70):
            x, y, w, h = item['bbox']
            if item['area'] > (1900 if head else 45000) or max(w, h) > (65 if head else 350) or min(w, h) < 5:
                continue
            xy = camera.pixels(item['pixels'], plane_z)[:, :2]
            if not np.isfinite(xy).all():
                continue
            span = np.ptp(xy, axis=0)
            if not (.014 < max(span) < .105 and min(span) > .005):
                continue
            hue = float(np.median(hsv[:, :, 0][item['mask']]))
            if hue >= 60 and float(np.median(hsv[:, :, 2][item['mask']])) < 140:
                continue
            item.update(hue=hue, center_xy=(xy.min(axis=0) + xy.max(axis=0)) / 2,
                        edge_clipped=(x <= 1 or y <= 1 or x + w >= 639 or y + h >= 479))
            found.append(item)
        return found

    def _bank(self, placed):
        angles = tuple(range(-90, 91, 3) if placed else range(-60, 61, 3))
        if angles not in self._templates:
            templates = []
            for record, yaw, mask in self.bank.rotated(0, angles):
                rows, columns = np.where(mask)
                center = np.array([(columns.min() + columns.max()) / 2,
                                   (rows.min() + rows.max()) / 2])
                shift = RASTER_SIZE / 2 - center
                aligned = cv2.warpAffine(mask, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]),
                                        (RASTER_SIZE, RASTER_SIZE))
                templates.append((record, yaw, aligned, shift))
            self._templates[angles] = templates
        return self._templates[angles]

    def project_candidate(self, candidate, camera, plane_z):
        corners = np.array([[0, 0], [RASTER_SIZE-1, 0], [RASTER_SIZE-1, RASTER_SIZE-1],
                            [0, RASTER_SIZE-1]], np.float32)
        xy = (corners - RASTER_SIZE / 2) / RASTER_SCALE
        angle = candidate['yaw_rad']
        rotation_xy = np.array([[math.cos(angle), -math.sin(angle)],
                                [math.sin(angle), math.cos(angle)]])
        world = xy @ rotation_xy.T + candidate['xy']
        uv = camera.project(np.c_[world, np.full(4, plane_z)]).astype(np.float32)
        if not np.isfinite(uv).all():
            return np.zeros((480, 640), np.uint8)
        transform = cv2.getPerspectiveTransform(corners, uv)
        record = self.records[candidate['digit']]
        return cv2.warpPerspective(self.bank.masks[record['model_id']], transform, (640, 480),
                                   flags=cv2.INTER_NEAREST)

    def volume_rank(self, item, camera, frame, placed=False):
        """Project the public extrusion through the fixed RGB camera.

        A planar backprojection enlarges visible glyph side walls and closes
        small holes. Search all ten public shapes at their known thickness,
        measuring overlap against the entire observed component.
        """
        x, y, w, h = item['bbox']
        origin = np.array([max(0, x-12), max(0, y-12)])
        roi = item['mask'][origin[1]:min(480, y+h+12), origin[0]:min(640, x+w+12)].astype(np.uint8)*255
        rh, rw = roi.shape
        corners = np.float32([[0, 0], [191, 0], [191, 191], [0, 191]])
        xy = (corners-RASTER_SIZE/2)/RASTER_SCALE + item['center_xy']
        support = .005 if placed else 0.
        best = {}
        for record, yaw, mask, shift in self._bank(placed):
            union = np.zeros((rh, rw), np.uint8)
            for z in np.linspace(frame.mounting_z+support,
                                 frame.mounting_z+support+record['height_m'], 7):
                uv = camera.project(np.c_[xy, np.full(4, z)]).astype(np.float32)-origin
                if not np.isfinite(uv).all():
                    continue
                transform = cv2.getPerspectiveTransform(corners, uv.astype(np.float32))
                union |= cv2.warpPerspective(mask, transform, (rw, rh), flags=cv2.INTER_NEAREST)
            pixels = cv2.findNonZero(union)
            if pixels is None:
                continue
            bx, by, bw, bh = cv2.boundingRect(pixels)
            crop = union[by:by+bh, bx:bx+bw]
            # Normalize against the full observed glyph, not a cropped window.
            scores = cv2.matchTemplate(roi, crop, cv2.TM_CCORR)/(255.*255.*
                math.sqrt(max(1, np.count_nonzero(roi)*np.count_nonzero(crop))))
            _, score, _, location = cv2.minMaxLoc(scores)
            digit = int(record['digit'])
            if digit not in best or score > best[digit]['score']:
                uv_shift = np.asarray(location)-[bx, by]
                center_pixel = camera.project(np.array([[*item['center_xy'], frame.mounting_z+support+.008]]))[0]
                shifted = camera.pixels(np.array([center_pixel+uv_shift]), frame.mounting_z+support+.008)[0, :2]
                best[digit] = dict(digit=digit, score=float(score),
                    xy=shifted+shift/RASTER_SCALE, yaw_rad=math.radians(yaw))
        return sorted(best.values(), key=lambda row: row['score'], reverse=True)

    def rank(self, item, camera, frame, state, *, placed=False, occlusion=None):
        """Return independently fitted candidates for ALL ten digits in one view."""
        center = item['center_xy']
        view_angle=float(np.degrees(np.arccos(np.clip(camera.rotation[2,2],-1.,1.))))
        plane_z = frame.mounting_z + (.019 if placed else .016)
        rows, columns = np.indices((RASTER_SIZE, RASTER_SIZE), dtype=float)
        plane = np.stack([columns - RASTER_SIZE / 2, rows - RASTER_SIZE / 2], axis=-1) / RASTER_SCALE + center
        uv = camera.project(np.c_[plane.reshape(-1, 2), np.full(RASTER_SIZE ** 2, plane_z)])
        uv = np.nan_to_num(uv, nan=-10000).reshape(RASTER_SIZE, RASTER_SIZE, 2).astype(np.float32)
        observed = cv2.remap(item['mask'].astype(np.uint8) * 255, uv[:, :, 0], uv[:, :, 1],
                             cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        padded = np.pad(observed, ((5, 5), (5, 5)))
        best = {}
        for record, yaw, template, shift in self._bank(placed):
            scores = cv2.matchTemplate(padded, template, cv2.TM_CCORR_NORMED)
            _, score, _, location = cv2.minMaxLoc(scores)
            digit = int(record['digit'])
            if digit not in best or score > best[digit]['score']:
                translation = np.asarray(location) - 5
                best[digit] = dict(digit=digit, score=float(score),
                                   xy=center + (shift + translation) / RASTER_SCALE,
                                   yaw_rad=math.radians(yaw))
        candidates = sorted(best.values(), key=lambda row: row['score'], reverse=True)
        first, second = candidates[:2]
        occluders, by_arm = occlusion if occlusion is not None else finger_occlusion_mask(camera, state, plane_z, self.rig)
        x, y, w, h = item['bbox']
        bbox_overlap = float(np.mean(occluders[y:y+h, x:x+w]))
        foreground = self.project_candidate(first, camera, plane_z) > 0
        overlap = float(np.count_nonzero(foreground & occluders) / max(1, np.count_nonzero(foreground)))
        blocking_arms = [arm for arm, mask in by_arm.items() if np.any(mask.astype(bool) & foreground)]
        observed_holes = hole_areas(item['mask'])
        topology = []
        for candidate in candidates[:3]:
            projected = self.project_candidate(candidate, camera, plane_z)
            holes = hole_areas(projected)
            topology.append(dict(digit=candidate['digit'], hole_areas_px=holes,
                                 hole_count_matches=len(holes) == len(observed_holes)))
        # Resolve only a close, otherwise strong fit whose runner-up requires
        # a clearly visible hole that is absent (or misses a visible hole).
        # Partial/occluded views still fail the independent completeness gate.
        runner_holes = topology[1]['hole_areas_px']
        extra_holes = (observed_holes[len(runner_holes):] if len(observed_holes) > len(runner_holes)
                       else runner_holes[len(observed_holes):])
        topology_resolves = (first['score'] >= .90 and item['area'] >= 140
                             and topology[0]['hole_count_matches'] and not topology[1]['hole_count_matches']
                             and min(extra_holes or [0]) >= 5)
        reasons = []
        if item['edge_clipped']:
            reasons.append('image_edge_clips_silhouette')
        if overlap > self.MAX_OCCLUSION:
            reasons.append('robot_finger_occlusion')
        significant_holes_match=(sum(area>=12 for area in observed_holes)==
                                 sum(area>=12 for area in topology[0]['hole_areas_px']))
        if view_angle<=12. and item['area']>=600 and not significant_holes_match:
            reasons.append('visible_hole_topology_conflict')
        margin = first['score'] - second['score']
        planar_candidates = None
        if view_angle>12. and first['score']>=.82 and (margin < self.MIN_MARGIN or first['score'] < self.MIN_SIMILARITY) and not (
                item['edge_clipped'] or overlap > self.MAX_OCCLUSION):
            volumetric = self.volume_rank(item, camera, frame, placed)
            # Extrusion is an approximate renderer. A new identity must also
            # remain among the two independently supported planar shapes.
            # Otherwise retain the planar ambiguity for another current view.
            volume_supported = (len(volumetric) == 10 and volumetric[0]['digit'] in
                {candidate['digit'] for candidate in candidates[:2]} and
                first['score'] - next(candidate['score'] for candidate in candidates
                    if candidate['digit'] == volumetric[0]['digit']) <= .035)
            if volume_supported and volumetric[0]['score'] >= .88 and (
                    volumetric[0]['score']-volumetric[1]['score'] >= .005):
                planar_candidates = candidates
                candidates = volumetric
                first, second = candidates[:2]
                margin = first['score']-second['score']
                topology_resolves = False
        # A distinctive, complete silhouette may have lower similarity because
        # its bright surface has texture gaps. Retain the independent margin.
        distinctive = first['score'] >= .86 and margin >= .07
        if first['score'] < self.MIN_SIMILARITY and not distinctive:
            reasons.append('incomplete_template_support')
        volume_distinctive=(planar_candidates is not None and first['score']>=.89 and margin>=.006)
        if margin < self.MIN_MARGIN and not topology_resolves and not volume_distinctive:
            reasons.append('competing_digits_too_close')
        return dict(camera=camera.name, view_angle_deg=view_angle, candidates=candidates,
                    planar_candidates=planar_candidates,
                    fit_geometry='public_USD_extrusion' if planar_candidates is not None else 'planar',
                    winner=first['digit'], similarity=first['score'], margin=float(margin),
                    reliable=not reasons, reasons=reasons, bbox=item['bbox'], hue=item['hue'],
                    pixel_area=item['area'], occluded_bbox_fraction=bbox_overlap,
                    occluded_glyph_fraction=overlap, blocking_arms=blocking_arms,
                    occlusion_semantics='public_opening_hypotheses_not_measured_width',
                    topology=dict(observed_hole_areas_px=observed_holes, candidates=topology,
                                  resolves_close_candidates=bool(topology_resolves),significant_holes_match=significant_holes_match),
                    center_xy=center, edge_clipped=item['edge_clipped'])

    def detect_digit_candidates(self, obs, frame, *, expected_xy=None, placed=False,inspection_xy=None):
        """Current-view candidates and independent consensus, with no identity hint."""
        cameras = self.rig.current(obs)
        items = {name: self.segment(obs.vision[name], cameras[name], frame, obs.state, placed=placed)
                 for name in CAMERAS}
        plane_z = frame.mounting_z + (.019 if placed else .016)
        occlusion = {name: finger_occlusion_mask(cameras[name], obs.state, plane_z, self.rig) for name in CAMERAS}
        ranked_cache={}
        edge_proposals=[]
        if self.material_profile=='random' or self.recovery_mode:
            selected={}
            for name in CAMERAS:
                groups=[]
                for index,item in enumerate(items[name]):
                    view=self.rank(item,cameras[name],frame,obs.state,placed=placed,occlusion=occlusion[name])
                    # Texture regions and unrelated clutter are not complete
                    # public digit silhouettes. Keep occluded strong remnants
                    # as uncertain observations for active visual clearing.
                    if view['similarity']<.85:continue
                    group=next((g for g in groups if np.linalg.norm(g[0]['center_xy']-item['center_xy'])<.025),None)
                    if group is None:groups.append([item,view])
                    elif view['similarity']>group[1]['similarity']:group[:]=[item,view]
                selected[name]=[g[0] for g in groups]
                for index,g in enumerate(groups):ranked_cache[(name,index)]=g[1]
            items=selected
            if self.recovery_mode or expected_xy is not None or inspection_xy is not None or len(items['cam_head'])<4 or any(not ranked_cache[('cam_head',index)]['reliable']
                    for index in range(len(items['cam_head']))):
                from skills.shared.perception.glyph_outline_search.implementation import propose,regions
                proposal_xy=expected_xy
                if proposal_xy is None and inspection_xy is not None:
                    proposal_xy=np.asarray(inspection_xy).reshape(-1,2)[-1]
                edge_proposals=propose(self,obs.vision['cam_head'],cameras['cam_head'],frame,obs.state,placed,proposal_xy)
                region_proposals=list(edge_proposals)
                if expected_xy is not None:
                    region_proposals.append(dict(xy=np.asarray(expected_xy),source='RGB_transfer_prediction_for_local_search'))
                if inspection_xy is not None:
                    for xy in np.asarray(inspection_xy,dtype=float).reshape(-1,2):
                        region_proposals.append(dict(xy=xy,source='RGB_measured_location_for_current_view_segmentation'))
                def unresolved_regions(name):
                    return [proposal for proposal in region_proposals if not any(
                        np.linalg.norm(old['center_xy']-proposal['xy'])<.018 and
                        ranked_cache[(name,index)]['reliable'] and ranked_cache[(name,index)]['similarity']>=.93
                        for index,old in enumerate(items[name]))]
                for item in regions(self,obs.vision['cam_head'],cameras['cam_head'],frame,unresolved_regions('cam_head'),placed):
                    view=self.rank(item,cameras['cam_head'],frame,obs.state,placed=placed,occlusion=occlusion['cam_head'])
                    edge=item['edge_proposal']
                    # Proposal and independent region must agree on identity.
                    # A disagreement stays uncertain until a new wrist view.
                    if 'digit' in edge and view['winner']!=edge['digit']:
                        view['reasons'].append('edge_foreground_identity_disagreement');view['reliable']=False
                        # A proposal is a location to inspect, not an object
                        # count or a replacement for independent color evidence.
                        continue
                    if view['similarity']<.85:continue
                    if not view['reliable'] and view['similarity']<.89:continue
                    existing=next((index for index,old in enumerate(items['cam_head'])
                        if np.linalg.norm(old['center_xy']-item['center_xy'])<.025),None)
                    if existing is None:
                        index=len(items['cam_head']);items['cam_head'].append(item);ranked_cache[('cam_head',index)]=view
                    elif view['similarity']>ranked_cache[('cam_head',existing)]['similarity']:
                        items['cam_head'][existing]=item;ranked_cache[('cam_head',existing)]=view
                for name in CAMERAS[1:]:
                    for item in regions(self,obs.vision[name],cameras[name],frame,unresolved_regions(name),placed):
                        view=self.rank(item,cameras[name],frame,obs.state,placed=placed,occlusion=occlusion[name])
                        if view['similarity']<.85:continue
                        if not view['reliable'] and view['similarity']<.89:continue
                        index=next((index for index,old in enumerate(items[name])
                            if np.linalg.norm(old['center_xy']-item['center_xy'])<.025),None)
                        if index is None:
                            index=len(items[name]);items[name].append(item);ranked_cache[(name,index)]=view
                        elif view['similarity']>ranked_cache[(name,index)]['similarity']:
                            items[name][index]=item;ranked_cache[(name,index)]=view
        self.edge_proposals=edge_proposals
        anchors=[('cam_head',index,item) for index,item in enumerate(items['cam_head'])]
        # A clear current wrist observation can supply an anchor when the
        # head foreground mask missed it. No historical view is used.
        if self.material_profile=='random' or self.recovery_mode:
            for name in CAMERAS[1:]:
                for index,item in enumerate(items[name]):
                    view=ranked_cache.get((name,index))
                    # RGB position seeds locate a new inspection. They do not
                    # weaken identity support for a wrist-only scene object.
                    if expected_xy is None:
                        normal=(view is not None and view['view_angle_deg']<=12.)
                        distinctive=(view is not None and view['similarity']>=.90 and view['margin']>=.06)
                        supported=(view is not None and view['reliable'] and
                            ((normal and ((view['similarity']>=.925 and view['margin']>=.025) or distinctive)) or
                             (not normal and view['similarity']>=.96 and view['margin']>=.04)))
                        if not supported:continue
                    if not any(np.linalg.norm(item['center_xy']-old[2]['center_xy'])<.025 for old in anchors):
                        anchors.append((name,index,item))
        if expected_xy is not None:
            anchors = [entry for entry in anchors if np.linalg.norm(entry[2]['center_xy'] - expected_xy) < .06]
        records = []
        for anchor_camera,anchor_index,anchor in anchors:
            views=[ranked_cache.get((anchor_camera,anchor_index)) or self.rank(anchor,
                cameras[anchor_camera],frame,obs.state,placed=placed,occlusion=occlusion[anchor_camera])]
            for name in CAMERAS:
                if name==anchor_camera:continue
                candidates = sorted(enumerate(items[name]),
                                    key=lambda pair: np.linalg.norm(pair[1]['center_xy'] - anchor['center_xy']))
                if not candidates:
                    continue
                index, item = candidates[0]
                distance = float(np.linalg.norm(item['center_xy'] - anchor['center_xy']))
                if distance > .025:
                    continue
                # Reject ambiguous cross-view association instead of joining neighboring glyphs.
                if len(candidates) > 1 and np.linalg.norm(candidates[1][1]['center_xy'] - anchor['center_xy']) < distance + .012:
                    continue
                key = (name, index)
                if key not in ranked_cache:
                    ranked_cache[key] = self.rank(item, cameras[name], frame, obs.state, placed=placed,
                                                   occlusion=occlusion[name])
                view = dict(ranked_cache[key], association_distance_m=distance)
                views.append(view)
            views = qualify_public_finger_overlap(views)
            reliable = [view for view in views if view['reliable']]
            # Independent clear views can corroborate a close candidate even
            # when neither individual margin reaches the single-view gate.
            # Occluded, clipped and low-support views never participate.
            clear = [view for view in views if not any(reason in view['reasons'] for reason in
                     ('robot_finger_occlusion', 'image_edge_clips_silhouette','edge_foreground_identity_disagreement','visible_hole_topology_conflict'))
                     and view['similarity'] >= .88 and (view['margin'] >= .005 or
                         (view['fit_geometry'] == 'planar' and view['similarity'] >= .92 and view['margin'] >= .001))]
            if len(clear) >= 2 and len({view['winner'] for view in clear}) == 1:
                reliable = list({view['camera']: view for view in reliable + clear}.values())
                for view in reliable:
                    if not view['reliable']:
                        view['confirmed_by_current_view_agreement'] = True
            normal=[view for view in reliable if view['view_angle_deg']<=12.
                    and view['fit_geometry']=='planar' and view['pixel_area']>=600
                    and view['similarity']>=.95 and view['margin']>=.02
                    and view['topology']['significant_holes_match']]
            superseded=[]
            if normal and len({view['winner'] for view in normal})==1:
                digit=normal[0]['winner']
                superseded=[view for view in reliable if view['winner']!=digit
                            and view['view_angle_deg']>12. and view['similarity']<.965 and view['margin']<.05]
                for view in superseded:
                    view['identity_judgment_superseded_by_current_near_normal_view']=True
                superseded_names={view['camera'] for view in superseded}
                reliable=[view for view in reliable if view['camera'] not in superseded_names]
            winners = {view['winner'] for view in reliable}
            status = 'confirmed' if len(winners) == 1 else 'uncertain' if not winners else 'view_conflict'
            detection = None
            if status == 'confirmed':
                digit = next(iter(winners))
                # Retain head localization when it is clear; use the wrist when the head is occluded.
                pose_views=[view for view in reliable if view['winner']==digit
                            and view['view_angle_deg']<=12. and view['fit_geometry']=='planar'
                            and view['similarity']>=.925 and view['margin']>=.025
                            and view['topology']['significant_holes_match']]
                selected = (max(pose_views,key=lambda view:view['similarity']) if pose_views else
                            next((view for view in reliable if view['camera']=='cam_head'),
                                 max(reliable,key=lambda view:view['similarity'])))
                fitted = selected['candidates'][0]
                public = self.records[digit]
                detection = dict(fitted, model_id=public['model_id'], family=public['family'],
                                 height_m=public['height_m'], origin_to_bottom_m=public['origin_to_bottom_m'],
                                 bbox_xy=(np.asarray(public['bbox'][1]) - public['bbox'][0])[:2],
                                 bbox=anchor['bbox'], hue=anchor['hue'], centroid_uv=anchor['centroid'],
                                 scores={row['digit']: row['score'] for row in selected['candidates']},
                                 source='current_RGB_independent_public_USD_candidates',
                                 selected_camera=selected['camera'], identity_margin=selected['margin'],
                                 pose_similarity=selected['similarity'],pose_view_angle_deg=selected['view_angle_deg'],
                                 observation_id=obs.observation_id, physical_step=obs.physical_step)
            records.append(dict(status=status, anchor_xy=anchor['center_xy'], views=views, detection=detection,
                arbitration=dict(normal_cameras=[v['camera'] for v in normal],
                                 superseded_oblique_judgments=[v['camera'] for v in superseded])))
        return dict(tool='detect_digit_candidates', observation_id=obs.observation_id,
                    physical_step=obs.physical_step, evidence=obs.evidence,
                    material_profile=self.material_profile,
                    recovery_mode=self.recovery_mode,
                    edge_proposals=edge_proposals,
                    score_semantics='template_similarity_not_probability', objects=records,
                    component_counts={name: len(value) for name, value in items.items()},
                    thresholds=dict(min_similarity=self.MIN_SIMILARITY, min_margin=self.MIN_MARGIN,
                                    max_occluded_glyph_fraction=self.MAX_OCCLUSION))

    def request_reobservation(self, report, reason):
        """An observation request only; this tool cannot execute robot actions."""
        arms = sorted({arm for obj in report.get('objects', []) for view in obj['views']
                       for arm in view['blocking_arms']})
        return dict(tool='request_digit_reobservation', reason=reason,
                    observation_id=report['observation_id'], physical_step=report['physical_step'],
                    suggested_clear_arms=arms or ['left', 'right'], max_clear_attempts=None,
                    remaining_native_controls=max(0,1050-report['physical_step']),
                    next_input='new_current_head_left_wrist_right_wrist_640x480_RGB_and_measured_state',
                    action_execution='existing_checked_active_vision_generator_only')

    def verify_digit_identity(self, obs, frame, expected_digit, expected_xy, *, placed=False):
        # expected_digit is deliberately excluded from candidate generation/ranking.
        report = self.detect_digit_candidates(obs, frame, expected_xy=expected_xy, placed=placed)
        report['tool'] = 'verify_digit_identity'
        report['expected_digit'] = int(expected_digit)
        candidates = sorted(report['objects'], key=lambda obj: np.linalg.norm(obj['anchor_xy'] - expected_xy))
        if len(candidates) > 1:
            distances = [np.linalg.norm(obj['anchor_xy'] - expected_xy) for obj in candidates[:2]]
            if distances[1] - distances[0] < .012:
                candidates = []
        if candidates and candidates[0]['status'] == 'confirmed':
            result = candidates[0]['detection']
            if result['digit'] == int(expected_digit):
                # Thin 1 has a distinctive identity even with a broken
                # silhouette. That does not certify its placement angle.
                minimum=.90 if result['pose_view_angle_deg']<=12. else .92
                if placed and result['digit']==1 and result['pose_similarity']<minimum:
                    report['status']='pose_uncertain'
                    reason='Digit identity is verified but its thin placement contour needs a clearer current view'
                    report['reobservation']=self.request_reobservation(report,reason)
                    raise RecognitionNeedsObservation(reason,report)
                report['status'] = 'verified'
                return result, report
            reason = f"Current RGB identifies {result['digit']}, contradicting expected digit {expected_digit}"
            report['status'] = 'identity_conflict'
        else:
            reason = f'Current RGB cannot independently verify digit {expected_digit} near the requested position'
            report['status'] = 'uncertain'
        report['reobservation'] = self.request_reobservation(report, reason)
        raise RecognitionNeedsObservation(reason, report)
def qualify_public_finger_overlap(views):
    qualified = [dict(view) for view in views]
    for view in qualified:
        if (view.get('occlusion_semantics') != 'public_opening_hypotheses_not_measured_width'
                or view['reasons'] != ['robot_finger_occlusion']
                or view['view_angle_deg'] > 12. or view['fit_geometry'] != 'planar'
                or view['pixel_area'] < 600 or view['similarity'] < .925
                or view['margin'] < .025 or view['edge_clipped']
                or not view['topology']['significant_holes_match']):
            continue
        independent = [other for other in views
                       if other['camera'] != view['camera'] and other['reliable']]
        if not independent or any(other['winner'] != view['winner'] for other in independent):
            continue
        view['reasons'] = []
        view['reliable'] = True
        view['public_finger_overlap_evidence'] = dict(
            state=None, real_width_observed=False,
            source='public_asset_opening_hypotheses',
            hypothetical_glyph_fraction=view['occluded_glyph_fraction'],
            current_rgb_judgment='independent_digit_identity_and_near_normal_pose',
            support_cameras=[view['camera']] + [other['camera'] for other in independent])
    return qualified
