from copy import deepcopy

from skills.shared.motion.garment_joint_sampling.implementation import current_snapshot
from skills.shared.motion.garment_command_execution.implementation import GarmentProgram, unavailable
from skills.shared.perception.current_garment_landmarks.implementation import read
from skills.shared.planning.current_garment_contacts.implementation import run as contact_route
from skills.shared.verification.observed_garment_mount_state.implementation import recorded


class GarmentSceneView(GarmentProgram):
    def review_completion(self):
        # Keep the landmark axis only across retries of this same part.  A
        # successful fold changes the garment topology and its PCA orientation;
        # the first view of the next part must choose a fresh public outline
        # axis, while recovery retries should remain on the same branch.
        # Keep the retained public anchor across parts: the packaged rule folds
        # the second sleeve from the first flat estimate (fold_sleeve refines
        # only the local contact), and only the paired-hem stage re-reads the
        # torso silhouette.  Re-fitting a flat shirt after one sleeve is folded
        # yields no anchor and ends the episode with zero robot actions.
        previous = self.current['anchor']
        measured = read(self.api, previous, reuse=self.current['part'] == 1,
                        phase='flat' if self.current['part'] < 2 else 'sleeves_folded')
        if measured['anchor'] is None:
            self.current['attempts'] += 1
            self.current['phase'] = 'view'
            self.status = 'uncertain'
            return
        alternatives = measured.get('alternatives') or [measured['anchor']]
        for candidate in alternatives:
            try:
                planned = contact_route(self.api, candidate, self.current['part'])
            except ValueError:
                # The packaged rule raised here too when the public outline
                # carried no support for the landmark it was asked to grasp.
                # A candidate that cannot be planned is simply not a usable
                # alternative: try the rest instead of letting the raise end
                # the episode with zero robot actions.
                continue
            if planned is not None:
                self.current.update(anchor=candidate, route=planned, phase='approach', chunk=0)
                return
        self.current['anchor'] = measured['anchor']
        self.current['attempts'] += 1
        self.status = 'uncertain'


def run(api, request):
    current = recorded(api.command_memory(), api.instruction())
    if current['phase'] != 'view' or current['attempts'] >= 3 or current['part'] >= 3:
        return unavailable(api, request, current)
    snapshot = current_snapshot(api)
    controls = []
    for index in range(16):
        command = deepcopy(snapshot)
        for side in ('left', 'right'):
            opening = snapshot[side + '_ee_joint_state'][0]
            command[side + '_ee_joint_state'] = [opening + (1 - opening) * (index + 1) / 16]
        controls.append(command)
    return GarmentSceneView(api, request, current, controls)
