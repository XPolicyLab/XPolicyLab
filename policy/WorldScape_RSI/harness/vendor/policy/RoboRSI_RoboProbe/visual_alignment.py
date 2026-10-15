"""Local image-based servo fit from observed gripper pixels and measured TCPs.

No camera model, simulator metadata, object coordinates or stored calibration.
Samples track the same contact-center landmark in one fixed RGB view.
"""
import math


def vec(value, n):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError('Expected %d coordinates' % n)
    result = [float(x) for x in value]
    if not all(math.isfinite(x) for x in result):
        raise ValueError('Coordinates must be finite')
    return result


def solve(matrix, values):
    """Small least-squares normal system, rejecting unobserved directions."""
    n=len(values);rows=[list(matrix[i])+[values[i]] for i in range(n)]
    scale=max(abs(x) for row in matrix for x in row)
    for col in range(n):
        pivot=max(range(col,n),key=lambda i:abs(rows[i][col]))
        if abs(rows[pivot][col])<max(1e-8,scale*1e-6):
            raise ValueError('Motion samples do not span the observed directions; add an independent observation')
        rows[col],rows[pivot]=rows[pivot],rows[col]
        divisor=rows[col][col];rows[col]=[x/divisor for x in rows[col]]
        for i in range(n):
            if i==col:continue
            factor=rows[i][col];rows[i]=[a-factor*b for a,b in zip(rows[i],rows[col])]
    return [row[-1] for row in rows]


def predict_target(track, lookahead_steps):
    """Predict one approach from RGB displacement per actual native action."""
    if not 2 <= len(track) <= 6:
        raise ValueError('Use 2-6 observations of the same moving target in one fixed camera')
    rows=[(float(s['steps']),vec(s['pixel'],2)) for s in track]
    horizon=float(lookahead_steps)
    if not math.isfinite(horizon) or horizon < 0 or any(not math.isfinite(s) for s,_ in rows):
        raise ValueError('Use finite native action counts and a nonnegative prediction horizon')
    if any(b[0] <= a[0] for a,b in zip(rows,rows[1:])):
        raise ValueError('Target observations must advance native steps; model thinking time is not motion')
    mean_s=sum(s for s,_ in rows)/len(rows)
    mean_p=[sum(p[k] for _,p in rows)/len(rows) for k in (0,1)]
    denominator=sum((s-mean_s)**2 for s,_ in rows)
    velocity=[sum((s-mean_s)*(p[k]-mean_p[k]) for s,p in rows)/denominator for k in (0,1)]
    predicted=[rows[-1][1][k]+horizon*velocity[k] for k in (0,1)]
    return {'target_pixel':predicted,'velocity_px_per_native_step':velocity,'lookahead_steps':horizon,
            'last_observed_step':rows[-1][0],'source':'same_target_RGB_displacement_and_native_steps'}


def correction(samples, current_tcp, gripper_pixel, target_pixel,
               max_correction_m=0.06, gain=0.8):
    if not 3 <= len(samples) <= 12:
        raise ValueError('Use 3-12 current-episode observations of the same gripper point')
    rows = [(vec(s['tcp'], 3), vec(s['pixel'], 2)) for s in samples]
    tcp = vec(current_tcp, 3)
    here, target = vec(gripper_pixel, 2), vec(target_pixel, 2)
    heights = [p[2] for p, _ in rows]
    dimensions=3 if max(heights)-min(heights)>.008 else 2
    if dimensions==3 and len(rows)<4:
        raise ValueError('Changing-height observations need at least four independent XYZ samples')
    if dimensions==2 and abs(tcp[2] - sum(heights)/len(heights)) > 0.04:
        raise ValueError('Height changed substantially; collect local probes at the new height')
    cap, gain = float(max_correction_m), float(gain)
    if not 0 < cap <= 0.12 or not 0 < gain <= 1:
        raise ValueError('Use a local correction cap in (0, .12] m and gain in (0, 1]')
    n = len(rows)
    axes=range(dimensions)
    mean_p = [sum(p[k] for p, _ in rows)/n for k in axes]
    mean_v = [sum(v[k] for _, v in rows)/n for k in (0, 1)]
    centered = [([p[k]-mean_p[k] for k in axes],
                 [v[k]-mean_v[k] for k in (0, 1)]) for p, v in rows]
    normal=[[sum(p[i]*p[j] for p,_ in centered) for j in axes] for i in axes]
    jacobian = []
    for k in (0, 1):
        jacobian.append(solve(normal,[sum(p[i]*v[k] for p,v in centered) for i in axes]))
    a, b = jacobian[0][:2]
    c, d = jacobian[1][:2]
    determinant = a*d-b*c
    scale = a*a+b*b+c*c+d*d
    if abs(determinant) < 0.02*scale:
        raise ValueError('This view cannot resolve both XY directions; use a clearer fixed view')
    residual = math.sqrt(sum((v[k]-sum(jacobian[k][j]*p[j] for j in axes))**2
                             for p, v in centered for k in (0, 1))/n)
    if residual > 8:
        raise ValueError('Gripper tracking is inconsistent (%.1f px); reidentify the same point' % residual)
    error = [target[k]-here[k] for k in (0, 1)]
    delta = [gain*(d*error[0]-b*error[1])/determinant,
             gain*(a*error[1]-c*error[0])/determinant]
    length = math.hypot(*delta)
    if length > cap:
        delta = [x*cap/length for x in delta]
    return {
        'position': [tcp[0]+delta[0], tcp[1]+delta[1], tcp[2]],
        'delta_xy_m': delta,
        'pixel_error': error,
        'fit_residual_px': residual,
        'local_jacobian_px_per_m': jacobian,
        'fit_axes': 'XYZ' if dimensions==3 else 'XY',
        'source': 'current_episode_rgb_and_measured_robot_motion',
        'next': 'Reobserve the same gripper point and target; verify error shrinks. '
                'XY alignment does not establish depth or a grasp. Descend separately and verify retention.'
    }
