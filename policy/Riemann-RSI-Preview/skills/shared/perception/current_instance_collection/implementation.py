import numpy as np

from skills.shared.perception.current_metric_instances.implementation import instances, openings
from skills.shared.perception.public_support_geometry_context.implementation import run as observe_support


def support(api, previous=None):
    return observe_support(api, previous, passive_view=True)['memory']


def remember_support(api, current):
    previous = current['support_memory']
    if previous is None and current['scene'] is not None:
        previous = current['scene']['support_memory']
    observed = support(api, previous)
    current['support_memory'] = observed
    return observed


def collect(api, prompts, observed_support, canonical=None, require_supported=True, diagnostics=None):
    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 3:
        raise ValueError('A current collection uses at most three bounded semantic descriptions')
    outputs = []
    for prompt in prompts:
        candidates = instances(api, prompt, observed_support['height_estimate_m'], observed_support['height_uncertainty_m'],
            require_supported=require_supported, diagnostics=diagnostics)
        for item in candidates:
            if canonical is not None:
                item['anchor']['description'] = canonical
            if not any(np.linalg.norm(np.asarray(item['anchor']['center_m']) - previous['anchor']['center_m']) < .015 for previous in outputs):
                outputs.append(item)
    return outputs[:12]


def missing_hint(diagnostics, receiver_outlines=()):
    from skills.shared.perception.current_metric_instances.implementation import outside
    for query in diagnostics:
        for head in query['heads']:
            if head['qualified'] or any(not outside(head, outline) for outline in receiver_outlines):
                continue
            return {'source': 'current_public_head_semantic_hint', 'description': query['description'],
                'head_centroid': list(head['head_centroid']), 'unqualified_cameras': list(query['unqualified_cameras'])}
    return None


def upper_opening(api, receiver, observed_support):
    boundaries = openings(api, receiver['regions'], upright_only=True)
    height = receiver['anchor']['height_estimate_m']
    def qualified(boundary):
        if boundary.get('boundary_source') == 'current_public_closed_mask_line_boundary':
            uncertainty = np.hypot(boundary['position_uncertainty_m'], observed_support['height_uncertainty_m'])
            return (boundary.get('current_upper_edges_agree') is True
                and boundary['pose'][2] - observed_support['height_estimate_m'] >= 3. * uncertainty)
        return boundary['pose'][2] >= observed_support['height_estimate_m'] + height * .45
    candidates = [boundary for boundary in boundaries if qualified(boundary)]
    candidates.sort(key=lambda boundary: (boundary['pose'][2], np.prod(boundary['spans_m'])), reverse=True)
    return candidates[0] if candidates else None
