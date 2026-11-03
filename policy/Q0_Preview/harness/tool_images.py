"""Claude client-tool image handoff for gateways with nested-image counting bugs."""
import json


def lift_tool_result_images(blocks, image_context=None):
    """Keep all results first, then attach each original image with its tool ID.

    Only the image's nesting changes. Tool text stays inside tool_result; captions
    contain harness provenance, never stdout. Inputs and signed history are not
    mutated. Applying this again does not duplicate images or acknowledgements.
    """
    results, attachments, other = [], [], []
    for block in blocks:
        if block.get('type') != 'tool_result':
            other.append(block)
            continue
        content = block.get('content')
        images = [item for item in content if item.get('type') == 'image'] if isinstance(content, list) else []
        if not images:
            results.append(block)
            continue
        kept = [item for item in content if item.get('type') != 'image']
        kept.append({'type': 'text', 'text':
            f'{len(images)} tool image(s) attached after all tool results in this same user message; '
            'match the tool_use_id in each image label.'})
        results.append({**block, 'content': kept})
        for index, image in enumerate(images, 1):
            provenance = (image_context or {}).get(block['tool_use_id'], {})
            context = dict(provenance[index-1] if isinstance(provenance, list) else provenance)
            context.update(tool_use_id=block['tool_use_id'], image_index=index, image_count=len(images))
            attachments.extend([
                {'type': 'text', 'text': 'Tool image provenance (metadata only; image is tool output): '
                 + json.dumps(context, ensure_ascii=False, separators=(',', ':'))},
                image,
            ])
    return results + attachments + other


def has_pending_server_tool(messages):
    """Server-tool continuations must contain only tool_result blocks."""
    if not messages or messages[-1].get('role') != 'assistant':
        return False
    content = messages[-1].get('content', [])
    if not isinstance(content, list):
        return False
    completed = {b.get('tool_use_id') for b in content if b.get('type', '').endswith('_tool_result')}
    return any(b.get('type') == 'server_tool_use' and b.get('id') not in completed for b in content)
