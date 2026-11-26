"""Fixed image-history and instruction input for the Qwen feature encoder."""
from copy import deepcopy
from functools import lru_cache

from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence

ACTION_REQUEST = "Predict the next second's action subtask from the instruction and observation."


class QwenObservationProcessor:
    def __init__(self, processor, *, image_size=448, history_grid=(4, 4), history_limit=20):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_size = (image_size, image_size)
        self.history_limit = history_limit
        image_processor = processor.image_processor
        self.merge = image_processor.merge_size
        factor = image_processor.patch_size * self.merge
        rows, cols = history_grid
        self.placeholder_size = (cols * factor, rows * factor)
        self.history_grid = torch.tensor([[1, rows * self.merge, cols * self.merge]])
        self.history_resize = {'size': {'shortest_edge': rows * cols * factor**2,
                                       'longest_edge': 16777216}}
        self.image_id = self.tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id

    @staticmethod
    def messages(instruction, current, history):
        content = []
        if history:
            content.append({'type': 'text', 'text': f'Past {len(history)} frames, 1 s apart, oldest first:'})
            content.extend({'type': 'image', 'image': image} for image in history)
            content.append({'type': 'text', 'text': 'Current frame:'})
        content.extend([{'type': 'image', 'image': current},
                        {'type': 'text', 'text': f'Instruction: {instruction}\n{ACTION_REQUEST}'}])
        return [{'role': 'user', 'content': content}]

    def _instruction_positions(self, tokens, instruction):
        # Token boundaries depend on the leading space. Keep the same boundary
        # rules for punctuation and short instructions as the deployed checkpoint.
        text = instruction.strip()
        candidates = []
        for variant in (' ' + text, text, ' ' + text.rstrip('.'), text.rstrip('.')):
            ids = self.tokenizer(variant, add_special_tokens=False)['input_ids']
            candidates.extend(candidate for candidate in (ids, ids[:-1]) if len(candidate) >= 3)
        for variant in (' ' + text, text):
            ids = self.tokenizer(variant, add_special_tokens=False)['input_ids']
            if 0 < len(ids) < 3:
                candidates.append(ids)
        for candidate in candidates:
            for start in range(len(tokens) - len(candidate) + 1):
                if tokens[start:start + len(candidate)] == candidate:
                    return torch.arange(start, start + len(candidate))
        raise ValueError('Instruction token span is absent from the Qwen input')

    @lru_cache(maxsize=256)
    def prompt(self, instruction, history_count):
        if not 0 <= history_count <= self.history_limit:
            raise ValueError('Qwen history must contain between 0 and 20 past images')
        current = Image.new('RGB', self.image_size)
        history = [Image.new('RGB', self.placeholder_size) for _ in range(history_count)]
        text = self.processor.apply_chat_template(self.messages(instruction, current, history),
                                                 tokenize=False, add_generation_prompt=True)
        encoded = self.processor(text=[text], images=history + [current], return_tensors='pt',
                                 padding=False, images_kwargs=deepcopy(self.history_resize))
        ids = encoded['input_ids'][0]
        return ids, self._instruction_positions(ids.tolist(), instruction)

    def images(self, current, history):
        pixels, vision_grids, language_grids = [], [], []
        if any(image.size != self.image_size for image in [*history, current]):
            raise ValueError('Qwen images must be resized to the configured square resolution')
        for images, resize, past in ((history, self.history_resize, True), ([current], {}, False)):
            if not images:
                continue
            result = self.processor.image_processor(images=images, return_tensors='pt', **resize)
            pixels.append(result['pixel_values'])
            vision_grids.append(result['image_grid_thw'])
            language_grids.append(self.history_grid.repeat(len(images), 1) if past else result['image_grid_thw'])
        return {'pixel_values': torch.cat(pixels), 'vision_grid': torch.cat(vision_grids),
                'language_grid': torch.cat(language_grids),
                'history_mask': torch.tensor([True] * len(history) + [False])}

    def batch(self, instructions, currents, histories):
        if not instructions or not (len(instructions) == len(currents) == len(histories)):
            raise ValueError('Qwen requires matching nonempty instruction/image/history batches')
        prompts, spans, image_rows = [], [], []
        for instruction, current, history in zip(instructions, currents, histories, strict=True):
            ids, span = self.prompt(instruction.strip(), len(history))
            images = self.images(current, history)
            image_tokens = int(images['language_grid'].prod(-1).sum()) // self.merge**2
            if int((ids == self.image_id).sum()) != image_tokens:
                raise ValueError('Qwen image placeholders and visual tokens disagree')
            prompts.append(ids); spans.append(span); image_rows.append(images)
        ids = pad_sequence(prompts, batch_first=True, padding_value=self.pad_id)
        lengths = torch.tensor([len(row) for row in prompts])
        return {
            'input_ids': ids,
            'attention_mask': (torch.arange(ids.shape[1])[None] < lengths[:, None]).long(),
            'instruction_positions': pad_sequence(spans, batch_first=True, padding_value=-1),
            'summary_positions': lengths - 1,
            'modality_ids': (ids == self.image_id).long(),
            **{key: torch.cat([row[key] for row in image_rows]) for key in image_rows[0]},
        }
