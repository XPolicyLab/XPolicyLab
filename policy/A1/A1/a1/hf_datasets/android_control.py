# Download the Android Control dataset here: https://console.cloud.google.com/storage/browser/gresearch/android_control
# This script loads in the data from the original format and processes it into a TFDS format needed for the Molmo pipeline.
# To use it:
# pip install tensorflow tensorflow_datasets pillow protobuf
#
# Clone https://github.com/google-deepmind/android_env.git
# install it, and add it to PYTHONPATH

import io
import json
from typing import Dict

import datasets
from PIL import Image
from datasets import tqdm

from a1.hf_datasets.android_control_utils import *

TF_RECORD_NAMES = [
    f"android_control-000{str(i).zfill(2)}-of-00020"
    for i in range(20)
]
TFRECORD_URLS = [
    f"https://storage.googleapis.com/gresearch/android_control/{name}"
    for name in TF_RECORD_NAMES
]
DATA_URLS = [
    "https://storage.googleapis.com/gresearch/android_control/splits.json",
    "https://storage.googleapis.com/gresearch/android_control/test_subsplits.json"
]


def process_data(data_files: Dict[str, str]):
    from google.protobuf.json_format import MessageToJson
    try:
        from android_env.proto.a11y import android_accessibility_forest_pb2
    except ImportError as e:
        raise ImportError("Unable to import android control protos, make sure https://github.com/google-deepmind/android_env"
                          " is in the PYTHONPATH")

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("Building android control requires tensorflow to be installed")

    # Train/Val/Test splits
    splits_file = data_files['splits.json']
    f2 = open(splits_file)
    ac_splits = json.load(f2)
    train_episodes = ac_splits['train']
    val_episodes = ac_splits['validation']
    test_episodes = ac_splits['test']
    total = len(train_episodes) + len(val_episodes) + len(test_episodes)

    # NOTE: If split == test then choose the subsplit if desired.
    test_subsplits_file = data_files['test_subsplits.json']
    test_splits_file = open(test_subsplits_file)
    test_subsplits = json.load(test_splits_file)
    test_subsplit_episodes = test_subsplits['IDD']

    # Preprocess Android Control datapoints
    filenames = [data_files[name] for name in TF_RECORD_NAMES]
    raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')

    processed_data = {
        'train': [],
        'val': [],
        'test': []
    }

    for record in tqdm(raw_dataset, desc="Processing", total=total):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())

        parsed_data = {
            'episode_id': example.features.feature['episode_id'].int64_list.value,
            'goal': example.features.feature['goal'].bytes_list.value[0].decode('utf-8'),
            'screenshots': [screenshot for screenshot in example.features.feature['screenshots'].bytes_list.value],
            'screenshot_widths': example.features.feature['screenshot_widths'].int64_list.value,
            'screenshot_heights': example.features.feature['screenshot_heights'].int64_list.value,
            'actions': [action.decode('utf-8') for action in example.features.feature['actions'].bytes_list.value],
            'step_instructions': [instruction.decode('utf-8') for instruction in example.features.feature['step_instructions'].bytes_list.value],
        }

        a11y_tree = android_accessibility_forest_pb2.AndroidAccessibilityForest().FromString(example.features.feature['accessibility_trees'].bytes_list.value[0])
        episode_id = parsed_data['episode_id'][0]

        action_history = []
        screenshots = parsed_data['screenshots']
        for i,screen in enumerate(screenshots):
            if i == len(screenshots)-1:
                continue

            img_bytes = screen
            dims = Image.open(io.BytesIO(img_bytes)).size
            # img = np.array(img)[:, :, :3]
            # dims = img.shape

            cur_action = process_action(parsed_data['actions'][i])
            instruction = parsed_data['step_instructions'][i]
            goal = parsed_data['goal']

            a11y_tree = android_accessibility_forest_pb2.AndroidAccessibilityForest().FromString(example.features.feature['accessibility_trees'].bytes_list.value[i])
            a11y_tree_dict = json.loads(MessageToJson(a11y_tree))

            json_data = {
                'episode_id': episode_id,
                'action': cur_action,
                'action_history': action_history,
                'instruction': instruction,
                'goal': goal,
                'a11y': a11y_tree_dict,
                'reduced_a11y': reduce_a11y_tree(a11y_tree_dict),
                'before_img': img_bytes,
                'dims': dims
            }

            action_history.append(cur_action)  # append the current action to the action history of the next datapoint

            if episode_id in train_episodes:
                processed_data['train'].append(json_data)
            elif episode_id in val_episodes:
                processed_data['val'].append(json_data)
            elif episode_id in test_episodes and episode_id in test_subsplit_episodes:
                processed_data['test'].append(json_data)

    return processed_data


def process_action(action):
    action = json.loads(action)
    action_type = action['action_type']

    action_str = ''
    if action_type == 'open_app':
        action_str = 'Open App ' + action['app_name']
    elif action_type == 'click':
        action_str = 'Click(' + str(action['x']) + ', ' + str(action['y']) + ')'
    elif action_type == 'long_press':
        action_str = 'Long Press(' + str(action['x']) + ', ' + str(action['y']) + ')'
    elif action_type == 'scroll':
        action_str = 'Scroll ' + action['direction']
    elif action_type == 'input_text':
        action_str = 'Type ' + action['text']
    elif action_type == 'navigate_home':
        action_str = 'Navigate Home'
    elif action_type == 'navigate_back':
        action_str = 'Navigate Back'
    elif action_type == 'wait':
        action_str = 'Wait'
    else:
        print("Unrecognized action.")
    return action_str


class AndroidControlBuilder(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def __init__(self, **kwargs):
        super().__init__(**kwargs, dataset_name="android_control")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                "question": datasets.Value("string"),
                'answer': datasets.Value("string"),
                'image': datasets.Image(),
                "ll_instruction": datasets.Value("string"),
                "hl_instruction": datasets.Value("string"),
                "target_action": datasets.Value("string"),
                "target_box": datasets.Value("string"),
                "a11y": datasets.Value("string"),
                "dims": datasets.Value("string"),
            }))

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        to_download = TFRECORD_URLS + DATA_URLS
        files = dl_manager.download(to_download)
        files = {k.split("/")[-1]: v for k, v in zip(to_download, files)}
        processed_data = process_data(files)
        return [
            datasets.SplitGenerator(name=k, gen_kwargs=dict(processed_data=processed_data[k]))
            for k in ["train", "val", "test"]
        ]

    def _generate_examples(self, processed_data):
        for row_id, data in enumerate(processed_data):
            dims = data['dims']
            reduced_a11y = data['reduced_a11y']

            # Map the target action to range of 0-100 with 1 decimal point
            target_action = f"{data['action']}"
            answer = self.map_coordinates_mmolmo(target_action, dims)

            # Find and extract the target bounding box that the target action's coordinates are within, only applicable to Click and Long Press.
            target_box = ''
            if target_action.startswith("Click") or target_action.startswith("Long Press"):
                gt_coords = re.findall(r'\d+', target_action)
                gt_coords = (int(gt_coords[0]), int(gt_coords[1]))
                bbs, bb_centers, bb_sizes, metadata = extract_bbs_from_a11y(data['a11y'], dims)
                target_box = find_gt_box(gt_coords, bb_centers, bb_sizes, bbs, strategy='smallest')
                target_box = self.map_bounding_box(target_box, dims)
            elif target_action.startswith("Open App"):
                app_name = target_action[9:]
                target_box = extract_app_bb(reduced_a11y, search_text='text='+app_name)
                if target_box != '':
                    target_box = self.map_bounding_box(target_box, dims)

            example = {
                "question": data['instruction'],  # repeat of ll_instruction for sake of vqa_preprocessor
                "answer": answer,
                "image": dict(bytes=data['before_img']),
                "ll_instruction": data['instruction'],
                "hl_instruction": data['goal'],
                "target_action": answer,
                "target_box": str(target_box),
                "a11y": str(reduced_a11y),
                "dims": str(dims)
            }

            yield row_id, example

    def map_coordinates_mmolmo(self, action, dims):
        W, H = dims
        def map_value(value, max_value):
            return round((value / max_value) * 100, 1)

        click_match = re.match(r"Click\((\d+), (\d+)\)", action)
        long_press_match = re.match(r"Long Press\((\d+), (\d+)\)", action)

        if click_match:
            x, y = int(click_match.group(1)), int(click_match.group(2))
            new_x = map_value(x, W)
            new_y = map_value(y, H)
            return f"Click({new_x}, {new_y})"

        elif long_press_match:
            x, y = int(long_press_match.group(1)), int(long_press_match.group(2))
            new_x = map_value(x, W)
            new_y = map_value(y, H)
            return f"Long Press({new_x}, {new_y})"

        return action

    def map_bounding_box(self, box, dims):
        xmin, ymin, xmax, ymax = box
        width, height = dims

        # Map each coordinate to the range 0-100 with 1 decimal point
        xmin_mapped = round((xmin / width) * 100, 1)
        ymin_mapped = round((ymin / height) * 100, 1)
        xmax_mapped = round((xmax / width) * 100, 1)
        ymax_mapped = round((ymax / height) * 100, 1)

        return [xmin_mapped, ymin_mapped, xmax_mapped, ymax_mapped]


if __name__ == "__main__":
    builder = AndroidControlBuilder()
    builder.download_and_prepare()
