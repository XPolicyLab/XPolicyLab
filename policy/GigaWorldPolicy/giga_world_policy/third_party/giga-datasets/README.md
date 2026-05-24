<p align="center">
<img width="55%" alt="GigaDatasets" src="./docs/source/imgs/logo.png?raw=true">
</p>
<h3 align="center">
A Unified and Lightweight Framework for Data Curation, Evaluation and Visualization
</h3>
<p align="center">
    | <a href="#-installation">Quick Start</a>
    | <a href="#-contributing">Contributing</a>
    | <a href="#-license">License</a>
    | <a href="#-citation">Citation</a> |
</p>

## ✨ Introduction

GigaDatasets is a unified and lightweight framework for data curation, evaluation and visualization. Designed to make handling massive datasets simple, efficient, and consistent.

<details open>
<summary>Major features</summary>

- 🔍 **Unified Workflow**: Unify all steps from data curation and packaging to loading, evaluation, and visualization.
- ⚡ **Lightweight and Easy to Use**: Simple pip/source install `pip3 install giga-datasets`, one line of code for data loading `dataset = load_dataset(data_path)`, one line of code for data evaluation `eval_results = FIDEvaluator(datasets)(pred_results)`.
- 🗂️ **Multi-format and Multi-structure Data Support**: File, LMDB, Pickle, and LeRobot datasets with flexible loading. Unified support for images, videos, 2D/3D boxes, 2D/3D points, and other structured data.
- 🚀 **Efficient Processing**: Optimized for speed and memory, suitable for large-scale data processing needs.

</details>

## ⚡ Installation

GigaDatasets can be installed from PyPi and has to be installed in a virtual environment (venv or conda for instance)

```bash
pip3 install giga-datasets
```

or you can install directly from source for the latest updates:

```bash
conda create -n giga_datasets python=3.11.10
conda activate giga_datasets
git clone https://github.com/open-gigaai/giga-datasets.git
cd giga-datasets
pip3 install -e .
```

## 🚀 Usage

We provide accessible demo data and Jupyter notebooks in [getting_started](getting_started). Utility scripts can
be found in the [scripts](scripts) folder.

### 1. load dataset

There is a simple way to load datasets using the `load_dataset` function from the `giga_datasets` library.
We provide a demo dataset in the `giga_data` directory for you to try out.
Here is a quick example, and the full code is available [here](getting_started/load_dataset.py):

```python
from giga_datasets import load_dataset

dataset = load_dataset('./getting_started/giga_data')
data_dict = dataset[0]
print('Dataset size:', len(dataset))
print('First item in dataset:', data_dict)
```

The `giga_data` directory contains the following structure:

```
giga_data/
├── config.json          # Configuration file describing the dataset
├── labels/              # Directory containing label files
│   ├── config.json      # Additional configuration for labels
│   ├── data.pkl         # Serialized label data
├── images/              # Directory containing image files
│   ├──config.json       # Additional configuration for images
│   ├──data.mdb          # Lmdb format for images
├   ├──lock.mdb
```

The `config.json` file in the `giga_data` directory contains the following structure:

```
{
    "_class_name": "Dataset",
    "config_paths": [
        "labels/config.json",
        "images/config.json"
    ]
}
```

This file specifies:

- \_class_name: Indicates the class type used for the dataset, which is Dataset in this case.
- config_paths: Lists paths to additional configuration files for specific components of the dataset, such as labels/config.json and images/config.json.

### 2. package dataset

For an unstructured dataset, you can use the Writer classes (including `PklWriter`, `FileWriter` and `LmdbWriter` to package
your data into a structured format. Below is an example of how to package a dataset consisting of images and labels.

The `raw_data` directory contains the following structure:

```
raw_data/
├── 0.json               # Annotation file for image 0
├── 0.png                # Image file 0
├── 1.json               # Annotation file for image 1
├── 1.png                # Image file 1
├── ...
```

You can run the following python code to package the dataset, the full code is available [here](getting_started/pack_images.py):

```python
image_paths = utils.list_dir(image_dir, recursive=True, exts=['.png', '.jpg', '.jpeg'])
label_writer = PklWriter(os.path.join(save_dir, 'labels'))
image_writer = LmdbWriter(os.path.join(save_dir, 'images'))
for idx in tqdm(range(len(image_paths))):
    label_path = image_paths[idx].replace('.png', '.json')
    label_dict = json.load(open(label_path))
    label_dict['data_index'] = idx
    label_writer.write_dict(label_dict)
    image_writer.write_image(idx, image_paths[idx])
label_writer.write_config()
image_writer.write_config()
label_writer.close()
image_writer.close()
label_dataset = load_dataset(os.path.join(save_dir, 'labels'))
image_dataset = load_dataset(os.path.join(save_dir, 'images'))
dataset = Dataset([label_dataset, image_dataset])
dataset.save(save_dir)
```

We supports packaging and reading different data formats. In addition to packaging images, we also provide an [example](getting_started/pack_videos.py) of packaging video data, where we store the video's metadata.

```python
# package video samples in the input directory to the output directory
python getting_started/pack_videos.py --video_dir /path/to/your/raw_videos --save_dir ./giga_videos

# if you want to package videos into lmdb format for better read performance
python getting_started/pack_videos.py --video_dir /path/to/your/raw_videos --save_dir ./giga_videos --pack-lmdb

# if you want to package samples, but not copy the video files and only store the metadata and absolute paths
python getting_started/pack_videos.py --video_dir /path/to/your/raw_videos --save_dir ./giga_videos --only_path
```

### 3. add new field

In models' training or inference, a sample is often represented as a dictionary with multiple fields. Our framework is designed to be easily extensible to accommodate new data fields.
Below is an example of how to add canny maps as a new field:

```python
python getting_started/add_new_filed.py --data_dir getting_started/giga_data
```

### Additional Usage Examples

> **Note:** More usage examples and feature documentation will be added in future updates—stay tuned!

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

```bibtex
@misc{gigaai2025gigadatasets,
    author = {GigaAI},
    title = {GigaDatasets: A Unified and Lightweight Framework for Data Curation, Evaluation and Visualization},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/open-gigaai/giga-datasets}}
}
```
