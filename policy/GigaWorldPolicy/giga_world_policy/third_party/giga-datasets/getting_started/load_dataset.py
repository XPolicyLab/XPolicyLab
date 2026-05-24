import tyro

from giga_datasets import load_dataset


def main(data_path: str = './giga_data'):
    # Load the dataset from the giga_data directory
    dataset = load_dataset(data_path)
    data_dict = dataset[0]
    print('Dataset size:', len(dataset))
    print('First item in dataset:', data_dict)
    # Access specific data fields
    print(f'The dataset has {dataset.datasets} fields.')
    # iterate through the dataset and print each item
    for i in range(len(dataset)):
        data_dict = dataset[i]
        print(f'Data item {i}:', data_dict)


if __name__ == '__main__':
    tyro.cli(main)
