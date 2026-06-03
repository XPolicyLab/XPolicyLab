import random
import numpy as np

class MultiHDF5VLADataset:
    """
    A "multiplexed" Dataset used to randomly sample among multiple HDF5VLADataset sub-datasets.
    Each time get_item(...) is called, it randomly decides which sub-dataset to use, and then returns a sample from it.
    """

    def __init__(self, dataset_list, dataset_weights=None):
        """
        Args:
            dataset_list (List): A list of sub-datasets to be mixed, e.g., [rh20t_dataset, agilex_dataset].
            dataset_weights (List, optional): Sampling weights for each sub-dataset, corresponding one-to-one with dataset_list.
                If None, uniform sampling is used by default.
                For example, [0.7, 0.3] means there is a 70% chance to choose the first dataset and 30% for the second.
        """
        assert len(dataset_list) > 0, "dataset_list cannot be empty"
        self.dataset_list = dataset_list
        self.num_datasets = len(dataset_list)

        if dataset_weights is None:
            # If not specified, each sub-dataset will have the same sampling probability
            self.dataset_weights = [1.0 / self.num_datasets] * self.num_datasets
        else:
            # Ensure sum(weights) = 1.0, etc.; or normalize them
            total_w = sum(dataset_weights)
            self.dataset_weights = [w / total_w for w in dataset_weights]

        # Simple check
        for ds in self.dataset_list:
            if not hasattr(ds, "get_item"):
                raise AttributeError("Sub-dataset does not have a get_item(...) method")
        print("MultiHDF5VLADataset: number of sub-datasets =", self.num_datasets, 
              "weights =", self.dataset_weights)

    def __len__(self):
        """
        Return the sum of the lengths of the sub-datasets as an example.
        """
        return sum(len(ds) for ds in self.dataset_list)

    def get_item(self, index=None):
        """
        Each time this is called, it first randomly chooses one sub-dataset based on self.dataset_weights,
        then randomly selects an index from that sub-dataset (or let it handle the random internally),
        and returns the result of its get_item(...).
        """
        # First choose a sub-dataset based on the weights
        chosen_ds_idx = np.random.choice(self.num_datasets, p=self.dataset_weights)
        chosen_ds = self.dataset_list[chosen_ds_idx]

        # Randomly select an index within the chosen sub-dataset
        ds_length = len(chosen_ds)
        random_idx = random.randint(0, ds_length - 1)

        # Get data
        return chosen_ds.get_item()