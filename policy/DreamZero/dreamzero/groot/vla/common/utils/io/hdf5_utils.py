import h5py
import numpy as np
from pydantic import BaseModel


def hdf5_save(data: BaseModel | dict, group: h5py.Group) -> None:
    """Recursively save Pydantic model or dict to HDF5 group."""
    if isinstance(data, BaseModel):
        # Convert to dict and exclude None values
        data_dict = data.model_dump(mode="python", exclude_none=True)
    else:
        data_dict = data

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, (BaseModel, dict)):
            subgroup = group.create_group(key)
            hdf5_save(value, subgroup)
        else:
            # For primitive types, convert to numpy array
            try:
                group.create_dataset(key, data=np.array(value))
            except TypeError:
                raise ValueError(f"Unsupported type: {type(value)} for key: {key}")


def hdf5_load(group: h5py.Group) -> dict:
    """Recursively load HDF5 group to Pydantic model or dict."""
    data_dict = {}
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            data_dict[key] = value[()]
        elif isinstance(value, h5py.Group):
            data_dict[key] = hdf5_load(value)
    return data_dict


def hdf5_is_subset(this: h5py.Group, other: h5py.Group, verbose: bool = False) -> bool:
    """Check if this HDF5 group is a subset of another HDF5 group."""
    for key, value in this.items():
        if key not in other:
            if verbose:
                print(f"Key {key} not in other")
            return False
        elif isinstance(value, h5py.Group):
            if not isinstance(other[key], h5py.Group):
                if verbose:
                    print(f"Key {key} is not a group in other")
                return False
            if not hdf5_is_subset(value, other[key], verbose):
                if verbose:
                    print(f"Key {key} is not a subset of other")
                return False
        elif isinstance(value, h5py.Dataset):
            if not isinstance(other[key], h5py.Dataset):
                if verbose:
                    print(f"Key {key} is not a dataset in other")
                return False
            if not np.array_equal(value, other[key]):
                if verbose:
                    print(f"Key {key} is not equal in other")
                return False
        elif isinstance(value, h5py.Datatype):
            if not isinstance(other[key], h5py.Datatype):
                if verbose:
                    print(f"Key {key} is not a datatype in other")
                return False
            if value != other[key]:
                if verbose:
                    print(f"Key {key} is not equal in other")
                return False
        else:
            # try to compare
            if value != other[key]:
                if verbose:
                    print(f"Key {key} is not equal in other")
                return False
    return True


def hdf5_is_equal(this: h5py.Group, other: h5py.Group, verbose: bool = False) -> bool:
    """Check if this HDF5 group is equal to another HDF5 group."""
    return hdf5_is_subset(this, other, verbose) and hdf5_is_subset(other, this, verbose)
