import dataclasses
from collections.abc import Sequence
from typing import Any, Dict, Protocol, SupportsIndex, TypeAlias, TypeVar, Union, runtime_checkable

DataDict: TypeAlias = Dict[str, Any]

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """
        ...


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class TransformedDataset(Dataset[DataDict]):
    def __init__(self, dataset: Dataset, transforms: Sequence[DataTransformFn], num_frames: int):
        self._dataset = dataset
        self._transform = compose(transforms)
        self.num_frames = num_frames

    def __getitem__(self, index: SupportsIndex) -> DataDict:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: Dict
    # 要进行normalize的键名，可以是字符串或字符串列表，默认是"action"
    key: str | list[str] = "action"

    def __call__(self, data: DataDict) -> DataDict:
        # 确保key是列表格式
        keys = [self.key] if isinstance(self.key, str) else self.key

        for k in keys:
            if k in data and k in self.norm_stats:
                data[k] = self._normalize(data[k], self.norm_stats[k])

        return data

    def _normalize(self, x, stats):
        return (x - stats["mean"]) / (stats["std"] + 1e-6)


def make_conversation(prompt: str, conversation_type: int = 0) -> str:
    if conversation_type == 0:
        return f"What action should the robot take to {prompt}?"
    elif conversation_type == 1:
        return f"What action should the robot take to {prompt}?"
    elif conversation_type == 2:
        return f"{prompt}"
    else:
        print(f"Conversation Type {conversation_type} is not implemented.")
        raise NotImplementedError()
