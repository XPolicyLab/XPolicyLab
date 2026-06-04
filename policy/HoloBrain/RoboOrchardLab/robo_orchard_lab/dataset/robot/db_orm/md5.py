# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
from __future__ import annotations
from abc import abstractmethod
from typing import Generic, Iterable, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import BLOB

T = TypeVar("T", bound="MD5FieldMixin")


class MD5FieldMixin(Generic[T]):
    md5: Mapped[bytes] = mapped_column(BLOB(length=16), index=True)

    @staticmethod
    def query_by_content_with_md5(
        session: Session, cls: Type[T], **kwargs
    ) -> T | None:
        """Query an instance of the class by its content and MD5 hash.

        This method use the MD5 hash to filter results, and then checks
        if all provided keyword arguments match the attributes of the
        instance.

        """

        md5 = cls(**kwargs).update_md5()
        stmt = select(cls).where(cls.md5 == md5)
        for result in session.execute(stmt).scalars():
            if all(getattr(result, k) == kwargs[k] for k in kwargs):
                return result
        return None

    @classmethod
    @abstractmethod
    def md5_content_fields(cls) -> list[str]:
        raise NotImplementedError(
            "Subclasses must implement the md5_content_fields method."
        )

    @abstractmethod
    def calculate_md5(self) -> bytes:
        """Calculate the MD5 hash based on the content fields."""
        raise NotImplementedError(
            "Subclasses must implement the calculate_md5 method."
        )

    def update_md5(self) -> bytes:
        """Update the MD5 hash based on the content fields."""
        ret = self.calculate_md5()
        if self.md5 != ret:
            self.md5 = ret
        return self.md5


MD5_ORM_TYPE = TypeVar("MD5_ORM_TYPE", bound=MD5FieldMixin)


class MD5ObjCache(Generic[MD5_ORM_TYPE]):
    """A cache for ORM objects based on their MD5 hash and columns."""

    def __init__(self, check_column_names: list[str]) -> None:
        self._cache: dict[bytes, list[MD5_ORM_TYPE]] = {}
        self._check_column_names = check_column_names

    def extend(self, objs: Iterable[MD5_ORM_TYPE]) -> None:
        """Extend the cache with a list of ORM objects."""
        for obj in objs:
            md5_value = obj.md5
            if md5_value not in self._cache:
                self._cache[md5_value] = []
            if self.find(obj) is None:
                self._cache[md5_value].append(obj)

    def find(self, obj: MD5_ORM_TYPE) -> MD5_ORM_TYPE | None:
        """Find an ORM object in the cache that matches the given object."""
        if obj.md5 not in self._cache:
            return None

        for exist_obj in self._cache[obj.md5]:
            if all(
                getattr(obj, col) == getattr(exist_obj, col)
                for col in self._check_column_names
            ):
                return exist_obj
        return None
