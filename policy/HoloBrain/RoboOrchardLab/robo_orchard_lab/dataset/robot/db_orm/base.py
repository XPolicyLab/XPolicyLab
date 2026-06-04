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


import functools
import pickle
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Type, TypeVar, Union

from sqlalchemy import inspect, select
from sqlalchemy.orm import DeclarativeBase, Mapper, Session, scoped_session
from typing_extensions import Self

__all__ = [
    "DatasetORMBase",
    "ColumnValueUnmatchedError",
    "PrimaryKeyMixin",
    "register_table_mapper",
    "DeprecatedDatasetORMBase",
]

TABLE_NAME2Mapper: Dict[str, Mapper[Any]] = dict()
ALL_TABLES: list[Type[DeclarativeBase]] = []

SessionType = Union[Session, scoped_session]
ORM_T = TypeVar("ORM_T", bound=DeclarativeBase)


def register_table_mapper(cls):
    """Register the table mapper to the table name."""
    if cls.__mapper__ != cls.__mapper__.base_mapper:
        # special case when cls share table with its base class.
        # we use base mapper to represent the table.
        if (
            cls.__mapper__.class_.__table__.name
            == cls.__mapper__.base_mapper.class_.__table__.name
        ):
            TABLE_NAME2Mapper[cls.__table__.name] = cls.__mapper__.base_mapper
            return

    TABLE_NAME2Mapper[cls.__table__.name] = cls.__mapper__
    ALL_TABLES.append(cls)
    return cls


class ColumnValueUnmatchedError(Exception):
    pass


@functools.total_ordering
class PrimaryKeyMixin:
    """Mixin class for classes that have primary keys."""

    def __lt__(self, other: "PrimaryKeyMixin") -> bool:
        """Compare two objects by their primary keys.

        If the two objects are of different types, compare by class name.
        """
        if type(self) is not type(other):
            return type(self).__name__ < type(other).__name__
        cmp_keys = type(self).primary_key_names()
        for k in cmp_keys:
            a = getattr(self, k)
            b = getattr(other, k)
            if a != b:
                return a < b
        return False

    @classmethod
    def pk_bytes(cls, obj: Any) -> bytes:
        """Get the primary key as bytes."""
        pk_dict = {}
        cls.pk_copy(obj, pk_dict)
        return pickle.dumps(pk_dict)

    @classmethod
    def pk_scalar(cls, session: SessionType, obj: Any) -> Optional[Any]:
        """Query an object by its primary key and return a scalar value.

        Args:
            session (SessionType): The session to query.
            obj (Any): The object to query.

        """
        stmt = select(cls).where(cls.pk_equal(cls, obj))
        return session.scalar(stmt)

    @classmethod
    def pk_equal(cls, lhs: Any, rhs: Any) -> Any:
        """Whether two objects are equal by their keys.

        Args:
            lhs (Any): The left hand side object.
            rhs (Any): The right hand side object.

        Returns:
            Any: The equality expression. Can be boolean or a SQL expression.

        """
        if lhs is None or rhs is None:
            raise ValueError("lhs and rhs must not be None")
        primary_keys = [t.name for t in cls.__mapper__.primary_key]  # type: ignore
        if len(primary_keys) == 0:
            raise ValueError("No primary key found")

        ret = getattr(lhs, primary_keys[0]) == getattr(rhs, primary_keys[0])
        for key in primary_keys[1:]:
            ret = (ret) & (getattr(lhs, key) == getattr(rhs, key))

        return ret

    @classmethod
    def pk_copy(cls, src: Any, dst: Any):
        """Copy primary keys from rhs to lhs.

        Args:
            src (Any): The source object.
            dst (Any): The destination object.

        """
        if src is None or dst is None:
            raise ValueError("src and dst must not be None")
        primary_keys = cls.primary_key_names()
        if len(primary_keys) == 0:
            raise ValueError("No primary key found")
        for key in primary_keys:
            if isinstance(dst, (dict, OrderedDict)):
                dst[key] = getattr(src, key)
            else:
                setattr(dst, key, getattr(src, key))

    @classmethod  # type: ignore
    def primary_key_names(
        cls, names_not_in: Optional[list[str]] = None
    ) -> list[str]:
        if names_not_in is None:
            return [t.name for t in cls.__mapper__.primary_key]  # type: ignore
        else:
            return [
                t.name
                for t in cls.__mapper__.primary_key  # type: ignore
                if t.name not in names_not_in
            ]

    @classmethod
    def pk_value_in(cls, instances: Iterable[Any]) -> Any:
        """Get the where condition for objects of given class and pk.

        Expression of (cls.key1, cls.key2) in ((inst.key1, inst.key2), ...).

        Args:
            instances (List[Any]): The list of objects to check.

        Returns:
            Any: The equality expression. Can be boolean or a SQL expression.

        """

        return cls.value_in(cls.primary_key_names(), instances)

    @classmethod
    def value_in(cls, keys: list[str], instances: Iterable[Any]) -> Any:
        """Get the where condition for objects of given class and keys.

        Get the condition of:
            cls.key1 == inst1.key1 and cls.key2 == inst1.key2...
            or cls.key1 == inst2.key1 and cls.key2 == inst2.key2...

        Args:
            keys (List[str]): The list of keys to check.
            instances (List[Any]): The list of objects to check.

        Returns:
            Any: The equality expression. Can be boolean or a SQL expression.

        """
        if len(keys) == 0:
            raise ValueError("keys cannot be empty list!")

        def value_eq(inst: Any):
            ret = getattr(cls, keys[0]) == getattr(inst, keys[0])
            for k in keys[1:]:
                ret &= getattr(cls, k) == getattr(inst, k)
            return ret

        where_cond = None
        for i, inst in enumerate(instances):
            if i == 0:
                where_cond = value_eq(inst)
            else:
                where_cond |= value_eq(inst)

        if where_cond is None:
            raise ValueError("input instances cannot be empty iterator!")

        return where_cond


class ORMBase(PrimaryKeyMixin, DeclarativeBase):
    """Base class for all ORM classes working with all db engines."""

    __version__: str

    @classmethod
    def column_equal(
        cls,
        lhs: dict | DeclarativeBase,
        rhs: dict | DeclarativeBase,
        skip_left_null=False,
    ) -> bool:
        """Whether two objects are equal by their columns.

        Args:
            lhs (Any): The left hand side object.
            rhs (Any): The right hand side object.
            skip_left_null (bool): Whether to skip null values in lhs.
                Defaults to False.
        """
        if lhs is None or rhs is None:
            raise ValueError("lhs and rhs must not be None")
        for key in cls.__table__.columns.keys():
            lhs_v = getattr(lhs, key)
            if skip_left_null and lhs_v is None:
                continue
            if getattr(lhs, key) != getattr(rhs, key):
                return False
        return True

    @classmethod
    def column_copy(
        cls, src: dict | DeclarativeBase, dst: dict | DeclarativeBase
    ):
        """Copy columns from src to dst."""
        if src is None or dst is None:
            raise ValueError("src and dst must not be None")
        for key in cls.__table__.columns.keys():
            if isinstance(dst, dict):
                dst[key] = getattr(src, key)
            else:
                setattr(dst, key, getattr(src, key))

    def clone_to_session(
        self, session: SessionType, with_relationships=False
    ) -> Any:
        """Clone the object and assign relationships to a new session."""
        column_kv = self.get_column_kv(with_relationships=with_relationships)
        for k, v in column_kv.items():
            # if v is relation ship, clone it.
            # we only support one-to-one relationship
            if isinstance(v, DatasetORMBase):
                new_v_pk = {}
                v.pk_copy(v, new_v_pk)
                column_kv[k] = session.get(type(v), new_v_pk)
        ret = type(self)(**column_kv)
        self._bind_instance(ret)  # type: ignore
        return ret

    def get_column_kv(
        self,
        with_relationships: bool = False,
        keep_null: bool = False,
        pk_prefix="",
    ) -> Dict[str, Any]:
        """Get all column key-value pairs.

        Args:
            with_relationships (bool): Whether to include relationships.
                Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary of column key-value pairs.

        """

        ret = {}
        for k, col in type(self).__table__.columns.items():
            dict_key = f"{pk_prefix}{k}" if col.primary_key else k
            if (v := getattr(self, k)) is not None:
                ret[dict_key] = v
            else:
                if keep_null:
                    ret[dict_key] = None

        if with_relationships and (not inspect(self).detached):
            # if with_relationships:
            # get all relationships. We only support one one mapping
            for k in type(self).__mapper__.relationships.keys():
                if (v := getattr(self, k)) is not None:
                    if isinstance(v, (list, tuple, set, dict)):
                        if len(v) == 0:
                            continue
                        raise ValueError(
                            f"Non-One-One Relationship {k} "
                            f"of {type(self)} is not supported."
                        )
                    ret[k] = v
        return ret

    def init_default_value(self):
        """Initialize default values for all columns if applicable.

        Note that this function does not initialize default values for
        autoincrement columns.

        """
        for k in type(self).__table__.columns.keys():
            if (
                getattr(self, k) is None
                and type(self).__table__.columns[k].default is not None
            ):
                default_v = type(self).__table__.columns[k].default
                if default_v.is_callable:
                    setattr(self, k, default_v.arg(self))
                else:
                    setattr(self, k, default_v.arg)

    def get_remote(
        self, session: SessionType, check_equal: bool = True
    ) -> Optional[Self]:
        """Get the remote object in the given session.

        Args:
            session (SessionType): The session to query.
            check_equal (bool): Whether to check if the remote object is equal
                to self. Defaults to False. If True, the function will raise
                ValueError if the remote object is not equal to self.

        """
        stmt = select(type(self)).where(type(self).pk_equal(type(self), self))
        ret = session.execute(stmt).scalar_one_or_none()
        if ret is not None and check_equal:
            if not type(self).column_equal(self, ret, skip_left_null=True):
                raise ColumnValueUnmatchedError(
                    "Remote object is not equal to self"
                )

        return ret

    def __repr__(self) -> str:
        """Return a string representation of the object."""
        return f"{type(self).__name__}({self.get_column_kv(keep_null=True)})"


class DatasetORMBase(ORMBase, DeclarativeBase):
    """Base class for ORM classes working with RoboOrchard datasets."""

    pass


class DeprecatedDatasetORMBase(ORMBase, DeclarativeBase):
    """Deprecated ORM classes that should not be used with new datasets."""

    pass
