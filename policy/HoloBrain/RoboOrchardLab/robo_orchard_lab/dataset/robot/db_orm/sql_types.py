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

import base64
import json
import uuid
from enum import Enum
from typing import Any, Generic, Optional, Type, TypeVar

from robo_orchard_core.datatypes.uuid import UUID64
from sqlalchemy import Dialect
from sqlalchemy.sql.sqltypes import TypeEngine
from sqlalchemy.types import (
    BIGINT,
    BINARY,
    BLOB as DEFAULT_BLOB,
    TEXT,
    String,
    TypeDecorator,
)

ENUM_T = TypeVar("ENUM_T", bound=Enum)

__all__ = [
    "SQLStringEnum",
    "BlobUuid64",
    "BigIntUuid64",
    "BlobUuid128",
    "HexUuid128",
    "Base64EncodedBLOB",
    "Base64JSONEncodedDict",
]


class SQLStringEnum(TypeDecorator, Generic[ENUM_T]):
    """SQLAlchemy type for String Enum.

    This class maps Enum to string in the database.

    """

    impl = String

    cache_ok = True
    """safe to be used as part of a cache key."""

    def __init__(self, enum: Type[ENUM_T], length: int = 12, *args, **kwargs):
        """Initialize the SQLStringEnum."""
        super().__init__(*args, **kwargs)
        self._enum_type = enum
        self._length = length

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        return dialect.type_descriptor(String(length=self._length))

    def process_bind_param(
        self, value: Optional[ENUM_T], dialect: Dialect
    ) -> Optional[str]:
        """Encode the value to be stored in the database."""
        if value is None:
            return None
        return value.value

    def process_result_value(
        self, value: Optional[str], dialect: Dialect
    ) -> Optional[ENUM_T]:
        """Decode the value from the database."""
        if value is None:
            return None
        return self._enum_type._value2member_map_[value]  # type: ignore


class BlobUuid64(TypeDecorator[UUID64]):
    """UUID type for SQLAlchemy.

    This class maps UUID64 to BINARY(8) in the database.
    """

    impl = BINARY

    cache_ok = True
    """safe to be used as part of a cache key."""

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        return dialect.type_descriptor(BINARY(8))

    def process_bind_param(
        self, value: Optional[UUID64], dialect: Dialect
    ) -> Optional[bytes]:
        """Encode the value to be stored in the database."""
        if not value:
            return None
        return value.bytes

    def process_result_value(
        self, value: Optional[bytes], dialect: Dialect
    ) -> Optional[UUID64]:
        """Decode the value from the database."""
        if not value:
            return None
        return UUID64(bytes=value)


class BigIntUuid64(TypeDecorator[UUID64]):
    """UUID type for SQLAlchemy.

    This class maps UUID64 to BIGINT in the database.

    """

    impl = BIGINT

    cache_ok = True
    """safe to be used as part of a cache key."""

    def process_bind_param(
        self, value: Optional[UUID64], dialect: Dialect
    ) -> Optional[int]:
        """Encode the value to be stored in the database."""
        if value is None:
            return None

        ret = value.signed_int
        return ret

    def process_result_value(
        self, value: Optional[int], dialect: Dialect
    ) -> Optional[UUID64]:
        """Decode the value from the database."""
        if value is None:
            return None

        ret = UUID64(int=value, signed=True)
        return ret


class BlobUuid128(TypeDecorator[uuid.UUID]):
    """UUID type for SQLAlchemy.

    This class maps UUID128 to BINARY(16) in the database.

    """

    impl = DEFAULT_BLOB

    # Necessary caching flag since SQLAlchemy version 1.4.14
    # https://docs.sqlalchemy.org/en/14/core/custom_types.html#sqlalchemy.types.TypeDecorator
    cache_ok = True
    """safe to be used as part of a cache key."""

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine:
        """Inherited, see superclass."""
        return dialect.type_descriptor(DEFAULT_BLOB(16))

    def process_bind_param(
        self, value: Optional[uuid.UUID], dialect: Dialect
    ) -> Optional[bytes]:
        """Encode the value to be stored in the database."""
        if not value:
            return None

        return value.bytes

    def process_result_value(
        self, value: Optional[bytes], dialect: Dialect
    ) -> Optional[uuid.UUID]:
        """Decode the value from the database."""
        if not value:
            return None
        return uuid.UUID(bytes=value)


class HexUuid128(TypeDecorator[uuid.UUID]):
    """UUID type for SQLAlchemy.

    This class maps UUID128 to String(32) as hex in the database.

    """

    impl = String(32)

    # Necessary caching flag since SQLAlchemy version 1.4.14
    # https://docs.sqlalchemy.org/en/14/core/custom_types.html#sqlalchemy.types.TypeDecorator
    cache_ok = True
    """safe to be used as part of a cache key."""

    def process_bind_param(
        self, value: Optional[uuid.UUID], dialect: Dialect
    ) -> Optional[str]:
        """Encode the value to be stored in the database."""
        if not value:
            return None

        return value.hex

    def process_result_value(
        self, value: Optional[str], dialect: Dialect
    ) -> Optional[uuid.UUID]:
        """Decode the value from the database."""
        if not value:
            return None
        return uuid.UUID(hex=value)


class Base64EncodedBLOB(TypeDecorator[bytes]):
    """Base64 encoded BLOB type.

    This type store base64 encoded BLOB as text in database.
    """

    impl = TEXT
    cache_ok = True
    """safe to be used as part of a cache key."""

    def __init__(self, length: int | None = None, **kwargs):
        super().__init__(length=length, **kwargs)
        self.length = length

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "doris":
            return TEXT()
        return super().load_dialect_impl(dialect)

    def process_bind_param(self, value: bytes | None, dialect) -> str | None:
        """Encode the value to be stored in the database."""
        if value is not None:
            return base64.b64encode(value).decode("utf-8")

    def process_result_value(self, value: str | None, dialect) -> bytes | None:
        """Decode the value from the database."""
        if value is not None:
            return base64.b64decode(value)


class Base64JSONEncodedDict(TypeDecorator[dict]):
    """Json encoded dict type.

    Use string to store base64 encoded json dict
    """

    impl = TEXT
    cache_ok = True
    """safe to be used as part of a cache key."""

    def __init__(self, length: int | None = None, **kwargs):
        super().__init__(length=length, **kwargs)
        self.length = length

    def process_bind_param(self, value: dict | None, dialect) -> str | None:
        """Encode the value to be stored in the database."""
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("value must be a dict")
            return base64.b64encode(
                json.dumps(value, sort_keys=True).encode("utf-8")
            ).decode("utf-8")

    def process_result_value(self, value: str | None, dialect) -> dict | None:
        """Decode the value from the database."""
        if value is not None:
            return json.loads(base64.b64decode(value))
