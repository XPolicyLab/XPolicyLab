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
import functools
import hashlib
import json
from enum import Enum

from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import BLOB, INTEGER, Text

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    DeprecatedDatasetORMBase,
    register_table_mapper,
)
from robo_orchard_lab.dataset.robot.db_orm.md5 import MD5FieldMixin
from robo_orchard_lab.dataset.robot.db_orm.sql_types import SQLStringEnum
from robo_orchard_lab.dataset.robot.db_orm.upgrade import (
    TableUpgradeRegistry,
    Version,
)

__all__ = ["RobotDescriptionFormat", "Robot"]


class RobotDescriptionFormat(str, Enum):
    """The format of the robot description."""

    URDF = "urdf"
    MJCF = "mjcf"


@register_table_mapper
class Robot(DatasetORMBase, MD5FieldMixin["Robot"]):
    """ORM model for a robot in a RoboOrchard dataset."""

    __tablename__ = "robot"
    __version__ = "0.0.1"

    index: Mapped[int] = mapped_column(
        INTEGER, primary_key=True, autoincrement=False
    )
    name: Mapped[str] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    """The description content of the robot.

    This can be in different formats, such as URDF or MJCF.
    """

    content_format: Mapped[RobotDescriptionFormat | None] = mapped_column(
        SQLStringEnum(RobotDescriptionFormat),
        default=None,
    )
    """The format of the robot description."""

    md5: Mapped[bytes] = mapped_column(BLOB(length=16), index=True)

    @classmethod
    @functools.cache
    def md5_content_fields(cls) -> list[str]:
        exclude_keys = ["index", "md5"]
        ret = []
        for key in cls.__table__.columns.keys():
            if key not in exclude_keys:
                ret.append(key)
        return ret

    def calculate_md5(self) -> bytes:
        dict_data = {}
        for field in self.md5_content_fields():
            value = getattr(self, field)
            dict_data[field] = value
        # Serialize the dict to a JSON string with sorted keys
        content_str = json.dumps(dict_data, sort_keys=True)
        return hashlib.md5(content_str.encode("utf-8")).digest()

    @staticmethod
    def query_by_content_with_md5(
        session: Session,
        name: str,
        content: str | None,
        content_format: RobotDescriptionFormat | None,
    ) -> Robot | None:
        """Query a robot by its name and URDF content."""

        return MD5FieldMixin[Robot].query_by_content_with_md5(
            session,
            Robot,
            name=name,
            content=content,
            content_format=content_format,
        )


class RobotDeprecatedVersionNONE(
    DeprecatedDatasetORMBase, MD5FieldMixin["RobotDeprecatedVersionNONE"]
):
    """Deprecated Robot ORM Version(None).

    This class is kept for database upgrade purposes only. Never use it
    in other places!
    """

    __tablename__ = "robot"
    # __version__ = "0.0.1"

    index: Mapped[int] = mapped_column(
        INTEGER, primary_key=True, autoincrement=False
    )
    name: Mapped[str] = mapped_column(Text)
    urdf_content: Mapped[str | None] = mapped_column(Text)
    """The URDF content of the robot.

    In current implementation, we only support URDF format.
    """

    md5: Mapped[bytes] = mapped_column(BLOB(length=16), index=True)


@TableUpgradeRegistry.register_upgrade(
    table_name=Robot.__tablename__,
    from_version=None,
    to_version=Version("0.0.1"),
    from_orm_type=RobotDeprecatedVersionNONE,
)
def upgrade_robot_to_0_0_1(session: Session, row: dict) -> dict:
    """Upgrade a Robot row to version 0.0.1.

    Args:
        session (Session): The database session.
        row (dict): The Robot row to upgrade.

    Returns:
        dict: The upgraded Robot row.
    """
    # In version 0.0.1, we changed urdf_content to content and added
    # content_format field.
    assert session.bind is not None
    row = row.copy()
    old = RobotDeprecatedVersionNONE(**row)
    # For version None, only urdf_content is available. We map it to
    # content field and set content_format to URDF.
    new = Robot(
        index=old.index,
        name=old.name,
        content=old.urdf_content,
        content_format=RobotDescriptionFormat.URDF
        if old.urdf_content is not None
        else None,
    )
    new.update_md5()
    new_dict = {}
    new.column_copy(new, new_dict)
    return new_dict
