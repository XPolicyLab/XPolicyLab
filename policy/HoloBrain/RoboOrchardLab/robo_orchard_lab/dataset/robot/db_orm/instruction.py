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
import hashlib
import json

from sqlalchemy.orm import Mapped, Session, mapped_column
from sqlalchemy.types import BIGINT, BLOB, JSON, TEXT

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    register_table_mapper,
)
from robo_orchard_lab.dataset.robot.db_orm.md5 import MD5FieldMixin
from robo_orchard_lab.dataset.robot.db_orm.upgrade import (
    TableUpgradeRegistry,
    Version,
)

__all__ = ["Instruction"]


@register_table_mapper
class Instruction(DatasetORMBase, MD5FieldMixin["Instruction"]):
    __tablename__ = "instruction"
    __version__ = "0.0.1"

    index: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=False
    )
    name: Mapped[str | None] = mapped_column(TEXT)

    json_content: Mapped[dict | None] = mapped_column(JSON)

    md5: Mapped[bytes] = mapped_column(BLOB(length=16), index=True)

    @classmethod
    def md5_content_fields(cls) -> list[str]:
        exclude_keys = ["index", "md5"]
        ret = []
        for key in cls.__table__.columns.keys():
            if key not in exclude_keys:
                ret.append(key)
        return ret

    def calculate_md5(self) -> bytes:
        content_str = (
            json.dumps(self.json_content, sort_keys=True)
            if self.json_content
            else ""
        )
        combined_str = f"{self.name}{content_str}".encode("utf-8")
        return hashlib.md5(combined_str).digest()

    @staticmethod
    def query_by_content_with_md5(
        session: Session, name: str | None, json_content: dict | None
    ) -> Instruction | None:
        """Query a robot by its name and URDF content."""

        return MD5FieldMixin[Instruction].query_by_content_with_md5(
            session=session,
            cls=Instruction,
            name=name,
            json_content=json_content,
        )


@TableUpgradeRegistry.register_upgrade(
    table_name=Instruction.__tablename__,
    from_version=None,
    to_version=Version("0.0.1"),
    from_orm_type=Instruction,
)
def upgrade_instruction_to_0_0_1(session: Session, row: dict) -> dict:
    """Upgrade an Instruction row to version 0.0.1.

    Since this is the initial version, no changes are made.

    Args:
        session (Session): The database session.
        row (dict): The Instruction row to upgrade.

    Returns:
        dict: The upgraded Instruction row.
    """
    return row
