# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import String

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    register_table_mapper,
)

__all__ = ["TableInfo"]


@register_table_mapper
class TableInfo(DatasetORMBase):
    """The table information.

    This table is used to store the information of the tables in the database.
    """

    __version__ = "0.0.1"
    __tablename__ = "table_info"

    @property
    def pk_str(self) -> str:
        """Return the primary key string."""
        pk_dict = dict()
        self.pk_copy(self, pk_dict)
        return "_".join([f"{v}" for k, v in pk_dict.items()])

    def __repr__(self) -> str:
        return (
            f"TableInfo("
            f"table_name={self.table_name}, "
            f"table_version={self.table_version}, "
            f"app_version={self.app_version},"
            f"creation_statement={self.creation_statement}"
            f")"
        )

    table_name: Mapped[str] = mapped_column(
        String(32),
        primary_key=True,
        comment="The name of the table.",
    )

    table_version: Mapped[str] = mapped_column(
        String(16),
        comment="The version of the table.",
    )

    app_version: Mapped[str] = mapped_column(
        String(48),
        comment="The version of the robo_orchard_lab used to create tables.",
    )

    creation_statement: Mapped[str] = mapped_column(
        String(2048),
        comment="The creation statement of the table.",
    )

    __table_args__ = (
        {"comment": "Record the table information in the database"},
    )
