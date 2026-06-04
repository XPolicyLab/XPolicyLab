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

from sqlalchemy import ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from sqlalchemy.types import BIGINT, INTEGER, JSON

from robo_orchard_lab.dataset.robot.db_orm.base import (
    DatasetORMBase,
    DeprecatedDatasetORMBase,
    register_table_mapper,
)
from robo_orchard_lab.dataset.robot.db_orm.robot import Robot
from robo_orchard_lab.dataset.robot.db_orm.task import Task
from robo_orchard_lab.dataset.robot.db_orm.upgrade import (
    TableUpgradeRegistry,
    Version,
)

__all__ = ["Episode"]


@register_table_mapper
class Episode(DatasetORMBase):
    """ORM model for an episode in a RoboOrchard dataset."""

    __tablename__ = "episode"
    __version__ = "0.0.2"

    index: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=False
    )
    """The unique index of the episode."""

    robot_index: Mapped[int | None] = mapped_column(
        INTEGER, ForeignKey(Robot.index), default=None, index=True
    )
    task_index: Mapped[int | None] = mapped_column(
        INTEGER, ForeignKey(Task.index), default=None, index=True
    )
    prev_episode_index: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey(f"{__tablename__}.index"), default=None, index=True
    )
    """The episode index of the previous episode.
    This is used to link episodes together in a sequence.
    """

    dataset_begin_index: Mapped[int] = mapped_column(BIGINT)
    """The index of the first dataset item in this episode.

    Can be -1 if not available. User should fix it after all
    processings are done.
    """

    frame_num: Mapped[int] = mapped_column(INTEGER)

    # ---------- added from version 0.0.2  BEGIN---------------
    truncated: Mapped[bool | None] = mapped_column(default=None)
    """Whether the episode was truncated (not ended naturally)."""
    success: Mapped[bool | None] = mapped_column(default=None)
    """Whether the episode was successful."""
    info: Mapped[dict | None] = mapped_column(JSON, default=None)
    """Additional info about the episode. Should be a JSON-serializable
    dictionary. """
    # ---------- added from version 0.0.2  END---------------

    @declared_attr
    def robot(cls) -> Mapped[Robot | None]:
        return relationship(
            "Robot",
            backref=cls.__tablename__,
            foreign_keys=[cls.robot_index],  # type: ignore
        )

    @declared_attr
    def task(cls) -> Mapped[Task | None]:
        return relationship(
            "Task",
            backref=cls.__tablename__,
            foreign_keys=[cls.task_index],  # type: ignore
        )

    @declared_attr
    def prev_episode(cls) -> Mapped[Episode | None]:
        return relationship(
            "Episode",
            back_populates="next_episode",
            remote_side=[cls.index],  # type: ignore
        )

    @declared_attr
    def next_episode(cls) -> Mapped[list[Episode]]:
        return relationship(
            "Episode",
            back_populates="prev_episode",
        )


class EpisodeDeprecatedVersion1(
    DeprecatedDatasetORMBase,
):
    """Deprecated ORM model for Episode version 0.0.1."""

    __tablename__ = Episode.__tablename__
    __version__ = "0.0.1"

    index: Mapped[int] = mapped_column(
        BIGINT, primary_key=True, autoincrement=False
    )
    """The unique index of the episode."""

    robot_index: Mapped[int | None] = mapped_column(
        INTEGER, ForeignKey(Robot.index), default=None, index=True
    )
    task_index: Mapped[int | None] = mapped_column(
        INTEGER, ForeignKey(Task.index), default=None, index=True
    )
    prev_episode_index: Mapped[int | None] = mapped_column(
        BIGINT, ForeignKey(f"{__tablename__}.index"), default=None, index=True
    )
    """The episode index of the previous episode.
    This is used to link episodes together in a sequence.
    """

    dataset_begin_index: Mapped[int] = mapped_column(BIGINT)
    """The index of the first dataset item in this episode.

    Can be -1 if not available. User should fix it after all
    processings are done.
    """

    frame_num: Mapped[int] = mapped_column(INTEGER)


@TableUpgradeRegistry.register_upgrade(
    table_name=Episode.__tablename__,
    from_version=None,
    to_version=Version("0.0.1"),
    from_orm_type=EpisodeDeprecatedVersion1,
)
def upgrade_episode_to_0_0_1(session: Session, row: dict) -> dict:
    """Upgrade an Episode row to version 0.0.1.

    Since this is the initial version, no changes are made.

    Args:
        session (Session): The database session.
        row (dict): The Episode row to upgrade.

    Returns:
        dict: The upgraded Episode row.
    """
    return row


@TableUpgradeRegistry.register_upgrade(
    table_name=Episode.__tablename__,
    from_version=Version("0.0.1"),
    to_version=Version("0.0.2"),
    from_orm_type=EpisodeDeprecatedVersion1,
)
def upgrade_episode_to_0_0_2(session: Session, row: dict) -> dict:
    """Upgrade an Episode row to version 0.0.2.

    Args:
        session (Session): The database session.
        row (dict): The Episode row to upgrade.

    Returns:
        dict: The upgraded Episode row.
    """
    row["truncated"] = None
    row["success"] = None
    row["info"] = None
    return row
