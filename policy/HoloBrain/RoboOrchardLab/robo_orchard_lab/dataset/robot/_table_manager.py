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

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import IO, Type

from packaging.version import Version
from robo_orchard_core.datatypes.tf_graph import EdgeGraph
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.pip import get_package_version
from sqlalchemy import URL, Engine, ForeignKey, inspect
from sqlalchemy.orm import Mapper, Session
from sqlalchemy.schema import CreateTable
from sqlalchemy.sql import select

from robo_orchard_lab.dataset.robot.db_orm import DatasetORMBase, TableInfo

logger = LoggerManager().get_child(__name__)


APP_NAME = "robo_orchard_lab"


class TableUpgradeAction(Enum):
    KEEP = "keep"
    ADD = "add"
    UPGRADE = "upgrade"


@dataclass
class TableUpgradeStep:
    action: TableUpgradeAction
    orm_class: Type[DatasetORMBase] | None = None
    """The ORM class associated with the action, if applicable."""
    from_version: Version | None = None
    """The source version for upgrade action, if applicable."""
    target_version: Version | None = None
    """The target version for upgrade action, if applicable."""


@dataclass
class TableUpgradePlan:
    actions: list[TableUpgradeStep]

    to_delete: list[str]
    """The tables to be deleted. This is a list of table names.

    We only record the table names but not delete the tables, as
    upgrade always creates new database and migrates data over.
    """


class VersionMissingError(Exception):
    """Exception raised when a table does not have __version__ attribute."""

    pass


class ORMTopoGraph(EdgeGraph[str, Type[DatasetORMBase]]):
    """The topology graph of ORM tables."""

    def __init__(self, orm_list: list[Type[DatasetORMBase]]) -> None:
        super().__init__()
        for orm in orm_list:
            self._add_node(orm.__tablename__, orm)
        for orm in orm_list:
            mapper = orm.__mapper__
            fks = self._get_fk(mapper)
            for fk in fks:
                referred_table = fk.column.table.name
                if referred_table != orm.__tablename__:
                    self._add_edge(
                        from_node=referred_table,
                        to_node=orm.__tablename__,
                        edge="fk",
                    )

    def get_node(self, name: str) -> Type[DatasetORMBase]:
        return self.nodes[name]

    def get_nodes(self) -> dict[str, Type[DatasetORMBase]]:
        return self.nodes.copy()

    def _get_fk(self, mapper: Mapper) -> list[ForeignKey]:
        ret = []
        for column in mapper.columns:
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    ret.append(fk)
        return ret

    def topo_sort(self) -> list[Type[DatasetORMBase]] | None:
        """Topologically sort the graph.

        This method sort the ORM class based on their dependencies,
        and return the sorted list. The order is from the least dependent
        to the most dependent.

        Returns:
            Optional[Sequence[T]]: The topologically sorted nodes.
            If the graph has a cycle, return None.
        """
        # calculate in degree
        in_deg = deepcopy(self._in_degree)

        # initialize the queue with nodes that have 0 in-degree
        # these nodes are the starting nodes of the graph
        que = []
        for node, d in in_deg.items():
            if d == 0:
                que.append(node)

        idx = 0
        while idx < len(que):
            u = que[idx]
            # for every child, update in-degree. If in-degree becomes
            # 0, add it to the queue.
            for v in self.edges[u]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    que.append(v)
            idx += 1

        if len(que) < len(self.edges):
            return None

        return [self.get_node(name) for name in que]


def add_table_info(
    engine: Engine,
    checkfirst: bool = True,
) -> None:
    """Add table information to table_info table.

    If the table already exists in table_info, skip it.

    Args:
        engine (Engine): The database engine.
        checkfirst (bool, optional): Whether to check first. Defaults to True.

    Raises:
        VersionMissing: If the table does not have __version__ attribute.

    """

    tables: list[Type[DatasetORMBase]] = [
        mapper.class_ for mapper in DatasetORMBase.registry.mappers
    ]

    existing_table_info = {}
    with Session(bind=engine) as session:
        if checkfirst:
            stmt = select(TableInfo).execution_options(yield_per=100)
            for ti in session.scalars(stmt):
                existing_table_info[ti.pk_str] = ti
        tb_info_dict = {}
        for tb in tables:
            tb_name = tb.__table__.name  # type: ignore
            tb_version = getattr(tb, "__version__", None)
            if tb_version is None:
                raise VersionMissingError(
                    f"Table {tb_name} does not have __version__ attribute. "
                    "Please add __version__ attribute to the table class."
                )
            tb_creation_statement = str(
                CreateTable(tb.__table__).compile(engine)  # type: ignore
            )
            ti = TableInfo(
                table_name=tb_name,
                table_version=tb_version,
                app_version=get_package_version(f"{APP_NAME}"),
                creation_statement=tb_creation_statement,
            )
            if ti.pk_str not in existing_table_info:
                tb_info_dict[ti.pk_str] = ti
            else:
                logger.debug(
                    f"Table {tb_name} already exists in table_info. Skipping."
                )

        if len(tb_info_dict) > 0:
            for _k, v in tb_info_dict.items():
                session.add(v)
            session.commit()


def get_table_update_plan(engine: Engine) -> TableUpgradePlan | None:
    """Get the table update plan for the database.

    Args:
        engine (Engine): The database engine.

    Returns:
        TableUpgradePlan|None: The table update plan, or None if no
        updates are needed.

    """
    # get the current tables in code
    orm_topo = ORMTopoGraph(
        [mapper.class_ for mapper in DatasetORMBase.registry.mappers]
    )
    # tables_now: dict[str, Type[DatasetORMBase]] = orm_topo.get_nodes()
    # get the table names based on topo sort
    topo_sorted_orm_cls = orm_topo.topo_sort()
    if topo_sorted_orm_cls is None:
        raise ValueError("The ORM table dependency graph has a cycle.")

    # exclude TableInfo in topo_sorted_orm_cls
    topo_sorted_orm_cls = [
        cls
        for cls in topo_sorted_orm_cls
        if cls.__tablename__ != TableInfo.__tablename__
    ]

    insp = inspect(engine)
    table_names = set(insp.get_table_names())
    if TableInfo.__tablename__ not in table_names:
        # No table_info table, all tables need to be updated
        # This happens when table version is before 0.0.1,
        # and table_info table is not created yet.
        # we call upgrade for all tables because table names
        # does not change and only table migration is needed.

        # This logic needs to be updated if the table schema
        # changes in a future version!

        return TableUpgradePlan(
            actions=[
                TableUpgradeStep(
                    action=TableUpgradeAction.UPGRADE,
                    orm_class=cls,
                    from_version=None,
                    target_version=Version(cls.__version__),
                )
                for cls in topo_sorted_orm_cls
            ],
            to_delete=[],
        )

    plan = TableUpgradePlan(
        actions=[],
        to_delete=[],
    )

    with Session(bind=engine) as session:
        stmt = select(TableInfo).execution_options(yield_per=100)
        # exclude TableInfo itself
        existing_table_info = {
            ti.table_name: ti
            for ti in session.scalars(stmt)
            if ti.table_name != TableInfo.__tablename__
        }
        # check each table in code against existing_table_info
        for cls in topo_sorted_orm_cls:
            tb_name = cls.__tablename__
            code_table_version = Version(cls.__version__)
            if tb_name in existing_table_info:
                # table exists in both db and code, compare versions
                # and decide whether to upgrade
                table_info = existing_table_info[tb_name]
                db_table_version = Version(table_info.table_version)
                if code_table_version > db_table_version:
                    # plan.need_upgrade.append((db_table_version, cls))
                    plan.actions.append(
                        TableUpgradeStep(
                            action=TableUpgradeAction.UPGRADE,
                            orm_class=cls,
                            from_version=db_table_version,
                            target_version=code_table_version,
                        )
                    )
                elif code_table_version == db_table_version:
                    plan.actions.append(
                        TableUpgradeStep(
                            action=TableUpgradeAction.KEEP,
                            orm_class=cls,
                        )
                    )
                else:
                    raise ValueError(
                        f"Database table {tb_name} version "
                        f"{db_table_version} is newer than code "
                        f"version {code_table_version}."
                    )
                # remove from existing_table_info
                existing_table_info.pop(tb_name)
            else:
                # table in code but not in db, mark as to_add
                # plan.to_add.append(cls)
                plan.actions.append(
                    TableUpgradeStep(
                        action=TableUpgradeAction.ADD,
                        orm_class=cls,
                        target_version=code_table_version,
                    )
                )
        # remaining tables in existing_table_info are to be deleted
        for tb_name in existing_table_info.keys():
            plan.to_delete.append(tb_name)
    # if no changes, return None
    # No change criteria:
    # 1. No actions in plan.actions or all actions are KEEP
    # 2. No tables to delete in plan.to_delete

    if (
        all(step.action == TableUpgradeAction.KEEP for step in plan.actions)
        or len(plan.actions) == 0
    ) and len(plan.to_delete) == 0:
        return None
    return plan


def get_tables_version_md5() -> str:
    """Get the MD5 hash of all current tables and their versions.

    This can be used to detect changes in the table definitions.

    Returns:
        str: The MD5 hash of the table names and versions.
    """
    version_dict = {}
    for mapper in DatasetORMBase.registry.mappers:
        cls: Type[DatasetORMBase] = mapper.class_
        tb_name = cls.__tablename__
        version = cls.__version__
        version_dict[tb_name] = version
    version_str = json.dumps(version_dict, sort_keys=True)
    md5_hash = hashlib.md5(version_str.encode("utf-8")).hexdigest()
    return md5_hash


def get_file_content_md5(handle: IO, buffer_size: int = 2**20 * 4) -> str:
    md5 = hashlib.md5()
    while chunk := handle.read(buffer_size):
        md5.update(chunk)
    return md5.hexdigest()


def extract_db_file_path(db_url: URL, check_exist: bool = True) -> str:
    """Extract the database file path from the database URL.

    Args:
        db_url (URL): The database URL.
        check_exist (bool, optional): Whether to check if the file exists.
            Defaults to True.

    Returns:
        str: The absolute path of the database file.
    """
    # first check if db_url is a local db.
    local_path = db_url.database
    if local_path is None:
        raise ValueError(
            f"The database URL '{db_url}' does not have a local path."
        )
    if check_exist and not os.path.exists(local_path):
        raise FileNotFoundError(
            f"The database file '{local_path}' does not exist."
        )
    # convert path to absolute path
    return os.path.abspath(local_path)


def get_local_db_md5(db_url: URL) -> tuple[str, str]:
    """Get the MD5 hash of the local database file.

    This can be used to detect changes in the database configuration.

    Returns:
        tuple[str, str]: The MD5 hashes of the database URL string, and
        the contents of the database file.
    """
    # convert path to absolute path
    abs_path = extract_db_file_path(db_url, check_exist=True)
    url_str = str(db_url)
    url_md5_hash = hashlib.md5(url_str.encode("utf-8")).hexdigest()
    with open(abs_path, "rb") as f:
        file_md5_hash = get_file_content_md5(f)
    return url_md5_hash, file_md5_hash
