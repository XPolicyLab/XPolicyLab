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
import json
import os
from typing import Any, Callable

import filelock
from robo_orchard_core.utils.logging import LoggerManager
from sqlalchemy import URL
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, make_transient
from sqlalchemy.sql import select

from robo_orchard_lab.dataset.robot._table_manager import (
    TableUpgradeAction,
    TableUpgradePlan,
    TableUpgradeStep,
    add_table_info,
    extract_db_file_path,
    get_local_db_md5,
    get_table_update_plan,
    get_tables_version_md5,
)
from robo_orchard_lab.dataset.robot.db_orm.base import DatasetORMBase
from robo_orchard_lab.dataset.robot.db_orm.upgrade import TableUpgradeRegistry
from robo_orchard_lab.dataset.robot.engine import (
    create_engine as _create_engine,
    create_tables as _create_tables,
)
from robo_orchard_lab.utils.env import get_robo_orchard_home

logger = LoggerManager().get_child(__name__)

__all__ = [
    "create_tables",
    "create_engine",
    "get_local_db_url",
    "get_ro_dataset_home",
    "need_upgrade",
    "try_upgrade_database",
]


def get_ro_dataset_home() -> str:
    """Get the RoboOrchard dataset home directory.

    Returns:
        str: The RoboOrchard dataset home directory.
    """
    return os.path.join(get_robo_orchard_home(), "datasets")


def create_engine(
    url: URL,
    readonly: bool = False,
    json_serializer: Callable[[Any], str] = lambda obj: json.dumps(
        obj, sort_keys=True, ensure_ascii=True
    ),
    auto_upgrade: bool = True,
    **kwargs,
):
    """Create a database engine.

    This method is a wrapper of :py:func:`sqlalchemy.create_engine` with
    additional support for readonly mode, and default JSON serializer.

    See :py:func:`sqlalchemy.create_engine` for more details.

    Args:
        url (URL): The database URL.
        readonly (bool, optional): Whether to create a readonly engine.
            Defaults to False.
        json_serializer (Callable[[Any], str], optional): The JSON serializer
            for the engine. Defaults to
            `lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=True).`
        auto_upgrade (bool, optional): Whether to automatically upgrade the
            database to the latest version when reading.
            Defaults to True.
        **kwargs: Additional keyword arguments for
            :py:func:`sqlalchemy.create_engine`.

    Returns:
        Engine: The database engine.

    """
    if auto_upgrade and readonly:
        db_path = extract_db_file_path(url, check_exist=True)
        drivername = url.drivername
        handler = DatasetMetaDBHandler(db_path, drivername)
        url = handler.get_latest_db_url_for_read()

    return _create_engine(
        url, readonly=readonly, json_serializer=json_serializer, **kwargs
    )


def try_upgrade_database(
    url: URL,
) -> URL:
    """Try to upgrade the database to the latest version.

    Args:
        url (URL): The database URL.

    Returns:
        URL: The upgraded database URL.
    """
    handler = DatasetMetaDBHandler(
        extract_db_file_path(url, check_exist=True), url.drivername
    )
    return handler.get_latest_db_url_for_read()


def create_tables(
    engine: Engine,
    checkfirst: bool = True,
):
    """Create all tables in the database and add table info.

    Args:
        engine (Engine): The database engine.
        checkfirst (bool, optional): Whether to check first. Defaults to True.
    """
    _create_tables(engine, base=DatasetORMBase, checkfirst=checkfirst)
    add_table_info(engine, checkfirst=checkfirst)


def get_local_db_url(db_path: str, drivername: str) -> URL:
    """Get the local database URL."""
    return URL.create(drivername=drivername, database=db_path)


def need_upgrade(url_or_engine: URL | Engine) -> bool:
    """Check whether the database needs to be upgraded.

    Args:
        url_or_engine (URL | Engine): The database URL or engine.

    Returns:
        bool: Whether the database needs to be upgraded.
    """
    if isinstance(url_or_engine, URL):
        engine = _create_engine(url_or_engine, readonly=True)
        dispose_engine = True
    else:
        engine = url_or_engine
        dispose_engine = False

    upgrade_plan = get_table_update_plan(engine)
    if dispose_engine:
        engine.dispose()
    return upgrade_plan is not None


class DatasetMetaDBHandler:
    def __init__(self, db_path: str, drivername: str):
        self._db_url = get_local_db_url(db_path, drivername)
        self._db_path = db_path
        self._drivername = drivername

    def get_latest_db_url_for_read(self) -> URL:
        old_engine = _create_engine(self._db_url, readonly=True)
        upgrade_plan = get_table_update_plan(old_engine)
        if upgrade_plan is None:
            old_engine.dispose()
            return self._db_url
        cache_folder = self.get_db_url_upgrade_cache_folder(
            self._db_path, self._drivername
        )
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder, exist_ok=True)
        target_db_path = os.path.join(
            cache_folder, os.path.basename(self._db_path)
        )
        lock_file_path = os.path.join(cache_folder, "upgrade_cache.lock")
        with filelock.FileLock(lock_file_path):
            if not os.path.exists(target_db_path):
                print(
                    f"Upgrading database {self._db_url} to latest version... "
                    f"Target path: {target_db_path}. "
                    "You can manually update the dataset database by "
                    "copying the upgraded database file from the target path, "
                    "or use `RODataset.upgrade_meta(dataset_path)` method. "
                )
                self.upgrade_to_cache(
                    src_engine=old_engine,
                    target_db_path=target_db_path,
                    plan=upgrade_plan,
                )
        old_engine.dispose()
        return get_local_db_url(target_db_path, self._drivername)

    def upgrade_to_cache(
        self, src_engine: Engine, target_db_path: str, plan: TableUpgradePlan
    ):
        if len(plan.actions) == 0 and len(plan.to_delete) == 0:
            raise ValueError("No upgrade actions to perform!")

        target_engine: Engine = _create_engine(
            get_local_db_url(target_db_path, self._drivername),
            readonly=False,
        )
        create_tables(target_engine, checkfirst=True)
        with (
            Session(src_engine) as src_session,
            Session(target_engine) as target_session,
        ):
            for action in plan.actions:
                self._upgrade_step(src_session, target_session, action)

        target_engine.dispose()

    def _upgrade_step(
        self,
        src_session: Session,
        target_session: Session,
        step: TableUpgradeStep,
        batch_size: int = 1024,
    ):
        assert step.orm_class is not None
        if step.action == TableUpgradeAction.KEEP:
            for i, obj in enumerate(
                src_session.execute(select(step.orm_class)).scalars()
            ):
                # make obj transient
                make_transient(obj)
                target_session.add(obj)
                if (i + 1) % batch_size == 0:
                    target_session.commit()
            target_session.commit()
        elif step.action == TableUpgradeAction.UPGRADE:
            assert step.target_version is not None
            old_orm_type = TableUpgradeRegistry.get_version_orm(
                step.orm_class.__tablename__,
                version=step.from_version,
            )
            upgrade_func = TableUpgradeRegistry.get_upgrade_func(
                step.orm_class.__tablename__,
                current_version=step.from_version,
                target_version=step.target_version,
            )
            for i, row in enumerate(
                src_session.execute(select(old_orm_type)).scalars()
            ):
                row_dict = {}
                old_orm_type.column_copy(row, row_dict)
                upgraded_row_dict = upgrade_func(src_session, row_dict)
                new_row = step.orm_class(**upgraded_row_dict)
                target_session.add(new_row)
                if (i + 1) % batch_size == 0:
                    target_session.commit()
            target_session.commit()
        elif step.action == TableUpgradeAction.ADD:
            assert step.target_version is not None
            create_func = TableUpgradeRegistry.get_create_func(
                step.orm_class.__tablename__,
                version=step.target_version,
            )
            create_func(src_session, target_session)
        else:
            raise ValueError(f"Unknown action: {step.action}")

    @staticmethod
    def get_db_url_upgrade_cache_folder(db_path: str, drivername: str) -> str:
        """Get the cache folder for the database upgradinng target.

        Make sure that the db_path exists before calling this function.

        Args:
            db_path (str): The database path.
            drivername (str): The database driver name.
        """
        db_url = get_local_db_url(db_path, drivername)
        robo_orchard_home = get_ro_dataset_home()
        url_md5, file_md5 = get_local_db_md5(db_url)
        table_version_md5 = get_tables_version_md5()
        return os.path.join(
            robo_orchard_home,
            "upgrade_cache",
            f"{url_md5}-{file_md5}",
            f"{table_version_md5}",
        )
