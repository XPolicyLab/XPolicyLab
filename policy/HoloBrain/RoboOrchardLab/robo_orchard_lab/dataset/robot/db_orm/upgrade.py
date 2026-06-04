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

from typing import Callable, Dict, Type

from packaging.version import Version
from sqlalchemy.orm import Session
from typing_extensions import TypeAlias

from robo_orchard_lab.dataset.robot.db_orm.base import ORMBase

__all__ = [
    "Version",
    "TableUpgradeRegistry",
]


UpgradeFuncType: TypeAlias = Callable[[Session, dict], dict]
CreateFuncType: TypeAlias = Callable[[Session, Session], None]


class TableUpgradeRegistry:
    """Registry and resolver for table schema upgrade functions.

    This class keeps a class-level mapping of upgrade steps for table rows,
    allowing rows stored at an older schema "version" to be migrated to a
    specified target version via a chain of registered functions.
    """

    _registry_update: Dict[
        str,
        Dict[Version | None, Dict[Version, UpgradeFuncType]],
    ] = dict()
    """Nested mapping of registered upgrade functions.

    The structure is as follows:
        {
            table_name: {
                from_version | None: {
                    to_version: upgrade_func
                }
            }
        }
        where upgrade_func(session, old_row) performs the migration
        using a SQLAlchemy Session and the old row data as a dictionary,
        and returns the new row data as a dictionary.
    """

    _registry_create: Dict[str, Dict[Version | None, CreateFuncType]] = dict()
    _registry_orm: Dict[str, Dict[Version | None, Type[ORMBase]]] = dict()

    @staticmethod
    def register_create(
        table_name: str,
        version: Version | None,
    ):
        """Decorator to register a table creation function.

        Args:
            table_name (str): The name of the table to create.
            version (str|None): The version to create the table for.

        Returns:
            Callable: The decorator function.

        """

        def decorator(
            func: CreateFuncType,
        ) -> CreateFuncType:
            create_registry = TableUpgradeRegistry._registry_create
            if table_name not in create_registry:
                create_registry[table_name] = dict()
            if version in create_registry[table_name]:
                raise ValueError(
                    f"Create function for table {table_name} "
                    f"version {version} is already registered."
                )
            create_registry[table_name][version] = func

            return func

        return decorator

    @staticmethod
    def register_upgrade(
        table_name: str,
        from_version: Version | None,
        to_version: Version,
        from_orm_type: Type[ORMBase],
    ):
        """Decorator to register a table upgrade function.

        Args:
            table_name (str): The name of the table to upgrade.
            from_version (str|None): The version to upgrade from.
            to_version (str): The version to upgrade to. If None, the
                upgrade function applies to all higher versions.

        Returns:
            Callable: The decorator function.

        """

        def decorator(
            func: UpgradeFuncType,
        ) -> UpgradeFuncType:
            update_registry = TableUpgradeRegistry._registry_update
            if table_name not in update_registry:
                update_registry[table_name] = dict()
            if from_version not in update_registry[table_name]:
                update_registry[table_name][from_version] = dict()
            if to_version in update_registry[table_name][from_version]:
                raise ValueError(
                    f"Upgrade function for table {table_name} "
                    f"from version {from_version} to {to_version} "
                    f"is already registered."
                )
            update_registry[table_name][from_version][to_version] = func

            version_orm = TableUpgradeRegistry._registry_orm
            if table_name not in TableUpgradeRegistry._registry_orm:
                version_orm[table_name] = dict()
            if from_version in version_orm[table_name]:
                existing_orm = version_orm[table_name][from_version]
                if existing_orm != from_orm_type:
                    raise ValueError(
                        f"ORM class for table {table_name} "
                        f"version {from_version} is already registered "
                        f"as {existing_orm}, cannot register {from_orm_type}."
                    )

            version_orm[table_name][from_version] = from_orm_type

            return func

        return decorator

    @staticmethod
    def get_upgrade_func(
        table_name: str,
        current_version: Version | None,
        target_version: Version,
    ) -> UpgradeFuncType:
        """Get the upgrade function for a table.

        Args:
            table_name (str): The name of the table.
            current_version (str|None): The current version of the table.
            target_version (str): The target version of the table.

        Returns:
            UpgradeFuncType: The upgrade function.

        Raises:
            ValueError: If no upgrade functions are registered for the table,
                or if no upgrade path is found.

        """
        update_registry = TableUpgradeRegistry._registry_update
        if table_name not in update_registry:
            raise ValueError(
                f"No upgrade functions registered for {table_name}."
            )
        version_map = update_registry[table_name]
        upgrade_path: list[Callable[[Session, dict], dict]] = []

        from_version = current_version
        while True:
            if from_version not in version_map:
                raise ValueError(
                    f"No upgrade functions registered for {table_name} "
                    f"from version {from_version}."
                )
            # find the next to_version that is <= target_version
            to_versions = sorted(version_map[from_version].keys())
            next_to_version: Version | None = None
            for to_version in to_versions:
                if to_version > target_version:
                    break
                next_to_version = to_version

            if next_to_version is None:
                raise ValueError(
                    f"No upgrade path found for {table_name} "
                    f"from version {current_version} to {target_version}."
                )
            upgrade_func = version_map[from_version][next_to_version]
            upgrade_path.append(upgrade_func)
            if next_to_version == target_version:
                break
            from_version = next_to_version

        def chained_upgrade_func(session: Session, old_row: dict) -> dict:
            new_row = old_row
            for func in upgrade_path:
                new_row = func(session, new_row)
            return new_row

        return chained_upgrade_func

    @staticmethod
    def get_create_func(
        table_name: str,
        version: Version | None,
    ) -> CreateFuncType:
        """Get the create function for a table.

        Args:
            table_name (str): The name of the table.
            version (str|None): The version of the table.

        Returns:
            CreateFuncType: The create function.
        """
        create_registry = TableUpgradeRegistry._registry_create
        if table_name not in create_registry:
            raise ValueError(
                f"No create functions registered for {table_name}."
            )
        version_map = create_registry[table_name]
        if version not in version_map:
            raise ValueError(
                f"No create function registered for {table_name} "
                f"version {version}."
            )
        return version_map[version]

    @staticmethod
    def get_version_orm(
        table_name: str,
        version: Version | None,
    ) -> Type[ORMBase]:
        """Get the ORM class for a table version.

        Args:
            table_name (str): The name of the table.
            version (str|None): The version of the table.

        """
        version_orm = TableUpgradeRegistry._registry_orm
        if table_name not in version_orm:
            raise ValueError(f"No ORM classes registered for {table_name}.")
        version_map = version_orm[table_name]
        if version not in version_map:
            raise ValueError(
                f"No ORM class registered for {table_name} version {version}."
            )
        return version_map[version]
