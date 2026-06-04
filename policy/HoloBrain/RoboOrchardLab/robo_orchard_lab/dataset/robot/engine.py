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

"""The database engine utilities that wrap SQLAlchemy.

For database engine which is used specifically for RoboOrchard dataset,
see :py:mod:`robo_orchard_lab.dataset.robot.dataset_db_engine`.

"""

import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Generator, Sequence, Type

from robo_orchard_core.utils.patches import patch_class_method
from sqlalchemy import (
    URL,
    Connection,
    Engine,
    Table,
    create_engine as _create_engine,
    event,
    text,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase

__all__ = [
    "create_engine",
    "create_tables",
    "drop_tables",
    "create_temp_engine",
]


def create_tables(
    engine: Engine,
    base: Type[DeclarativeBase],
    tables: Sequence[Table] | None = None,
    checkfirst: bool = True,
):
    return base.metadata.create_all(
        engine, tables=tables, checkfirst=checkfirst
    )


def drop_tables(
    engine: Engine,
    base: Type[DeclarativeBase],
    tables: Sequence[Table] | None = None,
    checkfirst: bool = True,
):
    """Drop tables in the database.

    Args:
        engine (Engine): The database engine.
        base (Type[DeclarativeBase]): The base class of the ORM models.
        tables (Sequence[Table], optional): The tables to drop. If None,
            all tables in the base are dropped. Defaults to None.
        checkfirst (bool, optional): Whether to check if the tables exist
            before dropping them. Defaults to True.
    """
    base.metadata.drop_all(engine, tables=tables, checkfirst=checkfirst)


@patch_class_method(Connection, "_commit_impl", check_method_name_exists=True)
def _commit_impl(self) -> None:
    """Adhoc patch to support readonly mode for SQLAlchemy Connection."""
    if getattr(self, "_orchard_readonly", False):
        raise OperationalError(
            statement=None,
            params=None,
            orig=RuntimeError("Cannot commit in readonly mode."),
        )
    self.__old__commit_impl()


def _readonly_begin(connection: Connection):
    name = connection.engine.url.get_backend_name()
    if name in ["mysql"]:
        connection.execute(text("START TRANSACTION READ ONLY"))
    elif name in ["duckdb"]:
        connection._orchard_readonly = True  # type: ignore
    else:
        warnings.warn(
            "`START TRANSACTION READ ONLY` is not supported "
            f"for backend {name}. "
            "Use ad-hoc `Connection` path instead."
        )
        connection._orchard_readonly = True  # type: ignore


def _make_readonly_engine(engine: Engine) -> Engine:
    be_name = engine.url.get_backend_name()
    # optimize isolation level
    if be_name in ["mysql"]:
        engine = engine.execution_options(isolation_level="AUTOCOMMIT")
    event.listen(engine, "begin", _readonly_begin)
    return engine


def create_engine(
    url: URL,
    readonly: bool = False,
    json_serializer: Callable[[Any], str] = lambda obj: json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
    ),
    **kwargs,
) -> Engine:
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
        **kwargs: Additional keyword arguments for
            :py:func:`sqlalchemy.create_engine`.

    Returns:
        Engine: The database engine.

    """
    if readonly:
        name = url.get_backend_name()
        if name == "sqlite":
            # the following does not work!
            # new_url = url.set(query={"mode": "rodd"})
            assert url.database is not None
            if not os.path.exists(url.database):
                raise IOError(f"Database `{url.database}` does not exist.")
            import sqlite3

            def creator():
                assert url.database is not None
                return sqlite3.connect(
                    "file:" + url.database + "?mode=ro", uri=True
                )

            # sqlite does not support pool_use_lifo
            if "pool_use_lifo" in kwargs:
                kwargs.pop("pool_use_lifo")
            return _create_engine(
                "sqlite:///",
                creator=creator,
                json_serializer=json_serializer,
                **kwargs,
            )
        if name == "duckdb":
            if "connect_args" not in kwargs:
                kwargs["connect_args"] = {}
            connect_args = kwargs["connect_args"]
            if connect_args.get("read_only", readonly) is not readonly:
                raise ValueError(
                    "connect_args['read_only'] must be consistent with "
                    "`readonly` argument."
                )
            connect_args["read_only"] = readonly

        engine = _create_engine(url, json_serializer=json_serializer, **kwargs)
        return _make_readonly_engine(engine)
    else:
        return _create_engine(url, json_serializer=json_serializer, **kwargs)


@contextmanager
def create_temp_engine(
    dir: str = os.path.abspath("./"),
    prefix: str = "duckdb_temp",
    drivername: str = "duckdb",
    **kwargs,
) -> Generator[Engine, Any, None]:
    """Create a temporary SQLite engine."""
    db_fd, db_path = tempfile.mkstemp(dir=dir, prefix=prefix, suffix=".tmp")

    db_file = db_path.replace(".tmp", f".{drivername}")
    tmp_db_url = URL.create(drivername=drivername, database=db_file)
    local_engine = create_engine(tmp_db_url, readonly=False, **kwargs)
    try:
        yield local_engine
    finally:
        local_engine.dispose()
        os.close(db_fd)
        os.unlink(db_path)
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(db_file):
            os.remove(db_file)
