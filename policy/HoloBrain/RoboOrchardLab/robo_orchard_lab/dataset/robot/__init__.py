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


"""The RoboOrchard datasets.

The datasets for training/evaluation for most applications are usually
tabular data, which includes features and labels as columns. Tabular datasets
are widely supported by many libraries, such as Hugging Face Datasets, Pyarrow.
However, for robotics applications, store all the data in a tabular format
is not always the best choice.

For example, each row in a tabular dataset may contain sensor data, labels,
and other information for one sample. The sample usually corresponds to a
small time step (Frame-level).
We also need to store episode-level information, such as the data collection
environment, the robot's URDF, and the task information. Storing all
frame-level and episode-level information in a single table leads to a
very large table in size, because the episode-level information is duplicated
for each frame. This is not efficient for storage and training.

One solution is to store non-frame-level information in seperate tables,
and use a unique identifier to link the frame-level and episode-level
information. This is similar to the relational database model, where
the data is stored in multiple tables and linked by foreign keys.

This is the approach used in RoboOrchard. For RoboOrchard datasets, we
use a tabular dataset to store the frame-level information, and a separate
database to store the episode-level information. We use huggingface datasets
(pyarrow_dataset)as table format, and use SQLAlchemy with DuckDB as
database API.

"""

from . import db_orm
from .columns import *
from .dataset import *
from .dataset_db_engine import *
from .dataset_ex import *
from .db_orm import RobotDescriptionFormat
from .packaging import *
from .re_packing import *
from .row_sampler import *
