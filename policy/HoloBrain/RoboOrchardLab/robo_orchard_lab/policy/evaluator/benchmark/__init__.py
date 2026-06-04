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

"""Benchmark-level policy evaluation interfaces.

Most callers should start with a domain evaluator, such as
``benchmark.robotwin.RoboTwinBenchmarkEvaluator``. A
:class:`BenchmarkEvaluator` accepts one policy or policy config, runs the full
benchmark, and returns a :class:`BenchmarkResult`.

:class:`BenchmarkDriver`, :class:`LocalBenchmarkBackend`, and
:class:`RemoteBenchmarkBackend` are extension points for implementing
benchmarks. A driver owns domain rules such as episode identity, readiness,
retry, seed advancement, artifacts, and aggregate metrics. A backend owns
execution mechanics such as worker capacity, prepare/evaluate scheduling,
worker lifecycle, and cleanup. The remote backend additionally owns timeout
handling and stale-completion handling.
"""

from robo_orchard_lab.policy.evaluator.benchmark.backend import (
    LocalBenchmarkBackend,
    LocalBenchmarkBackendConfig,
    RemoteBenchmarkBackend,
    RemoteBenchmarkBackendConfig,
)
from robo_orchard_lab.policy.evaluator.benchmark.core import (
    BenchmarkAttemptError,
    BenchmarkAttemptRequest,
    BenchmarkDriver,
    BenchmarkEpisode,
    BenchmarkEpisodeRecord,
    BenchmarkEvaluateFailedEvent,
    BenchmarkEvaluateSucceededEvent,
    BenchmarkEvaluator,
    BenchmarkPrepareFailedEvent,
    BenchmarkPrepareJob,
    BenchmarkPrepareSucceededEvent,
    BenchmarkResult,
    BenchmarkTerminalEvent,
)

__all__ = [
    "BenchmarkAttemptError",
    "BenchmarkAttemptRequest",
    "BenchmarkDriver",
    "BenchmarkEpisode",
    "BenchmarkEpisodeRecord",
    "BenchmarkEvaluateFailedEvent",
    "BenchmarkEvaluateSucceededEvent",
    "BenchmarkEvaluator",
    "BenchmarkPrepareFailedEvent",
    "BenchmarkPrepareJob",
    "BenchmarkPrepareSucceededEvent",
    "BenchmarkResult",
    "BenchmarkTerminalEvent",
    "LocalBenchmarkBackend",
    "LocalBenchmarkBackendConfig",
    "RemoteBenchmarkBackend",
    "RemoteBenchmarkBackendConfig",
]
