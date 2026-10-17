# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public error categories for VLA creation and inference."""


class VLAConfigError(ValueError):
    """Configuration or inference input is invalid."""


class VLAUnsupportedError(RuntimeError):
    """The selected provider does not support the requested capability."""


class VLAClosedError(RuntimeError):
    """The component has already released its resources."""


class VLABusyError(RuntimeError):
    """Another predict, reset or close operation is in progress."""


class VLAInferenceError(RuntimeError):
    """Inference failed or returned an invalid provider result."""
