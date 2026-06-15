import logging
import math
import os
import time
from typing import Any, Dict, Optional, Union

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only


class ConsoleLogger(Logger):
    """
    Custom PyTorch Lightning console logger
    Automatically calculates and adds speed and ETA metrics to every metrics log entry
    Provides clean, formatted output for training information
    """

    def __init__(
        self,
        max_steps: int,
        level: int = logging.INFO,
        format: str = "%(asctime)s - %(levelname)s - %(message)s",
        datefmt: str = "%Y-%m-%d %H:%M:%S",
        max_step_digits: int = 7,
        speed_unit: str = "it/s",  # Speed unit, e.g. it/s, samples/s
        speed_window: int = 10,  # Sliding window size for speed calculation
    ):
        super().__init__()
        self._name = "lightning.pytorch"
        self._version = "0.1"
        self.max_step_digits = max_step_digits
        self.format = format
        self.datefmt = datefmt
        self.speed_unit = speed_unit
        self.speed_window = speed_window  # Use last N steps to calculate average speed
        self.max_steps = max_steps

        # Storage for speed calculation
        self._step_timestamps = []  # Stores recent step timestamps (step, timestamp)
        self._start_time = time.time()

        # Configure logger
        self.logger = logging.getLogger(self._name)
        self.logger.setLevel(level)

        # Ensure our formatter is applied
        self._configure_handlers()

    def _configure_handlers(self) -> None:
        """Ensure log handlers use our specified format"""
        # Look for existing StreamHandler
        stream_handler = None
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                stream_handler = handler
                break

        # Create new handler if none exists
        if not stream_handler:
            stream_handler = logging.StreamHandler()
            self.logger.addHandler(stream_handler)

        # Apply our formatter to the handler
        formatter = logging.Formatter(self.format, datefmt=self.datefmt)
        stream_handler.setFormatter(formatter)

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Union[int, str]:
        return self._version

    def _format_number(self, value: float) -> str:
        """Dynamically format numbers for readability"""
        if math.isnan(value):
            return "nan"
        if value == 0:
            return "0.000"
        if abs(value) < 1e-4 or abs(value) > 1e6:
            return f"{value:.3e}"
        magnitude = int(math.floor(math.log10(abs(value)))) if value != 0 else 0
        if magnitude >= 3:
            return f"{value:.3f}"
        else:
            formatted = f"{value:.6f}"
            return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted

    def _format_speed(self, speed: float) -> str:
        """Special formatting for speed values"""
        if speed >= 1000:
            return f"{speed/1000:.3f}k {self.speed_unit}"
        return f"{speed:.3f} {self.speed_unit}"

    def _calculate_speed(self, step: int) -> float:
        """Calculate speed based on recent steps using a sliding window"""
        current_time = time.time()
        self._step_timestamps.append((step, current_time))

        # Maintain only the most recent timestamps up to window size
        if len(self._step_timestamps) > self.speed_window:
            self._step_timestamps.pop(0)

        # Need at least 2 data points to calculate speed
        if len(self._step_timestamps) < 2:
            return 0.0

        first_step, first_time = self._step_timestamps[0]
        last_step, last_time = self._step_timestamps[-1]

        steps_diff = last_step - first_step
        time_diff = last_time - first_time

        if time_diff <= 0:
            return 0.0

        return steps_diff / time_diff

    def _format_eta(self, seconds: float) -> str:
        """Format ETA from seconds to human-readable string"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters in a formatted list"""
        params_str = "\n  ".join([f"{k}: {v}" for k, v in params.items()])
        self.logger.info(f"[HYPERPARAMS]\n  {params_str}")

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics with automatically added speed and ETA information.
        Speed and ETA metrics are always included when step information is available.
        """
        if "train/loss" not in metrics:
            return
        metrics.pop("epoch", None)

        # Create a copy to avoid modifying original metrics
        metrics_with_speed_and_eta = dict(metrics)

        # Calculate and add speed if step is provided
        if step is not None and step > 0:
            speed = self._calculate_speed(step)
            metrics_with_speed_and_eta["speed"] = speed

            # Calculate ETA
            steps_remaining = self.max_steps - step
            if speed > 0:
                eta_seconds = steps_remaining / speed
                eta_str = self._format_eta(eta_seconds)
                metrics_with_speed_and_eta["eta"] = eta_str

        # Separate and format speed and ETA metrics (always show first)
        speed_eta_metrics = []
        other_metrics = []

        for k, v in metrics_with_speed_and_eta.items():
            if k == "speed":
                speed_eta_metrics.append(f"{k}: {self._format_speed(v)}")
            elif k == "eta":
                speed_eta_metrics.append(f"{k}: {v}")
            else:
                other_metrics.append(f"{k}: {self._format_number(v)}")

        # Combine all metrics with speed and eta first
        metrics_str = ", ".join(speed_eta_metrics + other_metrics)

        # Format and log the final message
        if step is not None:
            step_str = f"Step: {step:0{self.max_step_digits}d}"
            self.logger.info(f"[METRICS] {step_str} - {metrics_str}")
        else:
            self.logger.info(f"[METRICS] {metrics_str}")

    @rank_zero_only
    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """Log text information with optional step context"""
        if step is not None:
            step_str = f"Step {step:0{self.max_step_digits}d}"
            self.logger.info(f"[TEXT] {step_str} - {name}: {text}")
        else:
            self.logger.info(f"[TEXT] {name}: {text}")

    @rank_zero_only
    def info(self, message: str) -> None:
        self.logger.info(f"[INFO] {message}")

    @rank_zero_only
    def warning(self, message: str) -> None:
        self.logger.warning(f"[WARNING] {message}")

    @rank_zero_only
    def error(self, message: str) -> None:
        self.logger.error(f"[ERROR] {message}")

    @rank_zero_only
    def debug(self, message: str) -> None:
        self.logger.debug(f"[DEBUG] {message}")
