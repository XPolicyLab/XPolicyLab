from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .constants    import XONE_ROOT
from .data_handler import build_state, camera_meta, ensure_uint8_bgr, load_yaml

try:
    from PyQt5.QtCore import QLibraryInfo, Qt, QTimer
    from PyQt5.QtGui import QCloseEvent, QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    QT_IMPORT_ERROR: ImportError | None = exc
else:
    QT_IMPORT_ERROR = None


def require_qt() -> None:
    if QT_IMPORT_ERROR is not None:
        raise RuntimeError("PyQt5 is required for layout_shot") from QT_IMPORT_ERROR


def configure_qt_plugin_path() -> None:
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    if "cv2/qt/plugins" in plugin_path.replace("\\", "/"):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    pyqt_plugins_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    QApplication.setLibraryPaths([pyqt_plugins_path])


def extract_head_image(obs: dict[str, Any]) -> np.ndarray:
    return ensure_uint8_bgr(obs["vision"]["cam_head"]["color"])


def bgr_to_qimage(image_bgr: np.ndarray) -> "QImage":
    image_bgr = ensure_uint8_bgr(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = image_rgb.shape
    bytes_per_line = channels * width
    return QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


def layout_path(task_name: str, episode_idx: int) -> Path:
    return XONE_ROOT / "layouts" / task_name / f"layout_{int(episode_idx):06d}.png"


def cleanup_robot(robot) -> None:
    def join_owner_thread(owner) -> None:
        thread = getattr(owner, "_thread", None)
        if thread is not None:
            thread.join(timeout=1.0)

    def stop_scheduler(scheduler) -> None:
        scheduler.stop()
        join_owner_thread(scheduler)
        for node in getattr(scheduler, "all_nodes", []):
            join_owner_thread(node)

    for schedulers_name in ("collect_scheduler", "sensor_schedulers", "controller_schedulers"):
        schedulers = getattr(robot, schedulers_name, None)
        if schedulers is None:
            continue
        if isinstance(schedulers, dict):
            for scheduler in schedulers.values():
                stop_scheduler(scheduler)
        else:
            stop_scheduler(scheduler)

    for sensor_group in robot.sensors.values():
        for sensor in sensor_group.values():
            sensor.cleanup()


class LayoutCaptureEnv:
    def __init__(self, base_cfg: str, task_name: str) -> None:
        from robot.robot import get_robot

        config_path = XONE_ROOT / "config" / f"{base_cfg}.yml"
        self.m_base_cfg = load_yaml(config_path)
        self.m_base_cfg['collect']['task_name'] = task_name
        self.m_robot_lock = threading.Lock()
        self.m_robot = get_robot(base_cfg=self.m_base_cfg)
        self.m_robot.set_up(teleop=False)

        self.m_latest_obs: dict[str, Any] | None = None
        self.m_obs_lock = threading.Lock()
        self.m_closed = False

    def get_obs(self) -> dict[str, Any]:
        with self.m_robot_lock:
            controller_data, sensor_data = self.m_robot.get_obs()

        if not sensor_data:
            raise ValueError("sensor_data is empty")
        if not controller_data:
            raise ValueError("controller_data is empty")

        vision = {
            cam_name: camera_meta(
                self.m_base_cfg.get("robot", {}).get("camera_info", {}),
                cam_name,
                cam_data,
            )
            for cam_name, cam_data in sensor_data.items()
            if cam_data and cam_data.get("color") is not None
        }
        if "cam_head" not in vision:
            raise ValueError("cam_head is required for layout capture")

        return {
            "data_format_version": "v1.0",
            "env_idx": 0,
            "additional_info": {
                "frequency": self.m_base_cfg.get("frequency", 30),
            },
            "vision": vision,
            "state": build_state(controller_data),
        }

    def poll_obs(self) -> dict[str, Any]:
        obs = self.get_obs()
        with self.m_obs_lock:
            self.m_latest_obs = obs
        return obs

    def latest_obs(self) -> dict[str, Any] | None:
        with self.m_obs_lock:
            return self.m_latest_obs

    def close(self) -> None:
        if self.m_closed:
            return
        self.m_closed = True
        cleanup_robot(self.m_robot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture layout reference images from cam_head for RealEnvWorkbench. "
            "Output: {XONE_ROOT}/layouts/{task_name}/layout_{episode:06d}.png"
        )
    )
    parser.add_argument("--base_cfg", required=True, type=str, help="Robot config name under XONE_ROOT/config/")
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--layouts_count", required=True, type=int, help="Number of layout episodes to capture")
    parser.add_argument("--poll_hz", type=float, default=30.0)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--print_config_only", action="store_true")
    return parser.parse_args()


def print_launch_info(args: argparse.Namespace) -> None:
    print("[LayoutShot] launch config")
    print(f"  base_cfg:       {args.base_cfg}")
    print(f"  task_name:      {args.task_name}")
    print(f"  layouts_count:  {args.layouts_count}")
    print(f"  layout root:    {XONE_ROOT / 'layouts' / args.task_name}")
    print(f"  config path:    {XONE_ROOT / 'config' / f'{args.base_cfg}.yml'}")


if QT_IMPORT_ERROR is None:
    class ImageView(QLabel):
        def __init__(self) -> None:
            super().__init__()
            self.m_pixmap: QPixmap | None = None
            self.setMinimumSize(640, 360)
            self.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            self.setStyleSheet("background:#080a10; border:2px solid #1f2430;")

        def set_bgr_image(self, image_bgr: np.ndarray | None) -> None:
            if image_bgr is None:
                self.clear()
                self.m_pixmap = None
                return
            self.m_pixmap = QPixmap.fromImage(bgr_to_qimage(image_bgr))
            self._update_pixmap()

        def resizeEvent(self, event: Any) -> None:
            self._update_pixmap()
            super().resizeEvent(event)

        def _update_pixmap(self) -> None:
            if self.m_pixmap is None:
                return
            scaled = self.m_pixmap.scaled(
                self.contentsRect().size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.setPixmap(scaled)

    class LayoutShotWindow(QMainWindow):
        def __init__(
            self,
            env: LayoutCaptureEnv,
            task_name: str,
            layouts_count: int,
            poll_hz: float,
        ) -> None:
            super().__init__()
            self.m_env = env
            self.m_task_name = task_name
            self.m_layouts_count = int(layouts_count)
            self.m_poll_hz = float(poll_hz)
            self.m_idx = 0
            self.m_last_error: str | None = None
            self.m_logs: list[str] = []

            self.setWindowTitle("Layout 拍摄")
            self.resize(960, 780)

            self.m_status_label = QLabel()
            self.m_status_label.setStyleSheet("font-size:16px; padding:8px;")

            self.m_image_view = ImageView()

            self.m_log_view = QTextEdit()
            self.m_log_view.setReadOnly(True)
            self.m_log_view.setMinimumHeight(120)

            self.m_save_button = QPushButton("保存当前")
            self.m_save_button.clicked.connect(lambda: self.save_layout(advance=False))
            self.m_save_next_button = QPushButton("保存并下一轮")
            self.m_save_next_button.clicked.connect(lambda: self.save_layout(advance=True))
            self.m_next_button = QPushButton("下一轮")
            self.m_next_button.clicked.connect(self.next_layout)
            self.m_quit_button = QPushButton("退出")
            self.m_quit_button.clicked.connect(self.close)

            button_layout = QHBoxLayout()
            button_layout.addWidget(self.m_save_button)
            button_layout.addWidget(self.m_save_next_button)
            button_layout.addWidget(self.m_next_button)
            button_layout.addStretch(1)
            button_layout.addWidget(self.m_quit_button)

            root = QWidget()
            layout = QVBoxLayout(root)
            layout.addWidget(self.m_status_label)
            layout.addWidget(self.m_image_view, stretch=1)
            layout.addWidget(self.m_log_view)
            layout.addLayout(button_layout)
            self.setCentralWidget(root)

            self.m_timer = QTimer(self)
            self.m_timer.timeout.connect(self.tick)
            self.m_timer.start(max(1, int(1000.0 / self.m_poll_hz)))
            self._append_log(f"任务: {self.m_task_name}, 共 {self.m_layouts_count} 张 layout")
            self.render()

        def closeEvent(self, event: QCloseEvent) -> None:
            self.m_timer.stop()
            self.m_env.close()
            super().closeEvent(event)

        def tick(self) -> None:
            try:
                obs = self.m_env.poll_obs()
                self.m_image_view.set_bgr_image(extract_head_image(obs))
                self.m_last_error = None
            except Exception as exc:
                self.m_last_error = str(exc)
            self.render()

        def save_layout(self, *, advance: bool) -> None:
            obs = self.m_env.latest_obs()
            if obs is None:
                self.m_last_error = "还没有可用画面"
                self.render()
                return

            output_path = layout_path(self.m_task_name, self.m_idx)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            image = extract_head_image(obs)
            if not cv2.imwrite(str(output_path), image):
                raise ValueError(f"写入 layout 失败: {output_path}")

            metadata = {
                "task": self.m_task_name,
                "episode": self.m_idx,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "output_path": str(output_path),
                "shape": list(image.shape),
            }
            metadata_path = output_path.with_suffix(".json")
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            self.m_last_error = None
            self._append_log(f"已保存 layout_{self.m_idx:06d} -> {output_path}")
            if advance and self.m_idx < self.m_layouts_count - 1:
                self.m_idx += 1
                self._append_log(f"切换到第 {self.m_idx} 轮")
            self.render()

        def next_layout(self) -> None:
            if self.m_idx >= self.m_layouts_count - 1:
                return
            self.m_idx += 1
            self._append_log(f"切换到第 {self.m_idx} 轮")
            self.render()

        def _append_log(self, message: str) -> None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.m_logs.append(f"[{timestamp}] {message}")
            self.m_log_view.setPlainText("\n".join(self.m_logs))
            self.m_log_view.verticalScrollBar().setValue(
                self.m_log_view.verticalScrollBar().maximum()
            )

        def render(self) -> None:
            output_path = layout_path(self.m_task_name, self.m_idx)
            exists = "已存在" if output_path.is_file() else "未保存"
            status_parts = [
                f"任务: {self.m_task_name}",
                f"轮次: {self.m_idx}/{self.m_layouts_count - 1}",
                f"保存路径: {output_path}",
                f"状态: {exists}",
            ]
            if self.m_last_error is not None:
                status_parts.append(f"错误: {self.m_last_error}")
            self.m_status_label.setText(" | ".join(status_parts))
            self.m_next_button.setEnabled(self.m_idx < self.m_layouts_count - 1)
            self.m_save_next_button.setEnabled(self.m_idx < self.m_layouts_count - 1)


def main() -> int:
    args = parse_args()
    if int(args.layouts_count) <= 0:
        raise ValueError("layouts_count must be positive")

    print_launch_info(args)
    if args.print_config_only:
        return 0

    require_qt()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    configure_qt_plugin_path()

    app = QApplication.instance() or QApplication(sys.argv[:1])
    env = LayoutCaptureEnv(base_cfg=args.base_cfg, task_name=args.task_name)
    window = LayoutShotWindow(
        env=env,
        task_name=args.task_name,
        layouts_count=int(args.layouts_count),
        poll_hz=float(args.poll_hz),
    )
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
