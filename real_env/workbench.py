from __future__ import annotations

import cv2
import json
import numpy as np
import os
import shutil
import sys
import threading

from .constants       import XONE_ROOT
from .data_handler    import ensure_uint8_bgr
from .real_env_client import RealEnv
from dataclasses      import dataclass, field
from datetime         import datetime
from enum             import Enum
from pathlib          import Path
from typing           import Any

try:
    from PyQt5.QtCore    import QLibraryInfo, Qt, QTimer
    from PyQt5.QtGui     import QCloseEvent, QImage, QPixmap
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
        raise RuntimeError("PyQt5 is required for Workbench") from QT_IMPORT_ERROR

REASON_OPERATOR_EARLY_FINISH = "operator_early_finish"
REASON_OPERATOR_ABORT = "operator_abort"
REASON_OPERATOR_RETRY = "operator_retry"
REASON_OPERATOR_CLOSE = "operator_close"

class WorkflowState(str, Enum):
    TASK_INIT       = "TASK_INIT"
    LAYOUT_READY    = "LAYOUT_READY"
    PLACEMENT       = "PLACEMENT"
    EVALUATING      = "EVALUATING"
    AWAIT_RESULT    = "AWAIT_RESULT"
    EPISODE_ABORTED = "EPISODE_ABORTED"
    TASK_FINISHED   = "TASK_FINISHED"
    ERROR           = "ERROR"

@dataclass
class WorkbenchState:
    workflow_state: WorkflowState      = WorkflowState.TASK_INIT
    active_episode: int                = 0
    committed_episode_num: int         = 0
    success_num: int                   = 0
    target_episode_num: int            = 1
    alpha: float                       = 0.35
    live_image: np.ndarray | None      = None
    layout_image: np.ndarray | None = None
    layout_path: Path | None        = None
    episode_dir: Path | None           = None
    abort_reason: str | None           = None
    last_error: str | None             = None
    logs: list[str]                    = field(default_factory=list)

def load_bgr_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return ensure_uint8_bgr(image)

def extract_head_image(obs):
    return ensure_uint8_bgr(obs["vision"]["cam_head"]["color"])

def resize_like(image, target):
    image = ensure_uint8_bgr(image)
    target = ensure_uint8_bgr(target)
    if image.shape[:2] == target.shape[:2]:
        return image
    return cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_AREA)

def blend_bgr(live, layout, alpha):
    live = ensure_uint8_bgr(live)
    if layout is None or alpha <= 0:
        return live

    layout = resize_like(layout, live)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = live.astype(np.float32) * (1.0 - alpha) + layout.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)

def bgr_to_qimage(image_bgr) -> "QImage":
    image = ensure_uint8_bgr(image_bgr)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, ch = image.shape
    bytes_per_line = ch * w
    return QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

def configure_qt_plugin_path() -> None:
    plugin_path = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
    normalized_plugin_path = plugin_path.replace("\\", "/")
    if "cv2/qt/plugins" in normalized_plugin_path:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    pyqt_plugins_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    QApplication.setLibraryPaths([pyqt_plugins_path])

class WorkbenchController:
    def __init__(
        self,
        env,
        eval_episode_num: int = 1,
        recorder=None,
    ) -> None:
        self.m_env: RealEnv      = env
        self.m_recorder          = recorder
        self.m_deploy_cfg        = self.m_env.deploy_cfg
        self.m_task_name         = self.m_deploy_cfg['task_name']
        self.m_policy_name       = self.m_deploy_cfg.get('policy_name', self.m_task_name)
        self.m_ckpt_setting      = str(self.m_deploy_cfg['ckpt_setting'])
        self.m_eval_results_root = (
            XONE_ROOT / 'eval_results' / self.m_policy_name / self.m_ckpt_setting / self.m_task_name
        )
        self.m_layouts_root       = XONE_ROOT / 'layouts' / self.m_task_name
        self.m_state             = WorkbenchState(target_episode_num=int(eval_episode_num))
        self.m_eval_thread       = None
        self.m_eval_error        = None
        self.m_eval_done         = threading.Event()
        self.m_abort_requested   = False
        self.m_early_finish_requested = False
        self.m_closed            = False

    def prepare_task(self) -> None:
        if not self.m_layouts_root.is_dir():
            self._set_error(f"不存在的布局图文件夹: {self.m_layouts_root}")
            return

        self.m_state.workflow_state = WorkflowState.LAYOUT_READY
        self._append_log(f"评测任务准备完成: {self.m_task_name}")

    def start_episode(self, episode_idx=None):
        if episode_idx is not None:
            self.m_state.active_episode = int(episode_idx)
        episode_idx = self.m_state.active_episode

        layout_path = self.m_layouts_root / f"layout_{int(episode_idx):06d}.png"
        if not layout_path.is_file():
            self._set_error(f"找不到布局图: {layout_path}")
            return

        episode_dir = Path(self.m_eval_results_root) / f"episode_{int(episode_idx):06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        self.m_state.layout_path = layout_path
        self.m_state.layout_image = load_bgr_image(layout_path)
        self.m_state.episode_dir = episode_dir
        self.m_state.abort_reason = None
        self.m_abort_requested = False
        self.m_early_finish_requested = False
        self.m_state.workflow_state = WorkflowState.PLACEMENT
        self._append_log(f"开始进行第 {self.m_state.active_episode} 轮的摆放")

    def update_live_obs(self) -> None:
        obs = self.m_env.get_obs()
        self.m_state.live_image = extract_head_image(obs)

    def save_placement(self) -> Path:
        return self.finish_placement()

    def finish_placement(self) -> Path:
        episode_dir = self._save_placement()
        self._start_eval()
        return episode_dir

    def _save_placement(self) -> Path:
        if self.m_state.workflow_state != WorkflowState.PLACEMENT:
            raise RuntimeError(f"cannot save placement in state {self.m_state.workflow_state}")
        if self.m_state.live_image is None:
            raise ValueError("no live image to save")
        if self.m_state.episode_dir is None:
            raise ValueError("episode dir is not ready")

        episode_dir = self.m_state.episode_dir
        episode_dir.mkdir(parents=True, exist_ok=True)
        image_path = episode_dir / "placement.png"
        if not cv2.imwrite(str(image_path), ensure_uint8_bgr(self.m_state.live_image)):
            raise ValueError(f"failed to write placement image: {image_path}")

        metadata = {
            "task":              self.m_task_name,
            "policy":            self.m_policy_name,
            "ckpt_setting":      self.m_ckpt_setting,
            "episode":           self.m_state.active_episode,
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "layout_path":       str(self.m_state.layout_path) if self.m_state.layout_path is not None else None,
            "alpha":             self.m_state.alpha,
            "output_image_path": str(image_path),
        }

        metadata_path = episode_dir / "placement_metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        self._append_log(f"placement saved: {image_path}")
        return episode_dir

    def _start_eval(self) -> None:
        if self.m_eval_thread is not None and self.m_eval_thread.is_alive():
            raise RuntimeError("eval thread is already running")

        self.m_eval_error = None
        self.m_eval_done.clear()
        self.m_state.workflow_state = WorkflowState.EVALUATING
        self.m_eval_thread = threading.Thread(target=self._run_eval, name="WorkbenchEvalThread", daemon=True)
        self.m_eval_thread.start()
        self._append_log(f"第 {self.m_state.active_episode} 轮评测开始")

    def process_eval_events(self) -> None:
        if self.m_state.workflow_state != WorkflowState.EVALUATING:
            return
        if not self.m_eval_done.is_set():
            return

        if self.m_eval_thread is not None:
            self.m_eval_thread.join(timeout=0.0)
            self.m_eval_thread = None

        if self.m_abort_requested:
            self._finish_abort()
            return

        if self.m_early_finish_requested:
            self.m_early_finish_requested = False
            self.m_state.workflow_state = WorkflowState.AWAIT_RESULT
            self._append_log(f"第 {self.m_state.active_episode} 轮评测提前结束")
            return

        if self.m_eval_error is not None:
            self._set_error(str(self.m_eval_error))
            return

        self.m_state.workflow_state = WorkflowState.AWAIT_RESULT
        self._append_log(f"第 {self.m_state.active_episode} 轮评测结束")

    def finish_eval_early(self, reason: str = REASON_OPERATOR_EARLY_FINISH) -> None:
        if self.m_state.workflow_state != WorkflowState.EVALUATING:
            raise RuntimeError(f"cannot finish eval early in state {self.m_state.workflow_state}")
        if self.m_abort_requested or self.m_early_finish_requested:
            raise RuntimeError("eval stop already requested")

        stop_reason = str(reason or REASON_OPERATOR_EARLY_FINISH)
        self.m_early_finish_requested = True
        self.m_env.request_stop(stop_reason)
        self._append_log(f"第 {self.m_state.active_episode} 轮评测已请求提前结束")

    def abort_episode(self, reason: str = REASON_OPERATOR_ABORT) -> None:
        if self.m_state.workflow_state != WorkflowState.EVALUATING:
            raise RuntimeError(f"cannot abort episode in state {self.m_state.workflow_state}")
        if self.m_abort_requested or self.m_early_finish_requested:
            raise RuntimeError("eval stop already requested")

        stop_reason = str(reason or REASON_OPERATOR_ABORT)
        self.m_abort_requested = True
        self.m_state.abort_reason = stop_reason
        self.m_env.request_stop(stop_reason)
        self._append_log(f"第 {self.m_state.active_episode} 轮评测已请求异常终止")

    def retry_episode(self, reason: str = REASON_OPERATOR_RETRY) -> None:
        from_state = self.m_state.workflow_state
        if from_state not in {WorkflowState.AWAIT_RESULT, WorkflowState.EPISODE_ABORTED}:
            raise RuntimeError(f"cannot retry episode in state {from_state}")

        retry_reason = str(reason or REASON_OPERATOR_RETRY)
        episode_idx = self.m_state.active_episode
        self._cleanup_uncommitted_episode_artifacts()
        self._append_retry_event(retry_reason, from_state)
        self._append_log(f"第 {episode_idx} 轮重新评测")
        self.m_abort_requested = False
        self.m_early_finish_requested = False
        self.m_state.abort_reason = None
        self.m_eval_error = None
        self.m_eval_done.clear()
        self.start_episode(episode_idx)

    def close(self) -> None:
        if self.m_closed:
            return

        self._close_running_eval()
        self.m_env.close()
        self.m_closed = True

    def _close_running_eval(self) -> None:
        if self.m_state.workflow_state != WorkflowState.EVALUATING:
            return

        if not self.m_abort_requested:
            self.m_abort_requested = True
            self.m_state.abort_reason = REASON_OPERATOR_CLOSE
            self.m_env.request_stop(REASON_OPERATOR_CLOSE)
            self._append_log(f"第 {self.m_state.active_episode} 轮评测已请求异常终止（关闭窗口）")

        self.m_eval_done.wait()
        if self.m_eval_thread is not None:
            self.m_eval_thread.join()
            self.m_eval_thread = None

        if self.m_abort_requested:
            self._finish_abort()
        elif self.m_eval_error is not None:
            self._set_error(str(self.m_eval_error))

    def mark_success(self) -> None:
        self._commit_result('success')

    def mark_fail(self) -> None:
        self._commit_result('fail')

    def current_display_image(self) -> np.ndarray | None:
        if self.m_state.live_image is None:
            return None
        alpha = (
            0.0
            if self.m_state.workflow_state == WorkflowState.EVALUATING
            else self.m_state.alpha
        )
        return blend_bgr(self.m_state.live_image, self.m_state.layout_image, alpha)

    def _run_eval(self) -> None:
        try:
            self.m_env.reset_robot()
            if self.m_recorder is not None:
                assert self.m_state.episode_dir is not None
                self.m_recorder.start(self.m_state.episode_dir)
            try:
                eval_batch = bool(self.m_deploy_cfg.get('eval_batch', False))
                if eval_batch:
                    self.m_env.eval_one_episode_batch()
                else:
                    self.m_env.eval_one_episode()
            finally:
                try:
                    self.m_env.finish_episode()
                except BaseException as cleanup_exc:
                    if self.m_eval_error is None:
                        self.m_eval_error = cleanup_exc
                if self.m_recorder is not None:
                    if self.m_abort_requested:
                        self.m_recorder.abort()
                    else:
                        self.m_recorder.stop()
        except BaseException as exc:
            self.m_eval_error = exc
        finally:
            self.m_eval_done.set()

    def _commit_result(self, result: str) -> None:
        if self.m_state.workflow_state != WorkflowState.AWAIT_RESULT:
            raise RuntimeError(f"cannot mark result in state {self.m_state.workflow_state}")

        if result == 'success':
            self.m_state.success_num += 1
        self.m_state.committed_episode_num += 1
        self._append_result_event(result)

        if self.m_state.committed_episode_num >= self.m_state.target_episode_num:
            self.m_state.workflow_state = WorkflowState.TASK_FINISHED
            self._append_log("评测任务结束")
            return

        self.m_state.active_episode += 1
        self.m_state.workflow_state = WorkflowState.LAYOUT_READY
        self.start_episode()

    def _finish_abort(self) -> None:
        assert self.m_state.abort_reason is not None
        reason = self.m_state.abort_reason
        self._cleanup_uncommitted_episode_artifacts()
        self.m_state.workflow_state = WorkflowState.EPISODE_ABORTED
        self.m_abort_requested = False
        self._append_abort_event(reason)
        self._append_log(f"第 {self.m_state.active_episode} 轮已异常终止")

    def _cleanup_uncommitted_episode_artifacts(self) -> None:
        assert self.m_state.episode_dir is not None
        episode_dir = self.m_state.episode_dir

        for path in (
            episode_dir / "placement.png",
            episode_dir / "placement_metadata.json",
        ):
            path.unlink(missing_ok=True)

        for path in episode_dir.glob("*.mp4"):
            path.unlink()

        shutil.rmtree(episode_dir / "recorder", ignore_errors=True)

    def _append_result_event(self, result: str) -> None:
        self.m_eval_results_root.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "episode_committed",
            "episode": int(self.m_state.active_episode),
            "result": result,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "success": int(self.m_state.success_num),
            "total": int(self.m_state.committed_episode_num),
        }
        with open(self.m_eval_results_root / "result_events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _append_abort_event(self, reason: str) -> None:
        self.m_eval_results_root.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "episode_aborted",
            "episode": int(self.m_state.active_episode),
            "reason": reason,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.m_eval_results_root / "result_events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _append_retry_event(self, reason: str, from_state: WorkflowState) -> None:
        self.m_eval_results_root.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "episode_retried",
            "episode": int(self.m_state.active_episode),
            "reason": reason,
            "from_state": from_state.value,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.m_eval_results_root / "result_events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _set_error(self, message: str) -> None:
        self.m_state.last_error = message
        self.m_state.workflow_state = WorkflowState.ERROR
        self._append_log(f"ERROR: {message}")

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.m_state.logs.append(f"[{timestamp}] {message}")


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


    class WorkbenchWindow(QMainWindow):
        def __init__(self, controller: WorkbenchController, poll_hz: float = 30.0) -> None:
            super().__init__()
            self.m_controller = controller
            self.m_poll_hz = float(poll_hz)
            self.m_closed = False
            self.m_rendered_log_count = -1

            self.setWindowTitle("Real Env Workbench")
            self.resize(1280, 860)

            self.m_status_label = QLabel()
            self.m_status_label.setStyleSheet("font-size:16px; padding:8px;")

            self.m_image_view = ImageView()

            self.m_log_view = QTextEdit()
            self.m_log_view.setReadOnly(True)
            self.m_log_view.setMinimumHeight(110)

            self.m_prepare_button = QPushButton("准备任务")
            self.m_prepare_button.clicked.connect(self.prepare_task)
            self.m_save_button = QPushButton("摆放完成")
            self.m_save_button.clicked.connect(self.save_placement)
            self.m_success_button = QPushButton("成功")
            self.m_success_button.clicked.connect(self.mark_success)
            self.m_fail_button = QPushButton("失败")
            self.m_fail_button.clicked.connect(self.mark_fail)
            self.m_retry_button = QPushButton("重试本轮")
            self.m_retry_button.clicked.connect(self.retry_episode)
            self.m_early_finish_button = QPushButton("提前结束")
            self.m_early_finish_button.clicked.connect(self.finish_eval_early)
            self.m_abort_button = QPushButton("异常终止")
            self.m_abort_button.setStyleSheet("color: red;")
            self.m_abort_button.clicked.connect(self.abort_episode)
            self.m_quit_button = QPushButton("退出")
            self.m_quit_button.clicked.connect(self.close)

            button_layout = QHBoxLayout()
            button_layout.addWidget(self.m_prepare_button)
            button_layout.addWidget(self.m_save_button)
            button_layout.addWidget(self.m_success_button)
            button_layout.addWidget(self.m_fail_button)
            button_layout.addWidget(self.m_retry_button)
            button_layout.addWidget(self.m_early_finish_button)
            button_layout.addWidget(self.m_abort_button)
            button_layout.addStretch(1)
            button_layout.addWidget(self.m_quit_button)

            root = QWidget()
            layout = QVBoxLayout(root)
            layout.addWidget(self.m_status_label)
            layout.addWidget(self.m_log_view)
            layout.addWidget(self.m_image_view, stretch=1)
            layout.addLayout(button_layout)
            self.setCentralWidget(root)

            self.m_timer = QTimer(self)
            self.m_timer.timeout.connect(self.tick)
            self.m_timer.start(max(1, int(1000.0 / self.m_poll_hz)))
            self.render()

        def closeEvent(self, event: QCloseEvent) -> None:
            self.m_timer.stop()
            self.m_controller.close()
            self.m_closed = True
            super().closeEvent(event)

        def prepare_task(self) -> None:
            self.m_controller.prepare_task()
            if self.m_controller.m_state.workflow_state == WorkflowState.LAYOUT_READY:
                self.m_controller.start_episode(0)
            self.render()

        def save_placement(self) -> None:
            try:
                self.m_controller.save_placement()
            except Exception as exc:
                self.m_controller._set_error(str(exc))
            self.render()

        def mark_success(self) -> None:
            try:
                self.m_controller.mark_success()
            except Exception as exc:
                self.m_controller._set_error(str(exc))
            self.render()

        def mark_fail(self) -> None:
            try:
                self.m_controller.mark_fail()
            except Exception as exc:
                self.m_controller._set_error(str(exc))
            self.render()

        def abort_episode(self) -> None:
            try:
                self.m_controller.abort_episode(REASON_OPERATOR_ABORT)
            except Exception as exc:
                self.m_controller._set_error(str(exc))
            self.render()

        def finish_eval_early(self) -> None:
            try:
                self.m_controller.finish_eval_early(REASON_OPERATOR_EARLY_FINISH)
            except Exception as exc:
                self.m_controller._set_error(str(exc))
            self.render()

        def retry_episode(self) -> None:
            try:
                self.m_controller.retry_episode(REASON_OPERATOR_RETRY)
            except Exception as exc:
                self.m_controller._set_error(str(exc))
            self.render()

        def tick(self) -> None:
            try:
                if self.m_controller.m_state.workflow_state in {
                    WorkflowState.PLACEMENT,
                    WorkflowState.EVALUATING,
                    WorkflowState.AWAIT_RESULT,
                }:
                    self.m_controller.update_live_obs()
                self.m_controller.process_eval_events()
            except Exception as exc:
                self.m_controller._set_error(str(exc))
            self.render()

        def render(self) -> None:
            state = self.m_controller.m_state
            self.m_image_view.set_bgr_image(self.m_controller.current_display_image())
            self.m_status_label.setText(
                " | ".join(
                    [
                        f"state: {state.workflow_state.value}",
                        f"episode: {state.active_episode}/{state.target_episode_num}",
                        f"success: {state.success_num}/{state.committed_episode_num}",
                    ]
                )
            )
            self._render_buttons()

            if self.m_rendered_log_count != len(state.logs):
                self.m_log_view.setPlainText("\n".join(state.logs))
                self.m_log_view.verticalScrollBar().setValue(self.m_log_view.verticalScrollBar().maximum())
                self.m_rendered_log_count = len(state.logs)

        def _render_buttons(self) -> None:
            state = self.m_controller.m_state.workflow_state
            self.m_prepare_button.setEnabled(state in {WorkflowState.TASK_INIT, WorkflowState.ERROR})
            self.m_save_button.setEnabled(state == WorkflowState.PLACEMENT)
            self.m_success_button.setEnabled(state == WorkflowState.AWAIT_RESULT)
            self.m_fail_button.setEnabled(state == WorkflowState.AWAIT_RESULT)
            self.m_retry_button.setEnabled(state in {WorkflowState.AWAIT_RESULT, WorkflowState.EPISODE_ABORTED})
            eval_stop_pending = (
                self.m_controller.m_abort_requested or self.m_controller.m_early_finish_requested
            )
            self.m_early_finish_button.setEnabled(
                state == WorkflowState.EVALUATING and not eval_stop_pending
            )
            self.m_abort_button.setEnabled(
                state == WorkflowState.EVALUATING and not eval_stop_pending
            )


class RealEnvWorkbench:
    def __init__(
        self,
        env,
        eval_episode_num: int = 1,
        poll_hz: float = 30.0,
        recorder=None,
    ) -> None:
        self.m_controller = WorkbenchController(
            env=env,
            eval_episode_num=eval_episode_num,
            recorder=recorder,
        )
        self.m_poll_hz = float(poll_hz)
        self.m_app: QApplication | None = None
        self.m_window: WorkbenchWindow | None = None

    def start(self) -> None:
        require_qt()
        configure_qt_plugin_path()
        self.m_app = QApplication.instance() or QApplication(sys.argv[:1])
        self.m_window = WorkbenchWindow(self.m_controller, poll_hz=self.m_poll_hz)
        self.m_window.show()

    def run(self) -> int:
        self.start()
        assert self.m_app is not None
        return int(self.m_app.exec_())

    def process_events(self) -> None:
        if self.m_app is not None:
            self.m_app.processEvents()

    def close(self) -> None:
        if self.m_window is not None:
            self.m_window.close()
        else:
            self.m_controller.close()

    def prepare_task(self) -> None:
        self.m_controller.prepare_task()
        if self.m_controller.m_state.workflow_state == WorkflowState.LAYOUT_READY:
            self.m_controller.start_episode(0)

    def save_placement(self) -> Path:
        return self.m_controller.save_placement()

    def finish_placement(self) -> Path:
        return self.m_controller.finish_placement()

    def mark_success(self) -> None:
        self.m_controller.mark_success()

    def mark_fail(self) -> None:
        self.m_controller.mark_fail()

    def abort_episode(self, reason: str = REASON_OPERATOR_ABORT) -> None:
        self.m_controller.abort_episode(reason)

    def finish_eval_early(self, reason: str = REASON_OPERATOR_EARLY_FINISH) -> None:
        self.m_controller.finish_eval_early(reason)

    def retry_episode(self, reason: str = REASON_OPERATOR_RETRY) -> None:
        self.m_controller.retry_episode(reason)

    def current_display_image(self) -> np.ndarray | None:
        return self.m_controller.current_display_image()
