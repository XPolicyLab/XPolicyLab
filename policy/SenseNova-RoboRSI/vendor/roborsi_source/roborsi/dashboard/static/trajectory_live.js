// Copyright 2026 The RoboRSI Authors. Licensed under the Apache License, Version 2.0.
/* SSE selects the active run; bodies and media are loaded only when requested. */
const root = document.getElementById("trajectory");
const connection = document.getElementById("trajectoryConnection");
const chinese = document.documentElement.lang === "zh-cn";
const copy = chinese ? {
  title: "RoboRSI · 实时执行轨迹", back: "返回实时监控", waiting: "等待当前任务的轨迹…",
  connected: "已连接", reconnecting: "连接中断，正在重连…", failed: "轨迹读取失败：",
  changed: "当前任务已切换", invalid: "收到无效的轨迹更新",
} : {
  title: "RoboRSI · Live trajectory", back: "Back to live monitor", waiting: "Waiting for this task's trajectory…",
  connected: "Connected", reconnecting: "Disconnected; reconnecting…", failed: "Could not load trajectory: ",
  changed: "The selected task has changed", invalid: "Invalid trajectory update",
};
document.title = copy.title;
document.getElementById("backToMonitor").textContent = copy.back;
connection.textContent = copy.waiting;

let scope = { runId: null, generation: null, epoch: 0, controller: new AbortController() };
let viewer;
let lastSeq = 0;
let targetSeq = 0;
let inFlight = null;

function sameScope(expected) {
  return scope === expected;
}

function identityParams(expected) {
  return new URLSearchParams({ run_id: expected.runId, generation: expected.generation });
}

async function readJSON(url, expected) {
  const response = await fetch(url, { signal: expected.controller.signal, cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || body.error || `${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  if (!sameScope(expected) || data.task_generation !== expected.generation ||
      (data.summary?.run_id ?? data.run_id) !== expected.runId) {
    throw new DOMException(copy.changed, "AbortError");
  }
  return data;
}

function reset(runId, generation) {
  scope.controller.abort();
  scope = { runId, generation, epoch: scope.epoch + 1, controller: new AbortController() };
  const expected = scope;
  lastSeq = 0;
  targetSeq = 0;
  inFlight = null;
  root.replaceChildren();
  viewer = null;
  if (!runId) {
    const message = document.createElement("p");
    message.className = "trajectory";
    message.textContent = copy.waiting;
    root.append(message);
    connection.textContent = copy.waiting;
    return;
  }
  viewer = new window.RoboRSITrajectory(root, {
    loadTurn: id => readJSON(`/api/session/trajectory/turn/${encodeURIComponent(id)}?${identityParams(expected)}`, expected),
    resolveMedia: ref => {
      if (!sameScope(expected)) return null;
      const params = identityParams(expected);
      params.set("ref", ref);
      return `/api/session/trajectory/artifact?${params}`;
    },
  });
}

async function refresh(force = false) {
  if (!scope.runId || inFlight || (!force && lastSeq >= targetSeq)) return;
  const expected = scope;
  const request = { expected };
  inFlight = request;
  const params = identityParams(expected);
  params.set("after_seq", lastSeq);
  let succeeded = false;
  try {
    const data = await readJSON(`/api/session/trajectory?${params}`, expected);
    if (!sameScope(expected)) return;
    viewer.update(data);
    lastSeq = data.summary.last_seq;
    connection.textContent = copy.connected;
    succeeded = true;
  } catch (error) {
    if (sameScope(expected) && error.name !== "AbortError") connection.textContent = copy.failed + error.message;
  } finally {
    if (inFlight === request) {
      inFlight = null;
      // Coalesce updates arriving during the GET, then fetch only its suffix.
      // A failed read waits for the next SSE update/reconnect, avoiding polling.
      if (succeeded && sameScope(expected) && lastSeq < targetSeq) void refresh();
    }
  }
}

reset(null, null);
const events = new EventSource("/api/session/stream");
events.onopen = () => {
  connection.textContent = scope.runId ? copy.connected : copy.waiting;
  void refresh(true);
};
events.onerror = () => { connection.textContent = copy.reconnecting; };
events.onmessage = event => {
  try {
    const snapshot = JSON.parse(event.data);
    const runId = snapshot.trace?.run_id || null;
    const generation = snapshot.task_generation;
    if (runId !== scope.runId || generation !== scope.generation) reset(runId, generation);
    targetSeq = Math.max(targetSeq, snapshot.trace?.last_seq || 0);
    void refresh();
  } catch {
    connection.textContent = copy.invalid;
  }
};
window.addEventListener("pagehide", () => {
  events.close();
  scope.controller.abort();
});
