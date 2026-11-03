// Copyright 2026 The RoboRSI Authors. Licensed under the Apache License, Version 2.0.
/* Shared read-only trajectory viewer, also embedded in offline exports. */
(() => {
  const element = (tag, className = "", text = "") => {
    const el = document.createElement(tag);
    el.className = className;
    el.textContent = text;
    return el;
  };
  const number = value => Number(value || 0).toLocaleString();
  const ratio = value => value == null ? "N/A" : `${(100 * value).toFixed(1)}%`;

  class RoboRSITrajectory {
    constructor(root, adapter) {
      this.root = root;
      this.adapter = adapter;
      this.turns = new Map();
      this.requestEpoch = 0;
      root.classList.add("trajectory");
      const header = element("header", "tr-header");
      header.append(element("h1", "", "RoboRSI trajectory"));
      this.status = element("span", "tr-status");
      header.append(this.status);
      this.metrics = element("div", "tr-metrics");
      this.notice = element("p", "tr-notice");
      this.chart = element("div", "tr-chart");
      const workspace = element("div", "tr-workspace");
      const nav = element("aside", "tr-nav");
      const label = element("label", "", "Agent");
      this.filter = element("select");
      this.filter.setAttribute("aria-label", "Filter by agent");
      this.filter.addEventListener("change", () => this.renderTurns());
      label.append(this.filter);
      this.list = element("div", "tr-turns");
      nav.append(label, this.list);
      this.detail = element("section", "tr-detail");
      this.detail.setAttribute("aria-live", "polite");
      this.detail.append(element("p", "tr-empty", "Select a turn to inspect the actual model request and response."));
      workspace.append(nav, this.detail);
      this.media = element("section", "tr-media");
      this.breakdown = element("details", "tr-breakdown");
      root.append(header, this.metrics, this.notice, this.chart, this.breakdown, workspace, this.media);
    }

    update(data) {
      const summary = data.summary;
      if (this.runId !== summary.run_id) {
        this.runId = summary.run_id;
        this.turns.clear();
        this.selected = null;
        this.mediaSignature = null;
        this.requestEpoch++;
        this.detail.replaceChildren(element("p", "tr-empty", "Select a turn to inspect its input and output."));
      }
      for (const turn of data.turns || []) this.turns.set(turn.turn_id, turn);
      this.agents = new Map((data.agents || []).map(agent => [agent.agent_id, agent]));
      const previous = this.filter.value;
      this.filter.replaceChildren(new Option("All agents", ""));
      for (const agent of this.agents.values()) {
        const parent = agent.parent_agent_id ? " ↳ " : "";
        this.filter.append(new Option(parent + (agent.name || agent.agent_id), agent.agent_id));
      }
      this.filter.value = this.agents.has(previous) ? previous : "";
      this.status.textContent = `${summary.status} · ${number(summary.n_turns)} turns · ${number(summary.n_requests)} requests`;
      const usage = summary.usage;
      this.metrics.replaceChildren();
      for (const [label, value] of [
        ["Reported input tokens", number(usage.input_tokens)],
        ["Reported output tokens", number(usage.output_tokens)],
        ["Cache read / input", ratio(usage.cache_ratio)],
        ["Cache writes", number(usage.cache_write_tokens)],
        ["Provider attempts", number(usage.attempts)],
      ]) {
        const item = element("div", "tr-metric");
        item.append(element("span", "", label), element("strong", "", value));
        this.metrics.append(item);
      }
      const mode = data.manifest?.mode || data.manifest?.config?.mode;
      this.notice.textContent = [
        usage.unknown_usage_attempts ? `${usage.unknown_usage_attempts} attempts have no reported usage; totals may be incomplete.` : "",
        mode && mode !== "full" ? `Trace mode: ${mode}. Message bodies were not recorded.` : "",
        summary.status === "partial" ? "This recording is running or incomplete." : "",
        data.manifest?.errors?.length || data.manifest?.status === "incomplete" ? "Some trace records could not be written." : "",
      ].filter(Boolean).join(" ");
      this.renderTurns();
      this.renderChart();
      const previousMediaSignature = this.mediaSignature;
      this.renderMedia(data.media || []);
      this.breakdown.replaceChildren(element("summary", "", "Usage by agent, model and purpose"));
      const table = element("table");
      const head = element("tr");
      for (const name of ["Agent", "Model", "Purpose", "Input", "Output", "Cache"]) head.append(element("th", "", name));
      table.append(head);
      for (const group of summary.groups || []) {
        const row = element("tr");
        for (const value of [this.agents.get(group.agent_id)?.name || group.agent_id, group.model,
          group.purpose, number(group.usage.input_tokens), number(group.usage.output_tokens), ratio(group.usage.cache_ratio)]) {
          row.append(element("td", "", value));
        }
        table.append(row);
      }
      this.breakdown.append(table);
      if (this.selected && (this.turns.get(this.selected)?.last_seq !== this.selectedSeq || previousMediaSignature !== this.mediaSignature)) this.select(this.selected);
    }

    renderTurns() {
      this.list.replaceChildren();
      let index = 0;
      for (const turn of this.turns.values()) {
        index++;
        if (this.filter.value && turn.agent_id !== this.filter.value) continue;
        const button = element("button", `tr-turn${this.selected === turn.turn_id ? " selected" : ""}`);
        button.type = "button";
        button.append(element("strong", "", `${index}. ${this.agents.get(turn.agent_id)?.name || "Agent"}`),
          element("span", "", `${turn.status} · ${number(turn.usage.total_tokens)} tokens · cache ${ratio(turn.usage.cache_ratio)}`));
        button.addEventListener("click", () => this.select(turn.turn_id));
        this.list.append(button);
      }
      if (!this.list.children.length) this.list.append(element("p", "tr-empty", "No recorded turns yet."));
    }

    async select(id) {
      const epoch = ++this.requestEpoch;
      this.selected = id;
      this.selectedSeq = this.turns.get(id)?.last_seq;
      this.renderTurns();
      try {
        const turn = await this.adapter.loadTurn(id);
        if (epoch !== this.requestEpoch) return;
        this.detail.replaceChildren(element("h2", "", `Turn · ${this.agents.get(turn.agent_id)?.name || turn.agent_id || id}`));
        for (const event of turn.events || []) {
          const payload = event.payload || {};
          const block = element("details", "tr-event");
          block.open = ["model_request_start", "model_attempt_end", "tool_end"].includes(event.type);
          const title = event.type.replaceAll("_", " ");
          block.append(element("summary", "", `${title}${event.attempt ? ` · attempt ${event.attempt}` : ""}${payload.status ? ` · ${payload.status}` : ""}`));
          block.append(element("pre", "", JSON.stringify(payload, null, 2)));
          this.appendImages(block, payload);
          if (event.type === "tool_end" && this.mediaItems?.some(item => item.available && item.segments?.some(segment => segment.tool_call_id === event.tool_call_id))) {
            const play = element("button", "tr-play", "Show action video");
            play.addEventListener("click", () => this.showAction(event.tool_call_id));
            block.append(play);
          }
          this.detail.append(block);
        }
      } catch (error) {
        if (epoch === this.requestEpoch) this.detail.replaceChildren(element("p", "tr-error", `Could not load this turn: ${error.message}`));
      }
    }

    appendImages(container, value, seen = new Set()) {
      if (!value || typeof value !== "object") return;
      if (value.artifact_ref && !seen.has(value.artifact_ref) && (
        value.media_type?.startsWith("image/") || /\.(png|jpg|jpeg|webp)$/i.test(value.artifact_ref)
      )) {
        seen.add(value.artifact_ref);
        const url = this.adapter.resolveMedia(value.artifact_ref);
        if (url) {
          const image = element("img", "tr-image");
          image.src = url;
          image.alt = value.artifact_ref;
          image.loading = "lazy";
          container.append(image);
        }
      }
      Object.values(value).forEach(child => this.appendImages(container, child, seen));
    }

    renderChart() {
      this.chart.replaceChildren();
      const turns = [...this.turns.values()];
      if (!turns.length) return;
      this.chart.append(element("span", "", "Cumulative reported tokens"));
      const ns = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(ns, "svg");
      svg.setAttribute("viewBox", "0 0 800 90");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "Cumulative reported token usage by turn");
      const max = Math.max(1, ...turns.map(turn => turn.cumulative_tokens));
      const points = turns.map((turn, i) => [12 + i * 776 / Math.max(1, turns.length - 1), 78 - 65 * turn.cumulative_tokens / max]);
      const line = document.createElementNS(ns, "polyline");
      line.setAttribute("points", points.map(point => point.join(",")).join(" "));
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", "#2f6fd6");
      line.setAttribute("stroke-width", "2");
      svg.append(line);
      turns.forEach((turn, i) => {
        const dot = document.createElementNS(ns, "circle");
        dot.setAttribute("cx", points[i][0]); dot.setAttribute("cy", points[i][1]);
        dot.setAttribute("r", "4"); dot.setAttribute("fill", "#2f6fd6");
        const title = document.createElementNS(ns, "title");
        title.textContent = `Turn ${i + 1}: ${number(turn.cumulative_tokens)} tokens`;
        dot.append(title); svg.append(dot);
      });
      this.chart.append(svg);
    }

    renderMedia(items) {
      this.mediaItems = items;
      const signature = JSON.stringify(items);
      if (signature === this.mediaSignature) return;
      this.mediaSignature = signature;
      this.mediaNodes = new Map();
      this.media.replaceChildren(element("h2", "", "Recorded observations and video"));
      if (!items.length) this.media.append(element("p", "tr-empty", "No media supplied by this run."));
      const grid = element("div", "tr-media-grid");
      for (const item of items) {
        const figure = element("figure");
        const url = item.available && this.adapter.resolveMedia(item.artifact_ref);
        if (url) {
          const video = /\.(mp4|webm)$/i.test(item.artifact_ref);
          const media = element(video ? "video" : "img");
          media.src = url;
          if (video) { media.controls = true; media.preload = "metadata"; this.mediaNodes.set(item.artifact_ref, media); }
          else { media.alt = item.artifact_ref; media.loading = "lazy"; }
          figure.append(media);
        }
        figure.append(element("figcaption", "", `${item.step_idx != null ? `Step ${item.step_idx}: ` : ""}${item.artifact_ref}${url ? "" : ` — ${item.status || "unavailable"}`}`));
        if (item.error) figure.append(element("p", "tr-error", item.error));
        for (const segment of item.segments || []) {
          if (!url || !segment.fps) continue;
          const jump = element("button", "tr-play", `Step ${segment.step_idx}`);
          jump.addEventListener("click", () => this.seek(item.artifact_ref, segment));
          figure.append(jump);
        }
        grid.append(figure);
      }
      this.media.append(grid);
    }

    seek(reference, segment) {
      const video = this.mediaNodes.get(reference);
      if (!video || !segment.fps) return;
      const jump = () => { video.currentTime = segment.frame_start / segment.fps; };
      if (video.readyState) jump(); else video.addEventListener("loadedmetadata", jump, {once: true});
      video.scrollIntoView({block: "center"});
      video.focus();
    }

    showAction(toolCallId) {
      for (const item of this.mediaItems || []) {
        const segment = item.segments?.find(value => value.tool_call_id === toolCallId);
        if (item.available && segment) { this.seek(item.artifact_ref, segment); return; }
      }
    }
  }
  window.RoboRSITrajectory = RoboRSITrajectory;
})();
