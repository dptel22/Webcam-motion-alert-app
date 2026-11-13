(() => {
  "use strict";

  const MODE_META = {
    hybrid: {
      label: "Hybrid CVIP+DL",
      summary: "CLAHE + Lucas-Kanade Optical Flow + YOLOv8 + Kalman/Hungarian + Re-ID",
    },
    dl_sort: {
      label: "DL SORT",
      summary: "YOLOv8 Detector + Kalman Filter + Hungarian IoU Association",
    },
    cvip: {
      label: "CVIP MOG2",
      summary: "CLAHE + MOG2 Background Subtraction + Kalman + Hungarian IoU",
    },
  };

  const VIEW_LABELS = {
    annotated: "Tracking Output",
    quad: "2x2 CVIP+DL Quad",
    clahe: "CLAHE Enhanced",
    mog2: "MOG2 Mask",
    flow: "Optical Flow",
  };

  const el = {
    statusBadge: document.getElementById("status-badge"),
    activeModeLabel: document.getElementById("active-mode-label"),
    activeSourceLabel: document.getElementById("active-source-label"),
    streamIndicator: document.getElementById("stream-mode-indicator"),
    streamFeed: document.getElementById("stream-feed"),
    standbyOverlay: document.getElementById("standby-overlay"),
    standbyError: document.getElementById("standby-error"),
    trackTbody: document.getElementById("track-tbody"),
    registryCount: document.getElementById("registry-count"),
    btnStop: document.getElementById("btn-stop"),
    btnUpload: document.getElementById("btn-upload"),
    uploadInput: document.getElementById("upload-input"),
    controlFeedback: document.getElementById("control-feedback"),
    modeSummary: document.getElementById("mode-summary"),
    toggleClahe: document.getElementById("toggle-clahe"),
    toggleReid: document.getElementById("toggle-reid"),
    toggleAlerts: document.getElementById("toggle-alerts"),
    frameCounter: document.getElementById("frame-counter"),
    kpiFps: document.getElementById("kpi-fps"),
    kpiLatency: document.getElementById("kpi-latency"),
    kpiActive: document.getElementById("kpi-active"),
    kpiTotal: document.getElementById("kpi-total"),
    metricsChart: document.getElementById("metrics-chart"),
    tabButtons: Array.from(document.querySelectorAll("[data-view]")),
    modeButtons: Array.from(document.querySelectorAll("[data-mode]")),
    strideButtons: Array.from(document.querySelectorAll("[data-stride]")),
    actionButtons: Array.from(document.querySelectorAll("[data-action]")),
  };

  const state = {
    running: false,
    view: "annotated",
    usingSnapshotFallback: false,
    snapshotTimer: null,
    pollTimer: null,
    recentMetrics: [],
  };

  function setFeedback(message, isError = false) {
    if (!message) {
      el.controlFeedback.hidden = true;
      el.controlFeedback.textContent = "";
      el.controlFeedback.classList.remove("is-error");
      return;
    }
    el.controlFeedback.hidden = false;
    el.controlFeedback.textContent = message;
    el.controlFeedback.classList.toggle("is-error", isError);
  }

  async function requestJson(url, options = {}) {
    const res = await fetch(url, options);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || data.detail || `Request failed (${res.status})`);
    }
    return data;
  }

  function stopSnapshotFallback() {
    if (state.snapshotTimer) {
      clearInterval(state.snapshotTimer);
      state.snapshotTimer = null;
    }
    state.usingSnapshotFallback = false;
  }

  function syncStream(running, view) {
    const streamChanged = running !== state.running || view !== state.view;
    state.running = running;
    state.view = view;

    if (!running) {
      stopSnapshotFallback();
      el.streamFeed.hidden = true;
      el.streamFeed.removeAttribute("src");
      el.standbyOverlay.hidden = false;
      el.streamIndicator.textContent = "Stream: Standby";
      return;
    }

    el.standbyOverlay.hidden = true;
    el.streamFeed.hidden = false;
    el.streamIndicator.textContent = `Stream: ${VIEW_LABELS[view] || view}`;

    if (streamChanged || !el.streamFeed.getAttribute("src")) {
      stopSnapshotFallback();
      el.streamFeed.src = `/stream/mjpeg?view=${encodeURIComponent(view)}&t=${Date.now()}`;
    }
  }

  el.streamFeed.addEventListener("error", () => {
    if (!state.running || state.usingSnapshotFallback) return;
    state.usingSnapshotFallback = true;
    const refreshFrame = () => {
      if (!state.running) return;
      el.streamFeed.src = `/api/frame.jpg?view=${encodeURIComponent(state.view)}&t=${Date.now()}`;
    };
    refreshFrame();
    state.snapshotTimer = setInterval(refreshFrame, 250);
  });

  function renderTracks(tracks) {
    const activeCount = tracks.filter(
      (t) => String(t.state).toLowerCase() === "confirmed"
    ).length;
    el.registryCount.textContent = `${activeCount} active`;
    el.trackTbody.replaceChildren();

    if (!tracks.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 6;
      td.className = "empty-cell";
      td.textContent = state.running
        ? "No active tracks in current frame."
        : "No active tracks. Start webcam or sample demo to populate registry.";
      tr.appendChild(td);
      el.trackTbody.appendChild(tr);
      return;
    }

    for (const track of tracks) {
      const tr = document.createElement("tr");
      const speedVx = Number(track.vx ?? 0).toFixed(1);
      const speedVy = Number(track.vy ?? 0).toFixed(1);
      const confPct = `${(Number(track.confidence ?? 0) * 100).toFixed(0)}%`;
      const stateText = String(track.state || "active");
      const stateKey = stateText.toLowerCase();

      const cells = [
        { text: `#${track.track_id}`, cls: "tabular" },
        { text: track.class_name || "object", cls: "" },
        { text: confPct, cls: "num tabular" },
        { text: `${speedVx}, ${speedVy}`, cls: "num tabular" },
        { text: `${track.age ?? 0}f`, cls: "num tabular" },
      ];

      for (const cell of cells) {
        const td = document.createElement("td");
        if (cell.cls) td.className = cell.cls;
        td.textContent = cell.text;
        tr.appendChild(td);
      }

      const stateTd = document.createElement("td");
      const pill = document.createElement("span");
      let pillClass = "state-confirmed";
      if (stateKey.includes("lost") || stateKey.includes("tentative")) {
        pillClass = "state-tentative";
      } else if (stateKey.includes("alert")) {
        pillClass = "state-alert";
      }
      pill.className = `state-pill ${pillClass}`;
      pill.textContent = stateText;
      stateTd.appendChild(pill);
      tr.appendChild(stateTd);

      el.trackTbody.appendChild(tr);
    }
  }

  function applyStatus(status) {
    if (!status) return;

    const running = Boolean(status.running);
    const mode = status.mode || "hybrid";
    const view = status.view || "annotated";
    const tracks = Array.isArray(status.tracks) ? status.tracks : [];
    const summary = status.summary || {};
    const modeInfo = MODE_META[mode] || MODE_META.hybrid;

    el.statusBadge.textContent = status.error ? "ERROR" : running ? "RUNNING" : "STANDBY";
    el.statusBadge.className = `status-badge ${
      status.error ? "status-error" : running ? "status-running" : "status-standby"
    }`;

    el.activeModeLabel.textContent = `Mode: ${modeInfo.label}`;
    el.activeSourceLabel.textContent = `Source: ${status.source || "Idle"}`;
    el.modeSummary.textContent = modeInfo.summary;

    el.btnStop.disabled = !running;

    if (status.error) {
      el.standbyError.hidden = false;
      el.standbyError.textContent = status.error;
    } else {
      el.standbyError.hidden = true;
      el.standbyError.textContent = "";
    }

    for (const btn of el.tabButtons) {
      const active = btn.dataset.view === view;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", String(active));
    }

    for (const btn of el.modeButtons) {
      const active = btn.dataset.mode === mode;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-checked", String(active));
    }

    for (const btn of el.strideButtons) {
      const active = Number(btn.dataset.stride) === Number(status.detect_interval ?? 2);
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-checked", String(active));
    }

    el.toggleClahe.checked = Boolean(status.use_clahe);
    el.toggleReid.checked = Boolean(status.use_reid);
    el.toggleAlerts.checked = Boolean(status.enable_alerts);

    const activeCount = tracks.filter(
      (t) => String(t.state).toLowerCase() === "confirmed"
    ).length;
    el.frameCounter.textContent = `${Number(summary.total_frames ?? 0)} frames`;
    el.kpiFps.textContent = Number(summary.avg_fps ?? 0).toFixed(1);
    el.kpiLatency.textContent = Number(summary.avg_latency_ms ?? 0).toFixed(1);
    el.kpiActive.textContent = String(activeCount);
    el.kpiTotal.textContent = String(summary.total_tracks_created ?? 0);

    syncStream(running, view);
    renderTracks(tracks);
  }

  function drawChart(recent) {
    const canvas = el.metricsChart;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(rect.width * dpr);
    const height = Math.round(rect.height * dpr);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);

    const pad = { top: 10, right: 34, bottom: 18, left: 32 };
    const plotW = Math.max(10, w - pad.left - pad.right);
    const plotH = Math.max(10, h - pad.top - pad.bottom);

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = Math.round(pad.top + (plotH * i) / 3) + 0.5;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    if (!recent.length) {
      ctx.fillStyle = "#475467";
      ctx.font = '12px "IBM Plex Sans", sans-serif';
      ctx.textAlign = "center";
      ctx.fillText("Waiting for frame telemetry", w / 2, h / 2 + 4);
      return;
    }

    const fpsVals = recent.map((d) => Number(d.fps ?? 0));
    const latVals = recent.map((d) => Number(d.latency_ms ?? 0));
    const maxFps = Math.max(30, Math.ceil(Math.max(...fpsVals) / 10) * 10);
    const maxLat = Math.max(50, Math.ceil(Math.max(...latVals) / 25) * 25);

    ctx.font = '12px "IBM Plex Sans", sans-serif';
    ctx.fillStyle = "#0f4c81";
    ctx.textAlign = "right";
    ctx.fillText(String(maxFps), pad.left - 4, pad.top + 9);
    ctx.fillText("0", pad.left - 4, pad.top + plotH + 3);

    ctx.fillStyle = "#b45309";
    ctx.textAlign = "left";
    ctx.fillText(`${maxLat}ms`, pad.left + plotW + 4, pad.top + 9);
    ctx.fillText("0", pad.left + plotW + 4, pad.top + plotH + 3);

    const plotSeries = (values, maxVal, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.75;
      ctx.beginPath();
      values.forEach((val, idx) => {
        const x =
          values.length === 1
            ? pad.left + plotW
            : pad.left + (idx / (values.length - 1)) * plotW;
        const y = pad.top + plotH - Math.min(1, Math.max(0, val / maxVal)) * plotH;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    plotSeries(latVals, maxLat, "#b45309");
    plotSeries(fpsVals, maxFps, "#0f4c81");
  }

  async function postControl(payload) {
    try {
      setFeedback("");
      const updated = await requestJson("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      applyStatus(updated);
      await refreshMetrics();
    } catch (err) {
      setFeedback(err.message, true);
    }
  }

  async function refreshStatus() {
    const status = await requestJson("/api/status");
    applyStatus(status);
  }

  async function refreshMetrics() {
    const data = await requestJson("/api/metrics?limit=60");
    state.recentMetrics = Array.isArray(data.recent) ? data.recent : [];
    drawChart(state.recentMetrics);
  }

  async function pollLoop() {
    try {
      await Promise.all([refreshStatus(), refreshMetrics()]);
    } catch (err) {
      setFeedback(`Connection error: ${err.message}`, true);
    } finally {
      state.pollTimer = setTimeout(pollLoop, state.running ? 650 : 1400);
    }
  }

  el.actionButtons.forEach((btn) => {
    btn.addEventListener("click", () => postControl({ action: btn.dataset.action }));
  });

  el.tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => postControl({ view: btn.dataset.view }));
  });

  el.modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => postControl({ mode: btn.dataset.mode }));
  });

  el.strideButtons.forEach((btn) => {
    btn.addEventListener("click", () =>
      postControl({ detect_interval: Number(btn.dataset.stride) })
    );
  });

  el.toggleClahe.addEventListener("change", () =>
    postControl({ use_clahe: el.toggleClahe.checked })
  );
  el.toggleReid.addEventListener("change", () =>
    postControl({ use_reid: el.toggleReid.checked })
  );
  el.toggleAlerts.addEventListener("change", () =>
    postControl({ enable_alerts: el.toggleAlerts.checked })
  );

  el.btnUpload.addEventListener("click", () => el.uploadInput.click());
  el.uploadInput.addEventListener("change", async () => {
    const file = el.uploadInput.files?.[0];
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      setFeedback(`Uploading ${file.name}...`);
      const updated = await requestJson("/api/upload", {
        method: "POST",
        body: formData,
      });
      setFeedback(`Loaded ${file.name}`);
      applyStatus(updated);
      await refreshMetrics();
    } catch (err) {
      setFeedback(err.message, true);
    } finally {
      el.uploadInput.value = "";
    }
  });

  window.addEventListener("resize", () => drawChart(state.recentMetrics));

  drawChart([]);
  pollLoop();
})();
