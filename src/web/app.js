(() => {
  "use strict";

  /* ------------------------------------------------------------------ *
   * Tracking methods and their pipeline stages.
   * ------------------------------------------------------------------ */

  const PRESETS = {
    cvip: {
      mode: "cvip",
      desc: "Follows anything that moves, using background subtraction and optical flow. No neural network — lightest on CPU.",
    },
    dl: {
      mode: "dl_sort",
      desc: "Finds and labels people, cars and other objects with YOLOv8; appearance matching keeps their IDs stable.",
    },
    combined: {
      mode: "hybrid",
      desc: "YOLOv8 detections, with optical flow carrying boxes between them. Most robust, most CPU.",
    },
  };
  const MODE_TO_PRESET = { cvip: "cvip", dl_sort: "dl", hybrid: "combined" };
  const PRESET_ORDER = ["cvip", "dl", "combined"];

  const DEFAULT_STAGES = {
    cvip: ["input", "clahe", "mog2", "flow", "annotated"],
    dl_sort: ["input", "detections", "annotated"],
    hybrid: ["input", "clahe", "flow", "detections", "annotated"],
  };

  const STAGES = {
    input: {
      name: "Input",
      desc: "raw camera frame",
      caption: "Input: the raw camera frame, exactly as captured, before any processing.",
    },
    clahe: {
      name: "Contrast (CLAHE)",
      desc: "evens out lighting",
      caption: "Contrast (CLAHE): local histogram equalisation of the lightness channel evens out shadows and glare.",
    },
    mog2: {
      name: "Motion mask (MOG2)",
      desc: "what moved vs. background",
      caption: "Motion mask (MOG2): white pixels differ from the learned background.",
    },
    flow: {
      name: "Optical flow (Lucas-Kanade)",
      desc: "how pixels moved",
      caption: "Optical flow (Lucas-Kanade): each line shows how a tracked corner point moved since the previous frame.",
    },
    detections: {
      name: "Detections (YOLOv8)",
      desc: "objects found this frame",
      caption: "Detections (YOLOv8): boxes and class scores the neural detector found in this frame alone.",
    },
    annotated: {
      name: "Tracked",
      desc: "Kalman + Hungarian IDs over time",
      caption: "Tracked: Kalman-filtered boxes; Hungarian matching keeps each object’s ID from frame to frame.",
    },
  };
  const QUAD_CAPTION = "Compare all: the pipeline’s intermediate outputs side by side.";

  const COST_NAMES = {
    preprocess: "CLAHE",
    mog2: "MOG2",
    flow: "Flow",
    detect: "YOLOv8",
    reid: "Re-ID",
    track: "Tracking",
    render: "Render",
  };

  const ARROWS = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"];
  const STORAGE_KEY = "tracker.preset";

  const reducedMotion = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : { matches: false };

  /* ------------------------------------------------------------------ *
   * DOM references
   * ------------------------------------------------------------------ */

  const $ = (id) => document.getElementById(id);
  const el = {
    status: $("status"),
    statusWord: $("status-word"),
    source: $("source-label"),
    presetGroup: $("preset-group"),
    presetBtns: Array.from(document.querySelectorAll("[data-preset]")),
    presetDesc: $("preset-desc"),
    frame: $("frame"),
    mainImg: $("main-img"),
    standby: $("standby"),
    caption: $("main-caption"),
    errorLine: $("error-line"),
    strip: $("strip"),
    compare: $("compare-btn"),
    cost: $("cost"),
    costBar: $("cost-bar"),
    costLabels: $("cost-labels"),
    costTotal: $("cost-total"),
    stopBtn: $("stop-btn"),
    uploadBtn: $("upload-btn"),
    uploadInput: $("upload-input"),
    feedback: $("feedback"),
    settings: Array.from(document.querySelectorAll(".setting[data-for]")),
    settingSwitches: Array.from(document.querySelectorAll("[data-setting]")),
    intervalBtns: Array.from(document.querySelectorAll("[data-interval]")),
    nFps: $("n-fps"),
    nLat: $("n-lat"),
    nNow: $("n-now"),
    nTotal: $("n-total"),
    chart: $("chart"),
    tracks: $("tracks"),
    tracksEmpty: $("tracks-empty"),
    objCount: $("obj-count"),
    alertSwitch: $("alert-switch"),
    mailState: $("mail-state"),
    mailNote: $("mail-cvip-note"),
    mailWarn: $("mail-warn"),
    cooldown: $("cooldown"),
    cooldownFill: $("cooldown-fill"),
    cooldownNum: $("cooldown-num"),
    testBtn: $("test-btn"),
    testResult: $("test-result"),
    historyWrap: $("history-wrap"),
    history: $("history"),
    historyEmpty: $("history-empty"),
    dialog: $("snap-dialog"),
    snapImg: $("snap-img"),
    snapCaption: $("snap-caption"),
  };

  /* ------------------------------------------------------------------ *
   * State
   * ------------------------------------------------------------------ */

  const state = {
    status: null,
    running: false,
    mode: null,          // mode as last reported by the server
    view: "annotated",   // view shown in the main image
    stages: [],
    stripKey: "",
    // optimistic local overrides while a POST is in flight
    pendingMode: null,
    pendingView: null,
    pendingSettings: new Set(),
    wantedPreset: null,  // from URL or localStorage, applied once
    presetReconciled: false,
    // request sequencing: never apply an older status after a newer one
    seq: 0,
    appliedSeq: 0,
    // main image
    mainSrcKey: "",
    snapshotMode: false,
    snapshotTimer: 0,
    // loops
    statusTimer: 0,
    statusInFlight: false,
    metricsInFlight: false,
    lastMetricsAt: 0,
    recent: [],
    chartQueued: false,
    alertsTimer: 0,
    alertsInFlight: false,
    alertsAvailable: true,
    alerts: null,
    cooldownEnd: 0,
    cooldownTotal: 0,
    cooldownTimer: 0,
    testBusy: false,
    thumbTimer: 0,
    thumbRunning: false,
    thumbIndex: 0,
    stageEls: new Map(),   // view -> {li, btn, thumb, img, empty, onscreen}
    trackEls: new Map(),   // track_id -> {li, ...}
    trackOrder: "",
    histEls: new Map(),    // alert id -> {li, ...}
    histOrder: "",
    costEls: new Map(),    // key -> {seg, label, num}
    costOrder: "",
    chartColors: null,
  };

  /* ------------------------------------------------------------------ *
   * Small DOM helpers that avoid redundant writes
   * ------------------------------------------------------------------ */

  function setText(node, value) {
    const v = String(value);
    if (node.__t !== v) {
      node.__t = v;
      node.textContent = v;
    }
  }

  function setHidden(node, hidden) {
    if (node.hidden !== hidden) node.hidden = hidden;
  }

  function setAttr(node, name, value) {
    const v = String(value);
    if (node.getAttribute(name) !== v) node.setAttribute(name, v);
  }

  function toggleClass(node, cls, on) {
    if (node.classList.contains(cls) !== on) node.classList.toggle(cls, on);
  }

  function make(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function isVisible() {
    return document.visibilityState === "visible";
  }

  /* ------------------------------------------------------------------ *
   * Network
   * ------------------------------------------------------------------ */

  async function requestJson(url, options) {
    const res = await fetch(url, Object.assign({ cache: "no-store" }, options));
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.error || data.detail || "Request failed (" + res.status + ")");
      err.status = res.status;
      throw err;
    }
    return data;
  }

  function nextSeq() {
    state.seq += 1;
    return state.seq;
  }

  function acceptStatus(seq, status) {
    if (seq < state.appliedSeq) return;
    state.appliedSeq = seq;
    applyStatus(status);
  }

  async function control(payload) {
    const seq = nextSeq();
    const status = await requestJson("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    acceptStatus(seq, status);
    return status;
  }

  function setFeedback(message, isError) {
    setHidden(el.feedback, !message);
    setText(el.feedback, message || "");
    toggleClass(el.feedback, "is-error", Boolean(isError));
  }

  /* ------------------------------------------------------------------ *
   * Tracking method
   * ------------------------------------------------------------------ */

  function readStoredPreset() {
    try {
      const v = window.localStorage.getItem(STORAGE_KEY);
      return PRESETS[v] ? v : null;
    } catch (_e) {
      return null;
    }
  }

  function storePreset(preset) {
    try {
      window.localStorage.setItem(STORAGE_KEY, preset);
    } catch (_e) {
      /* storage unavailable: preference just isn't remembered */
    }
  }

  function readUrlPreset() {
    try {
      const v = new URLSearchParams(window.location.search).get("preset");
      return v && PRESETS[v.toLowerCase()] ? v.toLowerCase() : null;
    } catch (_e) {
      return null;
    }
  }

  function currentMode() {
    return state.pendingMode || state.mode || "hybrid";
  }

  async function choosePreset(preset) {
    const p = PRESETS[preset];
    if (!p) return;
    storePreset(preset);
    const mode = p.mode;
    if (mode === currentMode() && !state.pendingMode) {
      renderPresetUi();
      return;
    }
    state.pendingMode = mode;
    const stages = stagesFor(mode, null);
    const payload = { mode };
    // If the shown stage doesn't exist in the new pipeline, fall back to the tracked output.
    if (state.view !== "quad" && stages.indexOf(state.view) === -1) {
      payload.view = "annotated";
      state.pendingView = "annotated";
      showView("annotated");
    }
    renderPresetUi();
    renderStrip(stages, true);
    setAttr(el.presetGroup, "aria-busy", "true");
    try {
      await control(payload);
    } catch (err) {
      setFeedback("Could not switch method: " + err.message, true);
    } finally {
      state.pendingMode = null;
      state.pendingView = null;
      setAttr(el.presetGroup, "aria-busy", "false");
      if (state.status) applyStatus(state.status);
    }
  }

  function renderPresetUi() {
    const mode = currentMode();
    const preset = MODE_TO_PRESET[mode] || "combined";
    for (const btn of el.presetBtns) {
      setAttr(btn, "aria-checked", btn.dataset.preset === preset);
    }
    setText(el.presetDesc, PRESETS[preset].desc);
    for (const row of el.settings) {
      setHidden(row, row.dataset.for.split(" ").indexOf(mode) === -1);
    }
    setHidden(el.mailNote, mode !== "cvip");
  }

  /* ------------------------------------------------------------------ *
   * Pipeline strip
   * ------------------------------------------------------------------ */

  function stagesFor(mode, serverStages) {
    if (Array.isArray(serverStages) && serverStages.length) {
      return serverStages.filter((s) => typeof s === "string");
    }
    return DEFAULT_STAGES[mode] || DEFAULT_STAGES.hybrid;
  }

  function stageMeta(view) {
    return STAGES[view] || { name: view, desc: "", caption: view };
  }

  function renderStrip(stages, animate) {
    const key = stages.join(",");
    if (key === state.stripKey) return;
    state.stripKey = key;
    state.stages = stages.slice();
    state.stageEls.clear();
    state.thumbIndex = 0;

    const frag = document.createDocumentFragment();
    stages.forEach((view, i) => {
      if (i > 0) {
        const arrow = make("li", "arrow", "→");
        arrow.setAttribute("aria-hidden", "true");
        frag.appendChild(arrow);
      }
      const meta = stageMeta(view);
      const li = make("li", "stage");
      const btn = make("button", "stage-btn");
      btn.type = "button";
      btn.dataset.view = view;
      btn.setAttribute("aria-pressed", "false");
      btn.setAttribute("aria-label", "Stage " + (i + 1) + ": " + meta.name + " — " + meta.desc);

      const thumb = make("span", "thumb");
      const empty = make("span", "thumb-empty", "no frame");
      const onscreen = make("span", "thumb-onscreen", "shown above");
      onscreen.hidden = true;
      thumb.append(empty, onscreen);

      const label = make("span", "stage-label");
      label.append(make("span", "stage-num", String(i + 1)), make("span", "stage-name", meta.name));
      const desc = make("span", "stage-desc", meta.desc);

      btn.append(thumb, label, desc);
      li.appendChild(btn);
      frag.appendChild(li);
      state.stageEls.set(view, { li, btn, thumb, img: null, empty, onscreen });
    });

    el.strip.replaceChildren(frag);
    syncStripSelection();
    measureThumbs();

    if (animate && !reducedMotion.matches && el.strip.animate) {
      el.strip.animate(
        [{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }],
        { duration: 220, easing: "cubic-bezier(.2,.7,.2,1)" }
      );
    }
  }

  function syncStripSelection() {
    for (const [view, s] of state.stageEls) {
      const on = view === state.view;
      setAttr(s.btn, "aria-pressed", on);
      setHidden(s.onscreen, !(on && state.running && s.img));
      setText(s.empty, on && state.running ? "shown above" : "no frame");
    }
    setAttr(el.compare, "aria-pressed", state.view === "quad");
    setText(el.compare, state.view === "quad" ? "Back to single view" : "Compare all");
  }

  function clearThumbs() {
    for (const s of state.stageEls.values()) {
      if (s.img) {
        s.img.remove();
        s.img = null;
      }
      setHidden(s.empty, false);
      setHidden(s.onscreen, true);
    }
  }

  // Arrows sit at the vertical middle of the thumbnails; thumbnail height depends on width.
  let thumbObserver = null;
  function measureThumbs() {
    const first = state.stageEls.values().next().value;
    if (!first) return;
    const apply = () => {
      const h = first.thumb.offsetHeight;
      if (h) el.strip.style.setProperty("--thumb-h", h + "px");
    };
    if (window.ResizeObserver) {
      if (thumbObserver) thumbObserver.disconnect();
      thumbObserver = new ResizeObserver(apply);
      thumbObserver.observe(first.thumb);
    } else {
      requestAnimationFrame(apply);
    }
  }

  /* Thumbnails: ONE sequential round-robin loop over stages; each request waits for
   * the previous image to load and decode. Capped so no stage updates more than ~4/s. */
  function thumbsWanted() {
    return state.running && isVisible() && state.stages.length > 0;
  }

  function startThumbLoop() {
    if (state.thumbRunning || !thumbsWanted()) return;
    state.thumbRunning = true;
    clearTimeout(state.thumbTimer);
    thumbStep();
  }

  async function thumbStep() {
    if (!thumbsWanted()) {
      state.thumbRunning = false;
      return;
    }
    const candidates = state.stages.filter((v) => v !== state.view);
    if (!candidates.length) {
      state.thumbTimer = setTimeout(thumbStep, 400);
      return;
    }
    const view = candidates[state.thumbIndex % candidates.length];
    state.thumbIndex = (state.thumbIndex + 1) % candidates.length;
    const minSpacing = 250 / candidates.length; // => each stage at most ~4 updates/s
    const t0 = performance.now();
    const url = "/api/frame.jpg?view=" + encodeURIComponent(view) + "&w=240&t=" + Date.now();

    try {
      const img = new Image();
      img.alt = "";
      img.decoding = "async";
      img.src = url;
      await img.decode();
      const s = state.stageEls.get(view);
      // Stage may have disappeared (preset change) or we stopped meanwhile.
      if (s && state.running) {
        img.alt = stageMeta(view).name + " output";
        if (s.img) s.img.replaceWith(img);
        else s.thumb.insertBefore(img, s.thumb.firstChild);
        s.img = img;
        setHidden(s.empty, true);
      }
    } catch (_e) {
      /* frame not available yet; try again next round */
    }

    const elapsed = performance.now() - t0;
    state.thumbTimer = setTimeout(thumbStep, Math.max(16, minSpacing - elapsed));
  }

  /* ------------------------------------------------------------------ *
   * Main view: MJPEG with snapshot fallback
   * ------------------------------------------------------------------ */

  function updateCaption() {
    const idx = state.stages.indexOf(state.view);
    el.caption.replaceChildren();
    let figLabel;
    let text;
    if (state.view === "quad") {
      figLabel = "Fig. 1–" + Math.max(1, state.stages.length);
      text = QUAD_CAPTION;
    } else {
      figLabel = "Fig. " + (idx >= 0 ? idx + 1 : "–");
      text = stageMeta(state.view).caption;
    }
    el.caption.append(make("span", "fig", figLabel), document.createTextNode(" — " + text));
    el.mainImg.alt = state.view === "quad" ? "All pipeline stages side by side" : stageMeta(state.view).name + " view of the live video";
  }

  function stopSnapshot() {
    clearTimeout(state.snapshotTimer);
    state.snapshotTimer = 0;
    state.snapshotMode = false;
  }

  function syncMainImage() {
    const shouldStream = state.running && isVisible();
    if (!shouldStream) {
      stopSnapshot();
      if (state.mainSrcKey) {
        el.mainImg.removeAttribute("src"); // closes the MJPEG connection
        state.mainSrcKey = "";
      }
      setHidden(el.mainImg, !state.running);
      setHidden(el.standby, state.running);
      return;
    }
    setHidden(el.standby, true);
    setHidden(el.mainImg, false);
    const key = "mjpeg:" + state.view;
    if (state.mainSrcKey === key || (state.snapshotMode && state.mainSrcKey === "snap:" + state.view)) return;
    stopSnapshot();
    state.mainSrcKey = key;
    el.mainImg.src = "/stream/mjpeg?view=" + encodeURIComponent(state.view) + "&t=" + Date.now();
  }

  function snapshotTick() {
    if (!state.snapshotMode || !state.running || !isVisible()) return;
    const view = state.view;
    const img = new Image();
    img.src = "/api/frame.jpg?view=" + encodeURIComponent(view) + "&t=" + Date.now();
    img.decode().then(
      () => {
        if (state.snapshotMode && state.view === view) el.mainImg.src = img.src;
        state.snapshotTimer = setTimeout(snapshotTick, 80);
      },
      () => {
        state.snapshotTimer = setTimeout(snapshotTick, 500);
      }
    );
  }

  el.mainImg.addEventListener("error", () => {
    if (!state.running || state.snapshotMode || !state.mainSrcKey) return;
    state.snapshotMode = true;
    state.mainSrcKey = "snap:" + state.view;
    snapshotTick();
  });

  function showView(view) {
    if (view === state.view) return;
    state.view = view;
    if (state.snapshotMode) state.mainSrcKey = "snap:" + view;
    updateCaption();
    syncStripSelection();
    syncMainImage();
  }

  async function selectView(view) {
    if (!view) return;
    state.pendingView = view;
    showView(view);
    try {
      await control({ view });
    } catch (err) {
      setFeedback("Could not change view: " + err.message, true);
    } finally {
      state.pendingView = null;
      if (state.status) applyStatus(state.status);
    }
  }

  /* ------------------------------------------------------------------ *
   * Status
   * ------------------------------------------------------------------ */

  function describeSource(src) {
    if (!src) return "No source";
    const s = String(src);
    if (s === "sample") return "Sample clip";
    if (/^\d+$/.test(s)) return "Webcam " + s;
    const parts = s.split(/[\\/]/);
    return parts[parts.length - 1] || s;
  }

  function errorAdvice(msg) {
    const m = msg.toLowerCase();
    if (m.indexOf("failed to open") !== -1 && /source: \d+/.test(m)) {
      return "Close other apps using the camera, or run the sample clip instead.";
    }
    if (m.indexOf("failed to open") !== -1) return "Check the file, or run the sample clip instead.";
    if (m.indexOf("ended") !== -1) return "Start the source again, or choose another one.";
    return "Press Reset tracker and start the source again.";
  }

  function applyStatus(status) {
    if (!status || typeof status !== "object") return;
    const isFirst = !state.status;
    state.status = status;
    const wasRunning = state.running;
    state.running = Boolean(status.running);
    state.mode = status.mode || state.mode || "hybrid";

    // First status after load: the remembered / URL preset wins if the server differs.
    let reapply = null;
    if (!state.presetReconciled) {
      state.presetReconciled = true;
      const wanted = state.wantedPreset;
      if (wanted && PRESETS[wanted].mode !== state.mode) {
        reapply = wanted;
        state.pendingMode = PRESETS[wanted].mode;
      }
    }

    // Header status
    const word = status.error && !state.running ? "Error" : state.running ? "Running" : "Stopped";
    setAttr(el.status, "data-state", word.toLowerCase());
    setText(el.statusWord, word);
    setText(el.source, state.running ? describeSource(status.source) : "No source");
    el.stopBtn.disabled = !state.running;

    // Error near main view
    if (status.error) {
      if (el.errorLine.__t !== status.error) {
        el.errorLine.__t = status.error;
        const strong = make("strong", null, "Problem: ");
        el.errorLine.replaceChildren(
          strong,
          document.createTextNode(String(status.error).replace(/\.\s*$/, "") + ". " + errorAdvice(String(status.error)))
        );
      }
      setHidden(el.errorLine, false);
    } else {
      setHidden(el.errorLine, true);
    }

    // Preset (respect an in-flight switch)
    renderPresetUi();
    const mode = currentMode();
    if (!state.pendingMode) renderStrip(stagesFor(mode, status.stages), !isFirst);

    // Settings
    for (const sw of el.settingSwitches) {
      const k = sw.dataset.setting;
      if (!state.pendingSettings.has(k)) setAttr(sw, "aria-checked", Boolean(status[k]));
    }
    if (!state.pendingSettings.has("detect_interval")) {
      const n = Number(status.detect_interval || 1);
      for (const b of el.intervalBtns) setAttr(b, "aria-checked", Number(b.dataset.interval) === n);
    }
    if (!state.alerts) {
      // Alerts endpoint unavailable (or not polled yet): fall back to status.
      if (!state.pendingSettings.has("enable_alerts")) {
        setAttr(el.alertSwitch, "aria-checked", Boolean(status.enable_alerts));
      }
      setText(
        el.mailState,
        el.alertSwitch.getAttribute("aria-checked") === "true"
          ? "Armed — an email is sent when a matching object is tracked."
          : "Off — no emails will be sent."
      );
    }

    // View
    if (!state.pendingView && status.view && status.view !== state.view) {
      state.view = status.view;
      syncStripSelection();
    }
    updateCaption();
    syncMainImage();
    syncStripSelection();

    if (state.running && !wasRunning) {
      startThumbLoop();
    } else if (!state.running && wasRunning) {
      clearThumbs();
    }

    // Numbers + tracks + stage cost
    const tracks = Array.isArray(status.tracks) ? status.tracks : [];
    const summary = status.summary || {};
    const confirmed = tracks.filter((t) => String(t.state).toLowerCase() === "confirmed").length;
    setText(el.nNow, confirmed);
    setText(el.nTotal, Number(summary.total_tracks_created || 0));
    if (!state.running || !state.recent.length) {
      setText(el.nFps, summary.total_frames ? Number(summary.avg_fps || 0).toFixed(1) : "–");
      setText(el.nLat, summary.total_frames ? Number(summary.avg_latency_ms || 0).toFixed(1) : "–");
    }
    renderTracks(tracks);
    renderCost(status.stage_ms);

    if (reapply) choosePreset(reapply);
  }

  function scheduleStatus(delay) {
    clearTimeout(state.statusTimer);
    state.statusTimer = setTimeout(pollStatus, delay);
  }

  async function pollStatus() {
    if (!isVisible()) return; // resumed by visibilitychange
    if (state.statusInFlight) return;
    state.statusInFlight = true;
    const seq = nextSeq();
    try {
      const status = await requestJson("/api/status");
      acceptStatus(seq, status);
      if (el.feedback.__conn) {
        el.feedback.__conn = false;
        setFeedback("");
      }
    } catch (err) {
      el.feedback.__conn = true;
      setFeedback("Can’t reach the server (" + err.message + "). Retrying…", true);
    } finally {
      state.statusInFlight = false;
    }
    if (state.running) pollMetrics();
    scheduleStatus(state.running ? 700 : 2000);
  }

  /* ------------------------------------------------------------------ *
   * Metrics + chart
   * ------------------------------------------------------------------ */

  async function pollMetrics() {
    const now = performance.now();
    if (state.metricsInFlight || now - state.lastMetricsAt < 900) return;
    state.metricsInFlight = true;
    state.lastMetricsAt = now;
    try {
      const data = await requestJson("/api/metrics?limit=60");
      state.recent = Array.isArray(data.recent) ? data.recent : [];
      const tail = state.recent.slice(-10);
      if (tail.length) {
        const avg = (k) => tail.reduce((a, r) => a + Number(r[k] || 0), 0) / tail.length;
        setText(el.nFps, avg("fps").toFixed(1));
        setText(el.nLat, avg("latency_ms").toFixed(1));
      }
      queueChart();
    } catch (_e) {
      /* status loop reports connectivity */
    } finally {
      state.metricsInFlight = false;
    }
  }

  function readChartColors() {
    const cs = getComputedStyle(document.documentElement);
    state.chartColors = {
      ink: cs.getPropertyValue("--ink").trim() || "#161614",
      muted: cs.getPropertyValue("--muted").trim() || "#74726b",
      rule: cs.getPropertyValue("--rule").trim() || "#d9d7cf",
    };
  }

  function queueChart() {
    if (state.chartQueued) return;
    state.chartQueued = true;
    requestAnimationFrame(() => {
      state.chartQueued = false;
      drawChart();
    });
  }

  function drawChart() {
    const canvas = el.chart;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    if (!state.chartColors) readChartColors();
    const c = state.chartColors;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(w * dpr);
    const H = Math.round(h * dpr);
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = { t: 6, r: 34, b: 6, l: 26 };
    const pw = Math.max(10, w - pad.l - pad.r);
    const ph = Math.max(10, h - pad.t - pad.b);

    ctx.strokeStyle = c.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + ph + 0.5);
    ctx.lineTo(pad.l + pw, pad.t + ph + 0.5);
    ctx.stroke();

    const recent = state.recent;
    ctx.font = '10.5px "IBM Plex Mono", monospace';
    if (!recent.length) {
      ctx.fillStyle = c.muted;
      ctx.textAlign = "center";
      ctx.fillText("no data yet", pad.l + pw / 2, pad.t + ph / 2 + 4);
      return;
    }

    const fps = recent.map((d) => Number(d.fps || 0));
    const lat = recent.map((d) => Number(d.latency_ms || 0));
    const maxFps = Math.max(30, Math.ceil(Math.max.apply(null, fps) / 10) * 10);
    const maxLat = Math.max(50, Math.ceil(Math.max.apply(null, lat) / 25) * 25);

    ctx.fillStyle = c.ink;
    ctx.textAlign = "right";
    ctx.fillText(String(maxFps), pad.l - 4, pad.t + 8);
    ctx.fillStyle = c.muted;
    ctx.textAlign = "left";
    ctx.fillText(maxLat + "ms", pad.l + pw + 4, pad.t + 8);

    const line = (vals, max, color, dash) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dash);
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = vals.length === 1 ? pad.l + pw : pad.l + (i / (vals.length - 1)) * pw;
        const y = pad.t + ph - Math.min(1, Math.max(0, v / max)) * ph;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line(lat, maxLat, c.muted, [3, 3]);
    line(fps, maxFps, c.ink, []);
  }

  /* ------------------------------------------------------------------ *
   * Stage cost bar (keyed reuse)
   * ------------------------------------------------------------------ */

  function renderCost(stageMs) {
    if (!stageMs || typeof stageMs !== "object") {
      setHidden(el.cost, true);
      return;
    }
    const entries = Object.keys(stageMs)
      .map((k) => [k, Number(stageMs[k])])
      .filter((e) => Number.isFinite(e[1]) && e[1] >= 0);
    if (!entries.length) {
      setHidden(el.cost, true);
      return;
    }
    setHidden(el.cost, false);

    const order = entries.map((e) => e[0]).join(",");
    if (order !== state.costOrder) {
      state.costOrder = order;
      state.costEls.clear();
      const segs = document.createDocumentFragment();
      const labels = document.createDocumentFragment();
      for (const [k] of entries) {
        const seg = make("span", "cost-seg");
        const label = make("span", "cost-label");
        const name = make("b", null, COST_NAMES[k] || k);
        const num = make("span");
        label.append(name, num);
        segs.appendChild(seg);
        labels.appendChild(label);
        state.costEls.set(k, { seg, label, num });
      }
      el.costBar.replaceChildren(segs);
      el.costLabels.replaceChildren(labels);
    }

    let total = 0;
    let maxKey = null;
    let maxVal = -1;
    for (const [k, v] of entries) {
      total += v;
      if (v > maxVal) {
        maxVal = v;
        maxKey = k;
      }
    }
    const summary = [];
    for (const [k, v] of entries) {
      const parts = state.costEls.get(k);
      const grow = Math.max(v, total * 0.004).toFixed(2);
      if (parts.seg.__g !== grow) {
        parts.seg.__g = grow;
        parts.seg.style.flexGrow = grow;
        parts.label.style.flexGrow = grow;
      }
      const ms = v.toFixed(1);
      setText(parts.num, ms);
      const title = (COST_NAMES[k] || k) + ": " + ms + " ms";
      if (parts.seg.title !== title) {
        parts.seg.title = title;
        parts.label.title = title;
      }
      toggleClass(parts.seg, "is-max", k === maxKey && entries.length > 1);
      summary.push(title);
    }
    setText(el.costTotal, "= " + total.toFixed(1) + " ms");
    setAttr(el.costBar, "aria-label", "Stage cost per frame. " + summary.join(", ") + ". Total " + total.toFixed(1) + " ms.");
  }

  /* ------------------------------------------------------------------ *
   * Track list (keyed reuse)
   * ------------------------------------------------------------------ */

  function directionArrow(vx, vy) {
    if (Math.hypot(vx, vy) < 0.3) return "·";
    const a = Math.atan2(vy, vx); // y axis points down on screen
    const idx = ((Math.round(a / (Math.PI / 4)) % 8) + 8) % 8;
    return ARROWS[idx];
  }

  function renderTracks(tracks) {
    const sorted = tracks.slice().sort((a, b) => {
      const ca = String(a.state).toLowerCase() === "confirmed" ? 0 : 1;
      const cb = String(b.state).toLowerCase() === "confirmed" ? 0 : 1;
      return ca - cb || Number(a.track_id) - Number(b.track_id);
    });

    const seen = new Set();
    for (const t of sorted) {
      const id = String(t.track_id);
      seen.add(id);
      let r = state.trackEls.get(id);
      if (!r) {
        const li = make("li", "track");
        const dot = make("span", "track-dot");
        dot.style.background = "hsl(" + ((Number(t.track_id) * 137.508) % 360).toFixed(1) + ", 58%, 44%)";
        const idSpan = make("span", "track-id");
        const num = make("span", null, "#" + String(t.track_id).padStart(3, "0"));
        const cls = make("span", "cls");
        idSpan.append(num, cls);
        const meta = make("span", "track-meta");
        const st = make("span", "track-state");
        li.append(dot, idSpan, meta, st);
        r = { li, cls, meta, st };
        state.trackEls.set(id, r);
      }
      const vx = Number(t.vx || 0);
      const vy = Number(t.vy || 0);
      const speed = Math.hypot(vx, vy);
      setText(r.cls, t.class_name || "object");
      setText(
        r.meta,
        Math.round(Number(t.confidence || 0) * 100) + "%  " + speed.toFixed(1) + " " + directionArrow(vx, vy)
      );
      const stateWord = String(t.state || "").toLowerCase();
      setText(r.st, stateWord);
      toggleClass(r.li, "is-lost", stateWord === "lost");
      r.li.title = "Speed " + speed.toFixed(1) + " px/frame, age " + Number(t.age || 0) + " frames";
    }

    for (const [id, r] of state.trackEls) {
      if (!seen.has(id)) {
        r.li.remove();
        state.trackEls.delete(id);
      }
    }

    const order = sorted.map((t) => t.track_id).join(",");
    if (order !== state.trackOrder) {
      state.trackOrder = order;
      // Re-append in order; existing nodes are moved, not recreated.
      for (const t of sorted) el.tracks.appendChild(state.trackEls.get(String(t.track_id)).li);
    }

    setHidden(el.tracksEmpty, sorted.length > 0);
    setText(el.tracksEmpty, state.running ? "Nothing tracked in this frame." : "Start the webcam to see tracks.");
    setText(el.objCount, sorted.length ? String(sorted.length) : "");
  }

  /* ------------------------------------------------------------------ *
   * Email alerts
   * ------------------------------------------------------------------ */

  function pad2(n) {
    return String(n).padStart(2, "0");
  }

  function clock(epochSec) {
    const d = new Date(Number(epochSec) * 1000);
    if (isNaN(d.getTime())) return "--:--:--";
    return pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds());
  }

  function joinOr(list) {
    if (!list.length) return "an object";
    const withArticle = list.map((c) => (/^[aeiou]/i.test(c) ? "an " : "a ") + c);
    if (withArticle.length === 1) return withArticle[0];
    return withArticle.slice(0, -1).join(", ") + " or " + withArticle[withArticle.length - 1];
  }

  function applyAlerts(a) {
    state.alerts = a;
    const enabled = Boolean(a.enabled);
    if (!state.pendingSettings.has("enable_alerts")) setAttr(el.alertSwitch, "aria-checked", enabled);
    const shownEnabled = el.alertSwitch.getAttribute("aria-checked") === "true";
    const classes = Array.isArray(a.trigger_classes) ? a.trigger_classes.map(String) : [];
    const sentence = shownEnabled
      ? "Armed — emails " + (a.recipient || "the configured address") + " when " + joinOr(classes) + " is tracked."
      : "Off — no emails will be sent.";
    setText(el.mailState, sentence);
    setHidden(el.mailWarn, a.configured !== false);

    // Cooldown: server gives remaining seconds; tick locally between polls.
    const remaining = Number(a.cooldown_remaining_sec || 0);
    state.cooldownTotal = Number(a.cooldown_sec || 0);
    state.cooldownEnd = remaining > 0 ? performance.now() + remaining * 1000 : 0;
    tickCooldown();

    renderHistory(Array.isArray(a.history) ? a.history : []);
  }

  function tickCooldown() {
    clearTimeout(state.cooldownTimer);
    const left = state.cooldownEnd ? (state.cooldownEnd - performance.now()) / 1000 : 0;
    if (left <= 0 || !state.cooldownTotal) {
      setHidden(el.cooldown, true);
      return;
    }
    setHidden(el.cooldown, false);
    setText(el.cooldownNum, Math.ceil(left) + " s");
    el.cooldownFill.style.transform = "scaleX(" + Math.min(1, left / state.cooldownTotal).toFixed(3) + ")";
    if (isVisible()) state.cooldownTimer = setTimeout(tickCooldown, 250);
  }

  function statusText(rec) {
    if (rec.status === "sent") return ["Sent", "st-sent"];
    if (rec.status === "failed") return ["Failed" + (rec.error ? " — " + rec.error : ""), "st-failed"];
    return ["Sending…", "st-sending"];
  }

  function renderHistory(history) {
    const items = history.slice().sort((a, b) => Number(b.time) - Number(a.time)).slice(0, 20);
    const seen = new Set();
    for (const rec of items) {
      const id = String(rec.id);
      seen.add(id);
      let r = state.histEls.get(id);
      if (!r) {
        const li = make("li", "hist");
        const time = make("span", "hist-time", clock(rec.time));
        const snap = make("button", "hist-snap");
        snap.type = "button";
        if (rec.snapshot_url) {
          const img = make("img");
          img.loading = "lazy";
          img.alt = "";
          img.src = rec.snapshot_url;
          snap.appendChild(img);
          snap.setAttribute("aria-label", "Open snapshot from " + clock(rec.time));
          snap.addEventListener("click", () => openSnapshot(rec.snapshot_url, clock(rec.time), rec.test));
        } else {
          snap.disabled = true;
          snap.setAttribute("aria-label", "No snapshot");
        }
        const status = make("span", "hist-status");
        const statusWord = make("span");
        status.appendChild(statusWord);
        if (rec.test) status.appendChild(make("span", "tag", "test"));
        li.append(time, snap, status);
        r = { li, statusWord, status };
        state.histEls.set(id, r);
      }
      const [txt, cls] = statusText(rec);
      setText(r.statusWord, txt);
      if (r.statusWord.className !== cls) r.statusWord.className = cls;
      if (r.status.title !== txt) r.status.title = txt;
    }
    for (const [id, r] of state.histEls) {
      if (!seen.has(id)) {
        r.li.remove();
        state.histEls.delete(id);
      }
    }
    const order = items.map((i) => i.id).join(",");
    if (order !== state.histOrder) {
      state.histOrder = order;
      for (const rec of items) el.history.appendChild(state.histEls.get(String(rec.id)).li);
    }
    setHidden(el.historyEmpty, items.length > 0);
  }

  function openSnapshot(url, time, isTest) {
    el.snapImg.src = url;
    el.snapImg.alt = "Snapshot attached to the alert at " + time;
    el.snapCaption.replaceChildren(
      make("span", "fig", "Alert snapshot"),
      document.createTextNode(" — frame emailed at " + time + (isTest ? " (test alert)." : "."))
    );
    if (typeof el.dialog.showModal === "function") el.dialog.showModal();
    else el.dialog.setAttribute("open", "");
  }

  el.dialog.addEventListener("click", (e) => {
    if (e.target === el.dialog) el.dialog.close();
  });

  function scheduleAlerts(delay) {
    clearTimeout(state.alertsTimer);
    state.alertsTimer = setTimeout(pollAlerts, delay);
  }

  async function pollAlerts() {
    if (!isVisible()) return;
    if (state.alertsInFlight) return;
    state.alertsInFlight = true;
    try {
      applyAlerts(await requestJson("/api/alerts"));
      state.alertsAvailable = true;
      setHidden(el.historyWrap, false);
      setHidden(el.testBtn.parentNode, false);
    } catch (err) {
      if (err.status === 404) {
        state.alertsAvailable = false;
        setHidden(el.historyWrap, true);
        setHidden(el.testBtn.parentNode, true);
      }
    } finally {
      state.alertsInFlight = false;
    }
    scheduleAlerts(state.alertsAvailable ? 2000 : 15000);
  }

  async function sendTest() {
    if (state.testBusy) return;
    state.testBusy = true;
    el.testBtn.disabled = true;
    el.testBtn.setAttribute("aria-busy", "true");
    setText(el.testBtn, "Sending…");
    setText(el.testResult, "");
    el.testResult.className = "test-result";
    try {
      const a = await requestJson("/api/alerts/test", { method: "POST" });
      applyAlerts(a);
      const latest = (Array.isArray(a.history) ? a.history : [])
        .filter((h) => h.test)
        .sort((x, y) => Number(y.time) - Number(x.time))[0];
      if (!latest) {
        setText(el.testResult, "Requested.");
      } else {
        const [txt, cls] = statusText(latest);
        setText(el.testResult, latest.status === "sending" ? "Sending — check History." : "Test " + txt.charAt(0).toLowerCase() + txt.slice(1) + ".");
        el.testResult.className = "test-result " + cls;
      }
    } catch (err) {
      setText(el.testResult, "Failed — " + err.message);
      el.testResult.className = "test-result st-failed";
    } finally {
      state.testBusy = false;
      el.testBtn.disabled = false;
      el.testBtn.removeAttribute("aria-busy");
      setText(el.testBtn, "Send test alert");
    }
  }

  /* ------------------------------------------------------------------ *
   * Controls
   * ------------------------------------------------------------------ */

  async function runAction(action) {
    const label = { sample: "Starting sample clip…", webcam: "Starting webcam…", stop: "Stopping…", reset: "Resetting tracker…" }[action];
    setFeedback(label || "");
    try {
      await control({ action });
      setFeedback(action === "reset" ? "Tracker reset: IDs and background model cleared." : "");
      if (state.running) {
        state.lastMetricsAt = 0;
        pollMetrics();
      }
      scheduleStatus(state.running ? 700 : 2000);
    } catch (err) {
      setFeedback(err.message, true);
    }
  }

  async function toggleSetting(key, sw) {
    const next = sw.getAttribute("aria-checked") !== "true";
    state.pendingSettings.add(key);
    setAttr(sw, "aria-checked", next);
    if (key === "enable_alerts" && state.alerts) applyAlerts(state.alerts);
    try {
      await control({ [key]: next });
    } catch (err) {
      setAttr(sw, "aria-checked", !next);
      setFeedback("Could not change setting: " + err.message, true);
    } finally {
      state.pendingSettings.delete(key);
      if (key === "enable_alerts" && state.alertsAvailable) pollAlerts();
    }
  }

  document.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => runAction(btn.dataset.action));
  });

  el.presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => choosePreset(btn.dataset.preset));
  });

  // Arrow keys within the preset radiogroup
  el.presetGroup.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = el.presetBtns.indexOf(document.activeElement);
    if (i === -1) return;
    e.preventDefault();
    const n = el.presetBtns[(i + (e.key === "ArrowRight" ? 1 : -1) + el.presetBtns.length) % el.presetBtns.length];
    n.focus();
    choosePreset(n.dataset.preset);
  });

  el.strip.addEventListener("click", (e) => {
    const btn = e.target.closest(".stage-btn");
    if (btn) selectView(btn.dataset.view);
  });

  el.compare.addEventListener("click", () => {
    selectView(state.view === "quad" ? "annotated" : "quad");
  });

  el.settingSwitches.forEach((sw) => {
    sw.addEventListener("click", () => toggleSetting(sw.dataset.setting, sw));
  });
  el.alertSwitch.addEventListener("click", () => toggleSetting("enable_alerts", el.alertSwitch));

  el.intervalBtns.forEach((btn) => {
    btn.addEventListener("click", async () => {
      const n = Number(btn.dataset.interval);
      state.pendingSettings.add("detect_interval");
      for (const b of el.intervalBtns) setAttr(b, "aria-checked", b === btn);
      try {
        await control({ detect_interval: n });
      } catch (err) {
        setFeedback("Could not change setting: " + err.message, true);
      } finally {
        state.pendingSettings.delete("detect_interval");
        if (state.status) applyStatus(state.status);
      }
    });
  });

  el.testBtn.addEventListener("click", sendTest);

  el.uploadBtn.addEventListener("click", () => el.uploadInput.click());
  el.uploadInput.addEventListener("change", async () => {
    const file = el.uploadInput.files && el.uploadInput.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    el.uploadBtn.disabled = true;
    setFeedback("Uploading " + file.name + "…");
    const seq = nextSeq();
    try {
      const status = await requestJson("/api/upload", { method: "POST", body: form });
      acceptStatus(seq, status);
      setFeedback("Playing " + file.name + ".");
      scheduleStatus(700);
    } catch (err) {
      setFeedback(err.message, true);
    } finally {
      el.uploadBtn.disabled = false;
      el.uploadInput.value = "";
    }
  });

  /* Keyboard: 1-5 stages, M cycles tracking method, Space start / stop webcam. */
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    const t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    if (el.dialog.open) return;
    const key = e.key.toLowerCase();

    if (key === "m") {
      e.preventDefault();
      const cur = MODE_TO_PRESET[state.pendingMode || state.mode] || "combined";
      choosePreset(PRESET_ORDER[(PRESET_ORDER.indexOf(cur) + 1) % PRESET_ORDER.length]);
      return;
    }
    if (/^[1-9]$/.test(key)) {
      const view = state.stages[Number(key) - 1];
      if (view) {
        e.preventDefault();
        selectView(view);
      }
      return;
    }
    if (key === " " || e.code === "Space") {
      // A focused button/link already handles Space itself.
      if (t && t.closest && t.closest("button, a, [role=switch], [role=radio]")) return;
      e.preventDefault();
      runAction(state.running ? "stop" : "webcam");
    }
  });

  /* ------------------------------------------------------------------ *
   * Lifecycle
   * ------------------------------------------------------------------ */

  document.addEventListener("visibilitychange", () => {
    if (isVisible()) {
      scheduleStatus(0);
      scheduleAlerts(0);
      syncMainImage();
      startThumbLoop();
      tickCooldown();
    } else {
      clearTimeout(state.statusTimer);
      clearTimeout(state.alertsTimer);
      syncMainImage(); // drops the MJPEG connection while hidden
    }
  });

  let resizeQueued = false;
  window.addEventListener("resize", () => {
    if (resizeQueued) return;
    resizeQueued = true;
    requestAnimationFrame(() => {
      resizeQueued = false;
      drawChart();
    });
  });

  function init() {
    const urlPreset = readUrlPreset();
    if (urlPreset) storePreset(urlPreset);
    state.wantedPreset = urlPreset || readStoredPreset();

    // Render the wanted pipeline immediately so the page never starts empty.
    state.pendingMode = state.wantedPreset ? PRESETS[state.wantedPreset].mode : null;
    renderPresetUi();
    renderStrip(stagesFor(currentMode(), null), false);
    state.pendingMode = null;
    updateCaption();
    syncStripSelection();
    queueChart();

    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => {
        readChartColors();
        queueChart();
      });
    }

    pollStatus();
    pollAlerts();
  }

  init();
})();
