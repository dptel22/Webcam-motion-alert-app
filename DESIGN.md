# Design Specification — Optical Calibration Bench

## 1. Aesthetic Direction
**Optical Calibration Bench**: a high-contrast, clinical laboratory instrument interface built for inspecting real-time computer vision, optical flow, and multi-object tracking telemetry. Every visual decision favors legibility, precision, and zero decorative noise.

- **No glassmorphism, glow shadows, or ambient gradients:** all panels sit on crisp `1px` hairline borders with flat surfaces and `3px` corner radii.
- **No decorative chrome:** no fake window controls, no ordinal `01/02/03` numbering, no uppercase micro-labels, no animations on scroll or load.

## 2. Color Palette

| Token | Hex | Role |
|---|---|---|
| `--bg-ground` | `#EEF1F5` | Cool lab-bench slate grey canvas |
| `--bg-surface` | `#FFFFFF` | Panel, table, and control card surface |
| `--bg-subtle` | `#F8FAFC` | Table header, secondary readout wells, inactive tab hover |
| `--bg-viewport` | `#12171E` | Matte optical feed enclosure (16:9 stage) |
| `--border-hairline` | `#C8D1DC` | Primary `1px` panel and control borders |
| `--border-grid` | `#E2E8F0` | Inner table dividers and strip-chart gridlines |
| `--ink-primary` | `#12171E` | Headings, active labels, primary telemetry values |
| `--ink-secondary` | `#475467` | Supporting labels, axis ticks, inactive tabs (WCAG AA on `#EEF1F5` and `#FFFFFF`) |
| `--accent-primary` | `#0F4C81` | Calibrated cobalt: active modes, primary actions, FPS trace, focus rings |
| `--accent-primary-hover` | `#0B3A63` | Hover state for primary cobalt buttons |
| `--accent-warn` | `#B45309` | Amber: lost/tentative track states, latency trace, warnings |
| `--accent-danger` | `#B91C1C` | Crimson: stop action, alert flags, fault messages |

## 3. Typography
- **Font family:** `IBM Plex Sans` (weights `400`, `500`, `600`) loaded from Google Fonts, with sans-serif system fallbacks.
- **Numeric alignment:** `font-variant-numeric: tabular-nums` applied to all telemetry readouts, chart axes, stride selectors, and track registry table cells so live numbers never jitter horizontally.
- **Minimum label size:** All visible UI labels, table headers, and chart captions are `>= 12px` (`0.75rem`).

## 4. Layout Architecture
1. **Top Instrument Bar (`header.topbar`)**
   - Left: Application title, live state indicator (`RUNNING` / `STANDBY`), active tracking mode, and active input source.
   - Right: Direct export links (`CSV`, `JSON`, `Plot PNG`) wired to `/api/export?format=...`.
2. **Two-Column Workbench (`main.workbench`)**
   - **Left Primary Stage (`section.stage`, flexible primary column):**
     - **View Selector Tabs:** `Tracking Output` (`annotated`), `2x2 CVIP+DL Quad` (`quad`), `CLAHE Enhanced` (`clahe`), `MOG2 Mask` (`mog2`), `Optical Flow` (`flow`).
     - **Optical Viewport (`div.viewport`):** Fixed `16:9` aspect ratio on matte `#12171E`. Displays the live MJPEG stream (`/stream/mjpeg?view=...`) when running, or a centered standby prompt with `Start Webcam` and `Run Sample Demo` actions when stopped.
     - **Track Registry Table:** Live tabular readout (`ID`, `Class`, `Confidence`, `Velocity (px/f)`, `Age`, `State`) backed by `/api/status`.
   - **Right Instrument Control Rail (`aside.control-rail`, `360px` desktop / full-width below `1040px`):**
     - **Source Controls:** `Start Webcam`, `Run Sample Demo`, `Upload Video` (`.mp4/.avi/.mov/.mkv`), `Stop`, `Reset Tracker`.
     - **Ablation Mode Selector:** Segmented radio group (`Hybrid CVIP+DL`, `DL SORT`, `CVIP MOG2`) with a one-line pipeline summary of active stages.
     - **Pipeline Parameters:** Toggles for `CLAHE Normalization`, `Re-ID Appearance`, `Email Alerts`, and segmented `Detection Stride` (`1`, `2`, `3` frames).
     - **Telemetry & Strip Chart:** Four tabular KPI cells (`Avg FPS`, `Latency ms`, `Active Tracks`, `Total IDs`) paired with a vanilla `<canvas>` dual-axis strip chart driven by `/api/metrics?limit=60`.

## 5. Component & Accessibility Rules
- **Borders & Radii:** Uniform `border-radius: 3px` and `1px solid #C8D1DC` borders. Zero `box-shadow` elevation layers or `backdrop-filter`.
- **Keyboard Focus:** Every interactive `<button>`, `<a>`, `<input>`, and tab uses `outline: 2px solid #0F4C81; outline-offset: 2px` on `:focus-visible`.
- **Reduced Motion:** `@media (prefers-reduced-motion: reduce)` disables all CSS transitions. No entry or scroll animations.
- **Semantic Structure:** Uses `<header>`, `<main>`, `<section>`, `<aside>`, `<table>`, `role="tablist"`, `role="tab"`, and `aria-live="polite"` status regions.
