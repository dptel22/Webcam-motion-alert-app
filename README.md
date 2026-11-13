# Hybrid CVIP + Deep Learning Visual Tracking System

A real-time multi-object visual tracking, ablation, and motion-alert workbench built in Python, combining **classical Computer Vision & Image Processing (CVIP)** with **Deep Learning (DL)** detection and appearance re-identification, plus a Starlette + vanilla web dashboard and native OpenCV debug suite.

## Architecture & Algorithms

### Classical CVIP (`src/cvip/`)
- **Illumination Normalization (`CVIPPreprocessor.apply_clahe`):** Converts BGR frames to LAB color space and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the luminance channel, followed by Gaussian denoising.
- **Adaptive Background Subtraction (`CVIPPreprocessor.extract_foreground_mask`):** Uses `MOG2` with shadow suppression (`threshold > 200`) and morphological opening/closing + contour filtering.
- **Sparse Pyramidal Optical Flow (`OpticalFlowTracker`):** Tracks Shi-Tomasi corners via Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`) inside active bounding boxes to bridge non-keyframe intervals on CPU and visualize motion vectors.

### Deep Learning (`src/dl/`)
- **Object Detection (`YOLODetector`):** CPU-friendly `yolov8n.pt` detector with configurable frame-stride (`detect_interval`).
- **Appearance Re-ID (`ReIDExtractor`):** Batched `MobileNetV3-Small` backbone extracting L2-normalized 576-d appearance embeddings per detected crop.

### Tracking Engine (`src/tracking/`)
- **State Estimation (`KalmanBoxTracker`):** 8-state constant-velocity Kalman filter (`[cx, cy, aspect_ratio, height, vx, vy, va, vh]`).
- **Data Association (`MultiObjectTracker`):** Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) fusing IoU spatial cost and Re-ID cosine distance, with `TENTATIVE` / `CONFIRMED` / `LOST` lifecycle states, EMA gallery updates, and trajectory trails.

### Live Ablation Modes (`TrackingMode`)
| Mode | Pipeline |
|---|---|
| `hybrid` *(default)* | CLAHE + Lucas-Kanade Optical Flow + YOLOv8n + Kalman/Hungarian + MobileNetV3 Re-ID |
| `dl_sort` | YOLOv8n + Kalman/Hungarian IoU (SORT baseline) |
| `cvip` | CLAHE + MOG2 Background Subtraction + Kalman/Hungarian IoU |

### Alerts, Evaluation & Web UI
- **Alerts (`src/alerts/emailing.py`):** Python 3.14-safe MIME detection (no `imghdr`), cooldown enforcement, and race-free snapshot cleanup after SMTP dispatch.
- **Metrics (`src/eval/metrics.py`):** Records per-frame FPS, latency (ms), and track counts; exports CSV, JSON, and Matplotlib PNG summary charts.
- **Web Workbench (`src/web_server.py`, `src/web/`):** Live MJPEG streaming across 5 views (`annotated`, `quad`, `clahe`, `mog2`, `flow`), live track registry table, ablation toggles, video upload (`.mp4/.avi/.mov/.mkv`), synthetic demo clip mode, and live FPS/latency canvas strip chart.

## Setup

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Optional environment variables (`.env`) for email alerts:
- `GOOGLE_APP_PASSWORD`
- `EMAIL_SENDER` (optional override)
- `EMAIL_RECEIVER` (optional override)

## Usage

### Launch Web Dashboard
```powershell
.\.venv\Scripts\python.exe main.py --web --port 8000
```
Open `http://127.0.0.1:8000` in your browser. Click **Run Sample Demo** to test immediately without a webcam, **Start Webcam** for live feed, or **Upload Video** for an `.mp4/.avi/.mov/.mkv` file.

### Native OpenCV Window CLI
```powershell
# Live webcam in Hybrid CVIP+DL mode with 2x2 debug quad view
.\.venv\Scripts\python.exe main.py --source 0 --mode hybrid --debug-view

# Offline synthetic demo clip
.\.venv\Scripts\python.exe main.py --source sample --mode hybrid --debug-view
```

**OpenCV Hotkeys:** `m` cycle mode (`hybrid` / `dl_sort` / `cvip`) · `v` toggle 2x2 CVIP quad view · `c` toggle CLAHE · `r` reset tracker & background · `s` export metrics + plot · `q` quit

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
