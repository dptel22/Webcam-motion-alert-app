"""Email notification utilities and AlertManager for motion alert capture."""

from collections.abc import Callable
from email.message import EmailMessage
import itertools
import os
from pathlib import Path
import smtplib
import threading
import time

import cv2

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_SENDER = "devapp2005@gmail.com"
DEFAULT_RECEIVER = "dptel07@gmail.com"
DEFAULT_SUBJECT = "Intruder Detected"
DEFAULT_BODY = (
    "An intruder has been detected. View the attached image and take necessary action."
)


def detect_image_subtype(data: bytes, fallback: str = "octet-stream") -> str:
    """Detect image MIME subtype from header bytes without imghdr.

    Supports PNG and JPEG signatures, falling back cleanly to fallback string
    when unrecognized.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return fallback


def resolve_receiver(receiver: str | None = None) -> str:
    """Resolve the alert recipient the same way build_email_message does."""
    return receiver or os.getenv("EMAIL_RECEIVER") or DEFAULT_RECEIVER


def mask_email(address: str) -> str:
    """Mask an email address for display, e.g. "dptel07@gmail.com" -> "dp•••@gmail.com"."""
    address = (address or "").strip()
    local, sep, domain = address.partition("@")
    masked_local = local[:2] + "•••"
    return f"{masked_local}@{domain}" if sep else masked_local


def is_email_configured() -> bool:
    """Return True when GOOGLE_APP_PASSWORD is present and non-empty."""
    return bool((os.getenv("GOOGLE_APP_PASSWORD") or "").strip())


def _encode_preview_jpeg(frame, max_width: int = 480) -> bytes | None:
    """Downscale a frame so width <= max_width and encode as JPEG bytes."""
    try:
        h, w = frame.shape[:2]
        if w > max_width:
            new_h = max(1, int(round(h * max_width / float(w))))
            frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None
    except Exception:
        return None


def build_email_message(
    image_path: str | Path,
    sender: str | None = None,
    receiver: str | None = None,
    subject: str = DEFAULT_SUBJECT,
    body: str = DEFAULT_BODY,
) -> EmailMessage:
    """Build an EmailMessage with an attached image and appropriate MIME subtype."""
    resolved_sender = sender or os.getenv("EMAIL_SENDER") or DEFAULT_SENDER
    resolved_receiver = resolve_receiver(receiver)

    path = Path(image_path)
    content = path.read_bytes()
    subtype = detect_image_subtype(content, fallback="jpeg")
    maintype = "application" if subtype == "octet-stream" else "image"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = resolved_sender
    msg["To"] = resolved_receiver
    msg.set_content(body)
    msg.add_attachment(
        content,
        maintype=maintype,
        subtype=subtype,
        filename=path.name,
    )
    return msg


def send_email(
    image_path: str | Path,
    sender: str | None = None,
    receiver: str | None = None,
    password: str | None = None,
    subject: str = DEFAULT_SUBJECT,
    body: str = DEFAULT_BODY,
) -> None:
    """Send an alert email with snapshot attachment via Gmail SMTP.

    Raises:
        RuntimeError: If GOOGLE_APP_PASSWORD is not provided or set in environment.
    """
    app_password = password or os.getenv("GOOGLE_APP_PASSWORD")
    if not app_password:
        raise RuntimeError("GOOGLE_APP_PASSWORD environment variable is not set")

    msg = build_email_message(
        image_path=image_path,
        sender=sender,
        receiver=receiver,
        subject=subject,
        body=body,
    )
    sender_addr = msg["From"]

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender_addr, app_password)
        server.send_message(msg)


class AlertManager:
    """Coordinates motion snapshot capture, alert cooldown, and safe snapshot cleanup."""

    def __init__(
        self,
        snapshot_dir: str | Path = "images",
        cooldown_sec: float = 30.0,
        enabled: bool = True,
        sender_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.cooldown_sec = float(cooldown_sec)
        self.enabled = bool(enabled)
        self.sender_fn: Callable[[str], None] = sender_fn if sender_fn is not None else send_email
        self.last_trigger_time: float = -float("inf")
        self._last_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._history: list[dict] = []
        self._id_counter = itertools.count(1)

    HISTORY_LIMIT = 20

    def _add_record(self, frame, test: bool) -> dict:
        record = {
            "id": next(self._id_counter),
            "time": time.time(),
            "status": "sending",
            "error": None,
            "test": bool(test),
            "snapshot": _encode_preview_jpeg(frame),
        }
        with self._history_lock:
            self._history.insert(0, record)
            del self._history[self.HISTORY_LIMIT :]
        return record

    def _update_record(self, record_id: int, **fields) -> None:
        with self._history_lock:
            for rec in self._history:
                if rec["id"] == record_id:
                    rec.update(fields)
                    return

    def history(self) -> list[dict]:
        """Return a newest-first copy of alert records (without snapshot bytes)."""
        with self._history_lock:
            return [
                {k: v for k, v in rec.items() if k != "snapshot"} | {"has_snapshot": rec["snapshot"] is not None}
                for rec in self._history
            ]

    def get_snapshot(self, record_id: int) -> bytes | None:
        """Return in-memory JPEG snapshot bytes for an alert record, or None."""
        with self._history_lock:
            for rec in self._history:
                if rec["id"] == record_id:
                    return rec["snapshot"]
        return None

    def cooldown_remaining(self) -> float:
        """Seconds until the next non-test alert may fire (0.0 if ready)."""
        with self._lock:
            elapsed = time.monotonic() - self.last_trigger_time
        return max(0.0, self.cooldown_sec - elapsed)

    def cleanup_old_snapshots(self, exclude: Path | None = None) -> list[Path]:
        """Safely delete old snapshot files while strictly preserving the active snapshot."""
        cleaned: list[Path] = []
        if not self.snapshot_dir.exists():
            return cleaned

        exclude_resolved = exclude.resolve() if exclude is not None else None
        for file_path in self.snapshot_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                if exclude_resolved and file_path.resolve() == exclude_resolved:
                    continue
                try:
                    file_path.unlink()
                    cleaned.append(file_path)
                except OSError:
                    pass
        return cleaned

    def _process_alert(self, snapshot_path: Path, record_id: int | None = None) -> None:
        """Execute sender_fn, record the outcome, then clean up old snapshots.

        Exceptions from sender_fn are caught and recorded as a "failed" history entry
        instead of being lost in the background thread.
        """
        try:
            self.sender_fn(str(snapshot_path))
        except Exception as exc:
            if record_id is not None:
                self._update_record(record_id, status="failed", error=str(exc) or type(exc).__name__)
        else:
            if record_id is not None:
                self._update_record(record_id, status="sent", error=None)
        finally:
            self.cleanup_old_snapshots(exclude=snapshot_path)

    def _dispatch(self, frame, blocking: bool, test: bool) -> bool:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        prefix = "test_" if test else "alert_"
        snapshot_path = self.snapshot_dir / f"{prefix}{timestamp}.png"

        if not cv2.imwrite(str(snapshot_path), frame):
            return False

        record = self._add_record(frame, test=test)
        if blocking:
            self._process_alert(snapshot_path, record["id"])
        else:
            thread = threading.Thread(
                target=self._process_alert,
                args=(snapshot_path, record["id"]),
                daemon=True,
            )
            self._last_thread = thread
            thread.start()
        return True

    def send_test(self, frame, blocking: bool = False) -> bool:
        """Send a test alert, bypassing the enabled flag and cooldown (recorded with test=True)."""
        if frame is None:
            return False
        return self._dispatch(frame, blocking=blocking, test=True)

    def trigger(self, frame, blocking: bool = False) -> bool:
        """Trigger an alert for a motion frame.

        Saves frame to snapshot_dir, dispatches sender_fn, and cleans up older snapshots.
        Returns True if triggered, False if disabled, in cooldown, or invalid frame.
        """
        if not self.enabled:
            return False
        if frame is None:
            return False

        now = time.monotonic()
        with self._lock:
            if (now - self.last_trigger_time) < self.cooldown_sec:
                return False
            self.last_trigger_time = now

        return self._dispatch(frame, blocking=blocking, test=False)

    def wait(self, timeout: float | None = 5.0) -> None:
        """Wait for any active background alert thread to finish."""
        if self._last_thread and self._last_thread.is_alive():
            self._last_thread.join(timeout=timeout)
