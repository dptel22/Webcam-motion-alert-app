"""Email notification utilities and AlertManager for motion alert capture."""

from collections.abc import Callable
from email.message import EmailMessage
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


def build_email_message(
    image_path: str | Path,
    sender: str | None = None,
    receiver: str | None = None,
    subject: str = DEFAULT_SUBJECT,
    body: str = DEFAULT_BODY,
) -> EmailMessage:
    """Build an EmailMessage with an attached image and appropriate MIME subtype."""
    resolved_sender = sender or os.getenv("EMAIL_SENDER") or DEFAULT_SENDER
    resolved_receiver = receiver or os.getenv("EMAIL_RECEIVER") or DEFAULT_RECEIVER

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

    def _process_alert(self, snapshot_path: Path) -> None:
        """Execute sender_fn, then clean up old snapshots without racing the active snapshot."""
        try:
            self.sender_fn(str(snapshot_path))
        finally:
            self.cleanup_old_snapshots(exclude=snapshot_path)

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

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)
        snapshot_path = self.snapshot_dir / f"alert_{timestamp}.png"

        if not cv2.imwrite(str(snapshot_path), frame):
            return False

        if blocking:
            self._process_alert(snapshot_path)
        else:
            thread = threading.Thread(
                target=self._process_alert,
                args=(snapshot_path,),
                daemon=True,
            )
            self._last_thread = thread
            thread.start()

        return True

    def wait(self, timeout: float | None = 5.0) -> None:
        """Wait for any active background alert thread to finish."""
        if self._last_thread and self._last_thread.is_alive():
            self._last_thread.join(timeout=timeout)
