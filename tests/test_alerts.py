"""Unit tests for AlertManager and emailing module."""

import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import MagicMock, patch

# Ensure workspace root is on sys.path for direct pytest.exe execution
WORKSPACE_ROOT = str(Path(__file__).resolve().parent.parent)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import cv2
import numpy as np
import pytest

from src.alerts.emailing import (
    AlertManager,
    build_email_message,
    detect_image_subtype,
    send_email,
)


def test_detect_image_subtype_png():
    """Detects PNG image subtype from standard header bytes."""
    header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    assert detect_image_subtype(header) == "png"


def test_detect_image_subtype_jpeg():
    """Detects JPEG image subtype from standard header bytes."""
    header = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00"
    assert detect_image_subtype(header) == "jpeg"


def test_detect_image_subtype_fallback():
    """Falls back to octet-stream or jpeg cleanly on unrecognized bytes without imghdr."""
    unknown = b"arbitrary-non-image-binary-stream-12345"
    assert detect_image_subtype(unknown) in ("octet-stream", "jpeg")
    assert detect_image_subtype(unknown, fallback="octet-stream") == "octet-stream"
    assert detect_image_subtype(unknown, fallback="jpeg") == "jpeg"


def test_build_email_message_png(tmp_path):
    """build_email_message attaches PNG file with correct MIME type and headers."""
    img_path = tmp_path / "intruder.png"
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), frame)

    msg = build_email_message(
        image_path=img_path,
        sender="sender@example.com",
        receiver="receiver@example.com",
        subject="Intruder Alert",
        body="Intruder detected in room.",
    )

    assert msg["From"] == "sender@example.com"
    assert msg["To"] == "receiver@example.com"
    assert msg["Subject"] == "Intruder Alert"
    assert "Intruder detected in room." in msg.get_body().get_content()

    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    att = attachments[0]
    assert att.get_content_type() == "image/png"
    assert att.get_filename() == "intruder.png"


def test_build_email_message_jpeg(tmp_path):
    """build_email_message attaches JPEG file with correct MIME type."""
    img_path = tmp_path / "snapshot.jpg"
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), frame)

    msg = build_email_message(
        image_path=img_path,
        sender="sender@example.com",
        receiver="receiver@example.com",
    )

    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    att = attachments[0]
    assert att.get_content_type() == "image/jpeg"
    assert att.get_filename() == "snapshot.jpg"


def test_alert_manager_disabled(tmp_path):
    """AlertManager does nothing when enabled=False."""
    calls = []
    mgr = AlertManager(
        snapshot_dir=tmp_path,
        cooldown_sec=1.0,
        enabled=False,
        sender_fn=lambda p: calls.append(p),
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    res = mgr.trigger(frame, blocking=True)

    assert res is False
    assert len(calls) == 0
    assert list(tmp_path.glob("*.png")) == []


def test_alert_manager_autocreates_snapshot_dir_and_invokes_sender(tmp_path):
    """AlertManager auto-creates snapshot_dir and saves snapshot frame before invoking sender_fn."""
    snapshot_dir = tmp_path / "nested" / "alerts"
    assert not snapshot_dir.exists()

    calls = []

    def fake_sender(path):
        calls.append(path)
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0

    mgr = AlertManager(
        snapshot_dir=snapshot_dir,
        cooldown_sec=1.0,
        enabled=True,
        sender_fn=fake_sender,
    )
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    res = mgr.trigger(frame, blocking=True)

    assert res is True
    assert snapshot_dir.exists()
    assert len(calls) == 1
    assert Path(calls[0]).exists()


def test_alert_manager_enforces_cooldown(tmp_path, monkeypatch):
    """AlertManager ignores rapid triggers within cooldown_sec."""
    calls = []
    mgr = AlertManager(
        snapshot_dir=tmp_path,
        cooldown_sec=30.0,
        enabled=True,
        sender_fn=lambda p: calls.append(p),
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    current_time = 1000.0
    monkeypatch.setattr(time, "monotonic", lambda: current_time)

    # First trigger executes
    assert mgr.trigger(frame, blocking=True) is True
    assert len(calls) == 1

    # Second trigger within cooldown (15s later) is ignored
    current_time = 1015.0
    assert mgr.trigger(frame, blocking=True) is False
    assert len(calls) == 1

    # Third trigger after cooldown (31s later) executes
    current_time = 1031.0
    assert mgr.trigger(frame, blocking=True) is True
    assert len(calls) == 2


def test_alert_manager_safe_cleanup_after_sender(tmp_path):
    """Cleans up old snapshots safely *after* sender_fn finishes, never deleting active snapshot prematurely."""
    old_snapshot = tmp_path / "old_frame.png"
    old_snapshot.write_bytes(b"old-frame-content")

    sender_state = {
        "called": False,
        "active_file_exists_during_send": False,
        "active_file_is_readable": False,
    }

    def sender_fn(path):
        sender_state["called"] = True
        active_path = Path(path)
        sender_state["active_file_exists_during_send"] = active_path.exists()
        # Verify active snapshot can be read without race condition
        content = active_path.read_bytes()
        sender_state["active_file_is_readable"] = len(content) > 0

    mgr = AlertManager(
        snapshot_dir=tmp_path,
        cooldown_sec=0.0,
        enabled=True,
        sender_fn=sender_fn,
    )

    frame = np.zeros((25, 25, 3), dtype=np.uint8)
    res = mgr.trigger(frame, blocking=True)

    assert res is True
    assert sender_state["called"] is True
    assert sender_state["active_file_exists_during_send"] is True
    assert sender_state["active_file_is_readable"] is True
    # The old snapshot was safely cleaned up after sender_fn completed
    assert not old_snapshot.exists()


def test_alert_manager_threaded_execution(tmp_path):
    """AlertManager can trigger alerts in a background thread."""
    called_event = threading.Event()
    received_paths = []

    def async_sender(path):
        received_paths.append(path)
        called_event.set()

    mgr = AlertManager(
        snapshot_dir=tmp_path,
        cooldown_sec=1.0,
        enabled=True,
        sender_fn=async_sender,
    )
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    res = mgr.trigger(frame, blocking=False)

    assert res is True
    assert called_event.wait(timeout=3.0) is True
    mgr.wait(timeout=3.0)
    assert len(received_paths) == 1
    assert Path(received_paths[0]).exists()


def test_send_email_missing_password_raises(tmp_path, monkeypatch):
    """send_email raises RuntimeError if GOOGLE_APP_PASSWORD is not set."""
    img_path = tmp_path / "snapshot.png"
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), frame)

    monkeypatch.delenv("GOOGLE_APP_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_APP_PASSWORD"):
        send_email(img_path)


def test_send_email_success(tmp_path, monkeypatch):
    """send_email logs in and sends message via smtplib.SMTP when password is provided."""
    img_path = tmp_path / "snapshot.png"
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), frame)

    monkeypatch.setenv("GOOGLE_APP_PASSWORD", "mock_app_pass")
    monkeypatch.setenv("EMAIL_SENDER", "custom_sender@gmail.com")
    monkeypatch.setenv("EMAIL_RECEIVER", "custom_receiver@gmail.com")

    with patch("smtplib.SMTP") as mock_smtp:
        mock_instance = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_instance

        send_email(img_path)

        mock_smtp.assert_called_once_with("smtp.gmail.com", 587)
        mock_instance.starttls.assert_called_once()
        mock_instance.login.assert_called_once_with("custom_sender@gmail.com", "mock_app_pass")
        mock_instance.send_message.assert_called_once()


def test_root_emailing_compatibility_wrapper():
    """Root emailing.py exposes send_email and helpers for backwards compatibility."""
    import emailing

    assert hasattr(emailing, "send_email")
    assert hasattr(emailing, "AlertManager")
    assert hasattr(emailing, "detect_image_subtype")
    assert hasattr(emailing, "build_email_message")

