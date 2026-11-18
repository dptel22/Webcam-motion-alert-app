"""Tests for AlertManager history records, test alerts, cooldown_remaining, and mask_email."""

import numpy as np
import pytest

from src.alerts.emailing import AlertManager, mask_email


def _frame() -> np.ndarray:
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[10:40, 10:50] = (40, 200, 90)
    return frame


def _manager(tmp_path, sender_fn, **kwargs) -> AlertManager:
    return AlertManager(snapshot_dir=tmp_path / "snaps", sender_fn=sender_fn, **kwargs)


def test_history_records_sent_alert_with_snapshot(tmp_path) -> None:
    sent: list[str] = []
    mgr = _manager(tmp_path, sent.append, cooldown_sec=0.0)
    assert mgr.trigger(_frame(), blocking=True) is True

    history = mgr.history()
    assert len(history) == 1
    rec = history[0]
    assert set(rec.keys()) == {"id", "time", "status", "error", "test", "has_snapshot"}
    assert rec["status"] == "sent"
    assert rec["error"] is None
    assert rec["test"] is False
    assert rec["has_snapshot"] is True
    assert len(sent) == 1

    jpeg = mgr.get_snapshot(rec["id"])
    assert jpeg is not None and jpeg.startswith(b"\xff\xd8")
    assert mgr.get_snapshot(9999) is None


def test_history_records_failed_alert_with_error_text(tmp_path) -> None:
    def failing_sender(_path: str) -> None:
        raise RuntimeError("SMTP auth rejected")

    mgr = _manager(tmp_path, failing_sender, cooldown_sec=0.0)
    assert mgr.trigger(_frame(), blocking=True) is True
    rec = mgr.history()[0]
    assert rec["status"] == "failed"
    assert rec["error"] == "SMTP auth rejected"


def test_failed_alert_without_message_uses_exception_type(tmp_path) -> None:
    def failing_sender(_path: str) -> None:
        raise ValueError()

    mgr = _manager(tmp_path, failing_sender)
    mgr.send_test(_frame(), blocking=True)
    assert mgr.history()[0]["error"] == "ValueError"


def test_history_is_capped_at_20_newest_first(tmp_path) -> None:
    mgr = _manager(tmp_path, lambda _p: None)
    for _ in range(25):
        assert mgr.send_test(_frame(), blocking=True)
    history = mgr.history()
    assert len(history) == AlertManager.HISTORY_LIMIT == 20
    ids = [rec["id"] for rec in history]
    assert ids == list(range(25, 5, -1))
    assert mgr.get_snapshot(1) is None  # evicted
    assert all(rec["test"] for rec in history)


def test_send_test_bypasses_disabled_flag_and_cooldown(tmp_path) -> None:
    sent: list[str] = []
    mgr = _manager(tmp_path, sent.append, enabled=False, cooldown_sec=60.0)
    assert mgr.trigger(_frame(), blocking=True) is False  # disabled
    assert mgr.send_test(_frame(), blocking=True) is True
    assert mgr.send_test(_frame(), blocking=True) is True
    assert len(sent) == 2
    assert mgr.cooldown_remaining() == 0.0  # test alerts do not start the cooldown
    assert mgr.send_test(None) is False


def test_cooldown_remaining(tmp_path) -> None:
    mgr = _manager(tmp_path, lambda _p: None, cooldown_sec=30.0)
    assert mgr.cooldown_remaining() == 0.0
    assert mgr.trigger(_frame(), blocking=True) is True
    remaining = mgr.cooldown_remaining()
    assert 29.0 < remaining <= 30.0
    assert mgr.trigger(_frame(), blocking=True) is False  # still cooling down

    no_cooldown = _manager(tmp_path, lambda _p: None, cooldown_sec=0.0)
    no_cooldown.trigger(_frame(), blocking=True)
    assert no_cooldown.cooldown_remaining() == 0.0


@pytest.mark.parametrize(
    "address, expected",
    [
        ("dptel07@gmail.com", "dp•••@gmail.com"),
        ("a@b.com", "a•••@b.com"),
        ("  someone@example.org  ", "so•••@example.org"),
        ("noatsign", "no•••"),
        ("", "•••"),
        (None, "•••"),
    ],
)
def test_mask_email(address, expected) -> None:
    assert mask_email(address) == expected
