"""Alerting module for motion snapshot alerts and notifications."""

from src.alerts.emailing import (
    AlertManager,
    build_email_message,
    detect_image_subtype,
    send_email,
)

__all__ = [
    "AlertManager",
    "build_email_message",
    "detect_image_subtype",
    "send_email",
]
