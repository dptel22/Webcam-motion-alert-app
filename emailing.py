"""Backward-compatibility wrapper re-exporting emailing functionality from src.alerts.emailing."""

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

if __name__ == "__main__":
    send_email(image_path="images/30.png")
