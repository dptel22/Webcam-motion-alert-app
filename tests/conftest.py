"""Shared pytest fixtures."""

import smtplib

import pytest


class _NoNetworkSMTP:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("Real SMTP is disabled during tests")


@pytest.fixture(autouse=True)
def _block_real_smtp(monkeypatch):
    """Safety net: no test may ever open a real SMTP connection (tests may still patch SMTP)."""
    monkeypatch.setattr(smtplib, "SMTP", _NoNetworkSMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _NoNetworkSMTP)
    yield
