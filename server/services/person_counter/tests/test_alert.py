"""Unit tests for KakaoAlert (alert.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def alert_no_token():
    """KakaoAlert with empty tokens (default)."""
    with patch("alert.settings") as mock_s:
        mock_s.KAKAO_ACCESS_TOKEN = ""
        mock_s.KAKAO_REFRESH_TOKEN = ""
        mock_s.ALERT_THRESHOLD = 50
        mock_s.ALERT_COOLDOWN_SEC = 300
        from alert import KakaoAlert

        return KakaoAlert()


@pytest.fixture
def alert_with_token():
    """KakaoAlert with a fake token set."""
    with patch("alert.settings") as mock_s:
        mock_s.KAKAO_ACCESS_TOKEN = "fake_access_token"
        mock_s.KAKAO_REFRESH_TOKEN = "fake_refresh_token"
        mock_s.ALERT_THRESHOLD = 50
        mock_s.ALERT_COOLDOWN_SEC = 300
        from alert import KakaoAlert

        return KakaoAlert()


class TestBelowThreshold:
    def test_below_threshold(self, alert_with_token):
        """count < threshold → check_and_send returns False."""
        assert alert_with_token.check_and_send(10) is False
        assert alert_with_token.check_and_send(49) is False


class TestCooldown:
    def test_cooldown(self, alert_with_token):
        """Second call within cooldown period → returns False."""
        with patch.object(alert_with_token, "_send_message", return_value=True):
            # First call: above threshold, should send
            result1 = alert_with_token.check_and_send(60)
            assert result1 is True

            # Second call immediately: cooldown should block
            result2 = alert_with_token.check_and_send(60)
            assert result2 is False


class TestNoToken:
    def test_no_token(self, alert_no_token):
        """Empty access token → returns False with warning only."""
        result = alert_no_token.check_and_send(100)
        assert result is False


class TestSendAboveThreshold:
    def test_send_above_threshold(self, alert_with_token):
        """count >= threshold + token present → _send_message called, returns True."""
        with patch.object(
            alert_with_token, "_send_message", return_value=True
        ) as mock_send:
            result = alert_with_token.check_and_send(60)
            assert result is True
            mock_send.assert_called_once()

    def test_send_failure(self, alert_with_token):
        """_send_message returns False → check_and_send returns False."""
        with patch.object(alert_with_token, "_send_message", return_value=False):
            result = alert_with_token.check_and_send(60)
            assert result is False
