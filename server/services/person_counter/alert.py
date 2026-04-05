"""KakaoTalk alert notification module."""

from __future__ import annotations

import json
import time

import requests
from loguru import logger

from config import settings


class KakaoAlert:
    """Sends a KakaoTalk '나에게 보내기' alert when occupancy exceeds threshold."""

    MEMO_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    TOKEN_URL = "https://kauth.kakao.com/oauth/token"

    def __init__(self) -> None:
        self.access_token: str = settings.KAKAO_ACCESS_TOKEN
        self.refresh_token: str = settings.KAKAO_REFRESH_TOKEN
        self.threshold: int = settings.ALERT_THRESHOLD
        self.cooldown: int = settings.ALERT_COOLDOWN_SEC
        self.last_alert_time: float = 0.0
        logger.info(
            "KakaoAlert initialized: threshold={}, cooldown={}s, token_set={}",
            self.threshold,
            self.cooldown,
            bool(self.access_token),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_send(self, current_count: int) -> bool:
        """Check threshold & cooldown, then send alert if needed.

        Returns True if an alert was actually sent.
        """
        if current_count < self.threshold:
            return False

        now = time.time()
        if now - self.last_alert_time < self.cooldown:
            logger.debug(
                "Alert cooldown active ({:.0f}s remaining)",
                self.cooldown - (now - self.last_alert_time),
            )
            return False

        if not self.access_token:
            logger.warning(
                "KAKAO_ACCESS_TOKEN is empty — skipping alert (count={})",
                current_count,
            )
            return False

        text = (
            f"[People Counter 알림]\n"
            f"현재 인원: {current_count}명\n"
            f"임계값({self.threshold}명) 초과!"
        )
        sent = self._send_message(text)
        if sent:
            self.last_alert_time = now
        return sent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_message(self, text: str) -> bool:
        """Call Kakao '나에게 보내기' REST API."""
        template_object = json.dumps(
            {
                "object_type": "text",
                "text": text,
                "link": {"web_url": "", "mobile_web_url": ""},
            }
        )
        headers = {"Authorization": f"Bearer {self.access_token}"}
        data = {"template_object": template_object}

        try:
            resp = requests.post(self.MEMO_URL, headers=headers, data=data, timeout=10)
            if resp.status_code == 200:
                logger.info("Kakao alert sent successfully")
                return True
            if resp.status_code == 401:
                logger.warning("Kakao token expired — attempting refresh")
                if self._refresh_token():
                    return self._send_message(text)
                return False
            logger.error("Kakao API error: {} {}", resp.status_code, resp.text)
            return False
        except requests.RequestException as exc:
            logger.error("Kakao API request failed: {}", exc)
            return False

    def _refresh_token(self) -> bool:
        """Attempt to refresh the Kakao access token using the refresh token."""
        if not self.refresh_token:
            logger.warning("No refresh token available — cannot refresh")
            return False

        data = {
            "grant_type": "refresh_token",
            "client_id": "",  # Kakao REST API key — set via env if needed
            "refresh_token": self.refresh_token,
        }

        try:
            resp = requests.post(self.TOKEN_URL, data=data, timeout=10)
            if resp.status_code != 200:
                logger.error("Token refresh failed: {} {}", resp.status_code, resp.text)
                return False

            body = resp.json()
            self.access_token = body.get("access_token", self.access_token)
            if "refresh_token" in body:
                self.refresh_token = body["refresh_token"]
            logger.info("Kakao token refreshed successfully")
            return True
        except requests.RequestException as exc:
            logger.error("Token refresh request failed: {}", exc)
            return False
