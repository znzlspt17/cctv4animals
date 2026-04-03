"""Streamlit UI 런처 — .env의 STREAMLIT_PORT를 적용합니다."""

import subprocess
import sys

from server.config import settings

subprocess.run(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/app.py",
        "--server.port",
        str(settings.STREAMLIT_PORT),
        "--server.address",
        "0.0.0.0",
    ],
    check=True,
)
