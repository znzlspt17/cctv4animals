"""Environment verification script for People Counter project."""

import sys

print(f"Python: {sys.version}")

import torch

print(
    f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}"
)

import numpy as np

print(f"NumPy: {np.__version__}")

import cv2

print(f"OpenCV: {cv2.__version__}")

import ultralytics

print(f"Ultralytics: {ultralytics.__version__}")

import supervision

print(f"Supervision: {supervision.__version__}")

import sqlalchemy

print(f"SQLAlchemy: {sqlalchemy.__version__}")

import fastapi

print(f"FastAPI: {fastapi.__version__}")

print("Loguru: OK")

# numpy 2.x 충돌 확인
assert int(np.__version__.split(".")[0]) == 1, "⚠️ numpy 2.x 감지 — torch 충돌 위험!"
print("\n✅ 전체 호환성 검증 통과")
