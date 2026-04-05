"""Scan camera indices 0-4 to find a working webcam."""

import cv2
import numpy as np

for idx in range(5):
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"Camera {idx}: not available")
        continue
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Warmup
    for _ in range(5):
        cap.read()
    ret, f = cap.read()
    if ret:
        m = np.mean(f)
        s = np.std(f)
        label = "USABLE" if s > 10 else "BLACK/DEAD"
        print(f"Camera {idx}: {w}x{h} mean={m:.1f} std={s:.1f} --> {label}")
        if s > 10:
            cv2.imwrite(f"cam_{idx}_sample.jpg", f)
            print(f"  Saved cam_{idx}_sample.jpg")
    else:
        print(f"Camera {idx}: opened but read failed")
    cap.release()

print("Scan done")
