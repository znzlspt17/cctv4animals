"""Quick webcam diagnostic — run standalone, no dependencies on project modules."""

import time

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam index 0")
    exit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_prop = cap.get(cv2.CAP_PROP_FPS)
backend = cap.getBackendName()
print(f"Camera: {w}x{h} @ {fps_prop}fps  backend={backend}")

good, bad = 0, 0
jpeg_sizes = []
start = time.time()

for i in range(30):
    ret, frame = cap.read()
    if not ret:
        bad += 1
        continue
    good += 1
    mean_v = np.mean(frame)
    std_v = np.std(frame)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    sz = len(buf) if ok else 0
    jpeg_sizes.append(sz)
    if i < 3 or i == 29:
        print(
            f"  frame {i:>2}: {frame.shape[1]}x{frame.shape[0]}  mean={mean_v:.1f}  std={std_v:.1f}  jpeg={sz}B"
        )

elapsed = time.time() - start
print(f"\n{good} good / {bad} bad  in {elapsed:.1f}s  ({good / elapsed:.1f} fps)")
if jpeg_sizes:
    print(
        f"JPEG min={min(jpeg_sizes)}  max={max(jpeg_sizes)}  avg={sum(jpeg_sizes) // len(jpeg_sizes)}"
    )
    if max(jpeg_sizes) < 2000:
        print("PROBLEM: all frames nearly black / empty")
    elif min(jpeg_sizes) < 2000 and max(jpeg_sizes) > 5000:
        print("WARNING: inconsistent frame quality")
    else:
        print("OK: frames look consistent")

# Save one frame for visual inspection
if good > 0:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_frame.jpg", frame)
        print("Saved test_frame.jpg for visual check")

cap.release()
