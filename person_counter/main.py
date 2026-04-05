"""Main entry point for the People Counter application."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import supervision as sv
import uvicorn
from loguru import logger

from alert import KakaoAlert
from config import settings
from counter import LineCrossCounter
from dashboard.app import app, counter_manager, set_performance_monitor, stream_manager
from database import SessionLocal, init_db, restore_count, save_event
from detector import PersonDetector
from monitor import PerformanceMonitor
from snapshot import SnapshotManager
from tracker import PersonTracker


def _parse_video_source(source: str):
    """Convert VIDEO_SOURCE to int for webcam index, or keep as string for RTSP/file."""
    try:
        return int(source)
    except ValueError:
        return source


async def _broadcast_stream_bytes(data: bytes) -> None:
    """Send binary JPEG data to all stream WebSocket clients."""
    stale = []
    for ws in stream_manager.active_connections:
        try:
            await ws.send_bytes(data)
        except Exception:
            stale.append(ws)
    for ws in stale:
        stream_manager.disconnect(ws)


def _draw_overlay(
    frame: np.ndarray,
    counter: LineCrossCounter,
    tracked: sv.Detections,
    monitor: PerformanceMonitor,
) -> None:
    """Draw counting line, bounding boxes, tracker IDs, and stats on frame (in-place)."""
    h, w = frame.shape[:2]

    # Counting line — red, 2px
    cv2.line(
        frame,
        (settings.LINE_START_X, settings.LINE_START_Y),
        (settings.LINE_END_X, settings.LINE_END_Y),
        (0, 0, 255),
        2,
    )

    # Bounding boxes + tracker IDs — green, 2px
    if tracked is not None and len(tracked) > 0:
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if tracked.tracker_id is not None:
                tid = int(tracked.tracker_id[i])
                cv2.putText(
                    frame,
                    f"ID:{tid}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    # Current count — top-left, large
    cv2.putText(
        frame,
        f"Count: {counter.current_count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    cv2.putText(
        frame,
        f"IN: {counter.in_count}  OUT: {counter.out_count}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # FPS — top-right
    stats = monitor.get_stats()
    fps_text = f"FPS: {stats['fps']}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(
        frame,
        fps_text,
        (w - text_size[0] - 10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )


def main() -> None:
    """People Counter main loop."""
    logger.info("Starting People Counter — camera_id={}", settings.CAMERA_ID)

    # ---- 1–2. DB initialization ----
    try:
        init_db()
    except Exception as exc:
        logger.warning("DB init failed (continuing without DB): {}", exc)

    # ---- 3. State recovery ----
    restored_count = 0
    try:
        restored_count = restore_count(settings.CAMERA_ID)
    except Exception as exc:
        logger.warning("Count restore failed: {}", exc)

    # ---- 4. Module initialization ----
    detector = PersonDetector()
    tracker = PersonTracker()
    counter = LineCrossCounter()
    counter._current_count = restored_count
    snapshot_mgr = SnapshotManager()
    monitor = PerformanceMonitor()
    alert = KakaoAlert()
    logger.info("All modules initialized — restored count={}", restored_count)

    # ---- 5. FastAPI dashboard in daemon thread ----
    loop = asyncio.new_event_loop()
    dash_port = settings.DASHBOARD_PORT

    def _run_server() -> None:
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(app, host="0.0.0.0", port=dash_port, log_level="info")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()
    logger.info("Dashboard server started on http://0.0.0.0:{}", dash_port)

    # ---- 6. Inject monitor into dashboard ----
    set_performance_monitor(monitor)

    # ---- 7. Video capture ----
    source = _parse_video_source(settings.VIDEO_SOURCE)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: {}", settings.VIDEO_SOURCE)
        return

    logger.info("Video source opened: {}", settings.VIDEO_SOURCE)

    try:
        while True:
            # a. Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed — retrying …")
                time.sleep(0.1)
                continue

            # b. Start frame timing
            monitor.start_frame()

            # c. Detect or skip (Kalman prediction via empty detections)
            if tracker.should_detect():
                t0 = time.perf_counter()
                detections = detector.detect(frame)
                inference_ms = (time.perf_counter() - t0) * 1000
                monitor.record_inference(inference_ms)
            else:
                detections = sv.Detections.empty()

            # d. Track
            t0 = time.perf_counter()
            tracked = tracker.update(detections)
            tracking_ms = (time.perf_counter() - t0) * 1000
            monitor.record_tracking(tracking_ms)

            # e. Count line crossings
            events = counter.update(tracked)

            # f. Process each event
            for event in events:
                # Snapshot first (need path for DB)
                snap_path = ""
                try:
                    snap_path = snapshot_mgr.save(frame, event, settings.CAMERA_ID)
                except Exception as exc:
                    logger.error("Snapshot save failed: {}", exc)

                # DB save
                try:
                    db = SessionLocal()
                    event_data = {
                        "person_id": event.person_id,
                        "camera_id": settings.CAMERA_ID,
                        "direction": event.direction,
                        "count_change": event.count_change,
                        "current_count": counter.current_count,
                        "confidence": event.confidence,
                        "bbox_x": int(event.bbox[0]),
                        "bbox_y": int(event.bbox[1]),
                        "bbox_w": int(event.bbox[2] - event.bbox[0]),
                        "bbox_h": int(event.bbox[3] - event.bbox[1]),
                        "snapshot_path": snap_path,
                    }
                    save_event(db, event_data)
                except Exception as exc:
                    logger.error("DB save failed: {}", exc)
                finally:
                    db.close()

                # Alert check
                try:
                    alert.check_and_send(counter.current_count)
                except Exception as exc:
                    logger.error("Alert check failed: {}", exc)

                # WebSocket counter broadcast
                ws_msg = {
                    "type": "count_event",
                    "person_id": event.person_id,
                    "direction": event.direction,
                    "count_change": event.count_change,
                    "current_count": counter.current_count,
                    "in_count": counter.in_count,
                    "out_count": counter.out_count,
                    "confidence": event.confidence,
                    "timestamp": datetime.now().isoformat(),
                }
                try:
                    asyncio.run_coroutine_threadsafe(
                        counter_manager.broadcast(ws_msg), loop
                    )
                except Exception as exc:
                    logger.error("WS counter broadcast failed: {}", exc)

            # g. Draw overlay
            display_frame = frame.copy()
            _draw_overlay(display_frame, counter, tracked, monitor)

            # h. Stream frame via WebSocket (JPEG bytes)
            ok, jpeg = cv2.imencode(
                ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if ok:
                try:
                    asyncio.run_coroutine_threadsafe(
                        _broadcast_stream_bytes(jpeg.tobytes()), loop
                    )
                except Exception as exc:
                    logger.error("WS stream broadcast failed: {}", exc)

            # i. Local display (skip if headless OpenCV)
            try:
                cv2.imshow("People Counter", display_frame)
            except cv2.error:
                pass  # headless build — no GUI

            # j. End frame timing
            monitor.end_frame()

            # k. Advance frame counter
            tracker.increment_frame()

            # l. Quit on 'q' (skip if headless)
            try:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit key pressed — shutting down")
                    break
            except cv2.error:
                pass

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        logger.info("People Counter stopped.")


if __name__ == "__main__":
    main()
