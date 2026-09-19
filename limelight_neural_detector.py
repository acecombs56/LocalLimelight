#!/usr/bin/env python3
"""
LocalLimelight - Neural detector reader for a Limelight attached to your laptop.

Uses pipeline 8 (the neural network "Detector" pipeline) to find a trained
object and prints its X (tx), Y (ty), and an estimated distance to the
detected object.

Requirements:
    pip install requests

Usage:
    python limelight_neural_detector.py --host limelight.local
    python limelight_neural_detector.py --host 10.TE.AM.11 --loop

Notes:
    - This talks directly to the Limelight's HTTP JSON endpoint
      (http://<host>:5807/results), so no NetworkTables/robot code is needed -
      it just needs the Limelight reachable on your network (e.g. plugged
      into your laptop via ethernet, with the laptop on the same subnet).
    - Pipeline 8 must already be configured on the Limelight as a neural
      network "Detector" pipeline with your trained object model/class.
    - Distance estimation uses the standard camera-height/angle trig formula:

          distance = (objectHeight - cameraHeight) / tan(cameraAngle + ty)

      Adjust CAMERA_HEIGHT_M, CAMERA_ANGLE_DEG, and OBJECT_HEIGHT_M below (or
      via CLI flags) to match your physical mounting setup and the real
      height of the trained object for accurate distances. If you don't
      calibrate these, distance will still be computed but may not be
      accurate - it's a placeholder based on defaults.
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests


DEFAULT_HOST = "limelight.local"
RESULTS_PORT = 5807  # Limelight's HTTP JSON results endpoint
PIPELINE_PORT = 5807  # switching pipelines is done via the same host, /update-pipeline
NEURAL_DETECTOR_PIPELINE = 8  # pipeline index configured as the neural network Detector

# --- Physical setup defaults (calibrate these for accurate distance) -----
CAMERA_HEIGHT_M = 1.27      # height of the limelight lens off the ground/reference plane
CAMERA_ANGLE_DEG = -0.8     # mounting angle of the camera above horizontal
OBJECT_HEIGHT_M = 1.54      # height of the center of the trained object off the same reference plane
# --------------------------------------------------------------------------


@dataclass
class DetectionReading:
    pipeline: int
    class_id: int
    class_name: str
    confidence: float
    tx: float
    ty: float
    ta: float
    distance_m: Optional[float]


def get_results(host: str, timeout: float = 1.0) -> dict:
    """Fetch the raw JSON results dump from the Limelight."""
    url = f"http://{host}:{RESULTS_PORT}/results"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def set_pipeline(host: str, pipeline_index: int, timeout: float = 1.0) -> None:
    """Switch the Limelight's active pipeline."""
    url = f"http://{host}:{PIPELINE_PORT}/update-pipeline?index={pipeline_index}"
    try:
        requests.post(url, timeout=timeout)
    except requests.RequestException as exc:
        print(f"[warn] failed to switch to pipeline {pipeline_index}: {exc}", file=sys.stderr)


def estimate_distance(ty_deg: float,
                       camera_height_m: float,
                       camera_angle_deg: float,
                       object_height_m: float) -> Optional[float]:
    """Estimate distance to detected object using camera height/angle trig formula."""
    angle_total_rad = math.radians(camera_angle_deg + ty_deg)
    denom = math.tan(angle_total_rad)
    if abs(denom) < 1e-6:
        return None
    distance = (object_height_m - camera_height_m) / denom
    return distance


def read_neural_detections(host: str,
                            pipeline_index: int,
                            camera_height_m: float,
                            camera_angle_deg: float,
                            object_height_m: float,
                            settle_time: float = 0.25) -> list:
    """Switch to the neural detector pipeline, wait for it to settle, then read detections."""
    set_pipeline(host, pipeline_index)
    time.sleep(settle_time)

    data = get_results(host)
    results = data.get("Results", data)  # some LL versions nest under "Results"

    # Neural network "Detector" pipeline results are reported under "Detector"
    # (some LL versions use "DetectionResults" instead).
    detections = results.get("Detector", []) or results.get("DetectionResults", [])

    readings = []
    for det in detections:
        tx = det.get("tx", det.get("txnc", 0.0))
        ty = det.get("ty", det.get("tync", 0.0))
        ta = det.get("ta", 0.0)
        class_id = det.get("classID", det.get("class_id", -1))
        class_name = det.get("class", det.get("className", "unknown"))
        confidence = det.get("conf", det.get("confidence", 0.0))

        distance = estimate_distance(ty, camera_height_m, camera_angle_deg, object_height_m)

        readings.append(DetectionReading(
            pipeline=pipeline_index,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            tx=tx,
            ty=ty,
            ta=ta,
            distance_m=distance,
        ))
    return readings


def print_readings(readings: list) -> None:
    if not readings:
        print("  (no objects detected)")
        return
    for r in readings:
        dist_str = f"{r.distance_m:.2f} m" if r.distance_m is not None else "N/A"
        print(f"  class={r.class_name:<12} id={r.class_id:<3} conf={r.confidence:5.2f} "
              f"tx={r.tx:7.2f} ty={r.ty:7.2f} ta={r.ta:6.2f} distance≈{dist_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Read trained object detections from a Limelight neural detector pipeline.")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Limelight hostname or IP (default: %(default)s)")
    parser.add_argument("--pipeline", type=int, default=NEURAL_DETECTOR_PIPELINE,
                        help="Neural detector pipeline index (default: %(default)s)")
    parser.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT_M,
                        help="Camera lens height in meters")
    parser.add_argument("--camera-angle", type=float, default=CAMERA_ANGLE_DEG,
                        help="Camera mount angle in degrees above horizontal")
    parser.add_argument("--object-height", type=float, default=OBJECT_HEIGHT_M,
                        help="Trained object's center height in meters")
    parser.add_argument("--loop", action="store_true",
                        help="Continuously poll the pipeline until Ctrl+C")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between loop iterations")
    args = parser.parse_args()

    def run_once():
        print(f"\n=== Neural Detector (Pipeline {args.pipeline}) ===")
        try:
            readings = read_neural_detections(args.host, args.pipeline,
                                               args.camera_height, args.camera_angle,
                                               args.object_height)
            print_readings(readings)
        except requests.RequestException as exc:
            print(f"  [error] could not read pipeline {args.pipeline}: {exc}")

    if args.loop:
        try:
            while True:
                run_once()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        run_once()


if __name__ == "__main__":
    main()
