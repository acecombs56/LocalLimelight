#!/usr/bin/env python3
"""
LocalLimelight - AprilTag reader for a Limelight attached to your laptop.

Reads AprilTag detections from two different Limelight pipelines and prints
the tag's X (tx), Y (ty), and an estimated distance to the tag for each.

Requirements:
    pip install requests

Usage:
    python limelight_apriltag.py --host 10.TE.AM.11 --pipeline-a 0 --pipeline-b 1

Notes:
    - This talks directly to the Limelight's HTTP JSON endpoint
      (http://<host>:5807/results), so no NetworkTables/robot code is needed -
      it just needs the Limelight reachable on your network (e.g. plugged
      into your laptop via ethernet, with the laptop on the same subnet).
    - Distance estimation uses the standard camera-height/angle trig formula:

          distance = (tagHeight - cameraHeight) / tan(cameraAngle + ty)

      Adjust CAMERA_HEIGHT_M, CAMERA_ANGLE_DEG, and TAG_HEIGHT_M below (or via
      CLI flags) to match your physical mounting setup for real distances.
      If you don't calibrate these, distance will still be computed but may
      not be accurate - it's a placeholder based on defaults.
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

# --- Physical setup defaults (calibrate these for accurate distance) -----
CAMERA_HEIGHT_M = 1.27      # height of the limelight lens off the ground/reference plane
CAMERA_ANGLE_DEG = -0.8 # 20.0     # mounting angle of the camera above horizontal
TAG_HEIGHT_M = 1.54         # height of the center of the AprilTag off the same reference plane
# --------------------------------------------------------------------------


@dataclass
class TagReading:
    pipeline: int
    tag_id: int
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
                       tag_height_m: float) -> Optional[float]:
    """Estimate distance to tag using camera height/angle trig formula."""
    angle_total_rad = math.radians(camera_angle_deg + ty_deg)
    denom = math.tan(angle_total_rad)
    if abs(denom) < 1e-6:
        return None
    distance = (tag_height_m - camera_height_m) / denom
    return distance


def read_pipeline(host: str,
                   pipeline_index: int,
                   camera_height_m: float,
                   camera_angle_deg: float,
                   tag_height_m: float,
                   settle_time: float = 0.25) -> list:
    """Switch to a pipeline, wait for it to settle, then read tag detections."""
    set_pipeline(host, pipeline_index)
    time.sleep(settle_time)

    data = get_results(host)
    results = data.get("Results", data)  # some LL versions nest under "Results"

    fiducials = results.get("Fiducial", []) or results.get("fiducialResults", [])

    readings = []
    for tag in fiducials:
        tx = tag.get("tx", tag.get("txnc", 0.0))
        ty = tag.get("ty", tag.get("tync", 0.0))
        ta = tag.get("ta", 0.0)
        tag_id = tag.get("fID", tag.get("id", -1))

        distance = estimate_distance(ty, camera_height_m, camera_angle_deg, tag_height_m)

        readings.append(TagReading(
            pipeline=pipeline_index,
            tag_id=tag_id,
            tx=tx,
            ty=ty,
            ta=ta,
            distance_m=distance,
        ))
    return readings


def print_readings(readings: list) -> None:
    if not readings:
        print("  (no tags detected)")
        return
    for r in readings:
        dist_str = f"{r.distance_m:.2f} m" if r.distance_m is not None else "N/A"
        print(f"  tag_id={r.tag_id:<3} tx={r.tx:7.2f} ty={r.ty:7.2f} ta={r.ta:6.2f} "
              f"distance≈{dist_str}")


def main():
    parser = argparse.ArgumentParser(description="Read AprilTags from two Limelight pipelines.")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Limelight hostname or IP (default: %(default)s)")
    parser.add_argument("--pipeline-a", type=int, default=8, help="First pipeline index")
    parser.add_argument("--pipeline-b", type=int, default=9, help="Second pipeline index")
    parser.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT_M,
                        help="Camera lens height in meters")
    parser.add_argument("--camera-angle", type=float, default=CAMERA_ANGLE_DEG,
                        help="Camera mount angle in degrees above horizontal")
    parser.add_argument("--tag-height", type=float, default=TAG_HEIGHT_M,
                        help="AprilTag center height in meters")
    parser.add_argument("--loop", action="store_true",
                        help="Continuously poll both pipelines until Ctrl+C")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between loop iterations")
    args = parser.parse_args()

    def run_once():
        print(f"\n=== Pipeline {args.pipeline_a} ===")
        try:
            readings_a = read_pipeline(args.host, args.pipeline_a,
                                        args.camera_height, args.camera_angle, args.tag_height)
            print_readings(readings_a)
        except requests.RequestException as exc:
            print(f"  [error] could not read pipeline {args.pipeline_a}: {exc}")

        print(f"=== Pipeline {args.pipeline_b} ===")
        try:
            readings_b = read_pipeline(args.host, args.pipeline_b,
                                        args.camera_height, args.camera_angle, args.tag_height)
            print_readings(readings_b)
        except requests.RequestException as exc:
            print(f"  [error] could not read pipeline {args.pipeline_b}: {exc}")

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
