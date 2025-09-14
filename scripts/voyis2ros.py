#!/usr/bin/env python3
import os, re, glob, json, time, argparse
from typing import List, Tuple
from datetime import datetime, timezone

import numpy as np
import cv2

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from rclpy.serialization import serialize_message

# ---------- filename timestamp parsing ----------
# Matches ...SYSTEM_YYYY-MM-DDTHHMMSS.ssssss...
_TS_RE = re.compile(r"SYSTEM_(\d{4}-\d{2}-\d{2})T(\d{6})\.(\d+)", re.IGNORECASE)

def parse_ts_ns_from_filename(path: str) -> int:
    """
    Extract epoch-ns timestamp from Voyis filename.
    Example: image_left_processed_SYSTEM_2025-05-29T214942.980413_CAL_559465.jpg
    """
    name = os.path.basename(path)
    m = _TS_RE.search(name)
    if not m:
        # Fallback: file mtime if no timestamp in name
        return int(os.path.getmtime(path) * 1e9)

    date_str, hms_str, frac = m.groups()
    # hms is HHMMSS (no colon)
    hh, mm, ss = int(hms_str[0:2]), int(hms_str[2:4]), int(hms_str[4:6])

    # normalize fractional seconds to nanoseconds
    # (pad/truncate to 9 digits)
    frac = (frac + "000000000")[:9]
    nsec = int(frac)

    dt = datetime(
        year=int(date_str[0:4]), month=int(date_str[5:7]), day=int(date_str[8:10]),
        hour=hh, minute=mm, second=ss, tzinfo=timezone.utc
    )
    return int(dt.timestamp()) * 1_000_000_000 + nsec

# ---------- CameraInfo from Voyis JSON ----------
def load_voyis_json(json_path: str):
    with open(json_path, "r") as f:
        J = json.load(f)
    fx = float(J["camera_intrinsics"]["fx"])
    fy = float(J["camera_intrinsics"]["fy"])
    cx = float(J["camera_intrinsics"]["cx"])
    cy = float(J["camera_intrinsics"]["cy"])
    width  = int(J["camera_intrinsics"]["width"])
    height = int(J["camera_intrinsics"]["height"])
    baseline = float(J.get("stereo_baseline", 0.0))
    return fx, fy, cx, cy, width, height, baseline

def make_cam_info(fx, fy, cx, cy, width, height, frame_id, stamp_ns, Tx=0.0) -> CameraInfo:
    ci = CameraInfo()
    ci.header = Header()
    ci.header.frame_id = frame_id
    ci.header.stamp.sec = stamp_ns // 1_000_000_000
    ci.header.stamp.nanosec = stamp_ns % 1_000_000_000
    ci.width = width
    ci.height = height
    ci.distortion_model = "plumb_bob"
    # No distortion provided in JSON → assume rectified (D = [])
    ci.d = []
    ci.k = [fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0]
    ci.r = [1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0]
    # P with Tx term (left Tx=0; right Tx=-fx*baseline)
    ci.p = [fx, 0.0, cx, Tx,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0]
    return ci

# ---------- image helpers ----------
def to_encoding(img, encoding: str):
    if encoding == "mono8":
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype == np.uint16:
            img = (img/256).astype(np.uint8)
    elif encoding == "rgb8":
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")
    return img

def make_image_msg(cv_img, encoding: str, stamp_ns: int, frame_id: str) -> Image:
    msg = Image()
    msg.header = Header()
    msg.header.stamp.sec = stamp_ns // 1_000_000_000
    msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
    msg.header.frame_id = frame_id
    msg.encoding = encoding
    msg.height, msg.width = cv_img.shape[:2]
    msg.is_bigendian = 0
    msg.step = msg.width * (1 if encoding == "mono8" else 3)
    msg.data = cv_img.tobytes()
    return msg

# ---------- main ----------
def main():
    start_time = time.time()
    ap = argparse.ArgumentParser(description="Convert Voyis stereo image folders to rosbag2 for Isaac ROS Visual SLAM")
    ap.add_argument("--left_dir", required=True)
    ap.add_argument("--right_dir", required=True)
    ap.add_argument("--voyis_json", required=True, help="Voyis calibration JSON (intrinsics + baseline)")
    ap.add_argument("--left_frame_id", default="camera_left_optical")
    ap.add_argument("--right_frame_id", default="camera_right_optical")
    ap.add_argument("--pattern", default="*.jpg")
    ap.add_argument("--encoding", default="mono8", choices=["mono8","rgb8"])
    ap.add_argument("--bag", required=True)
    ap.add_argument("--storage", default="sqlite3", choices=["sqlite3","mcap"])
    args = ap.parse_args()

    left_files  = sorted(glob.glob(os.path.join(args.left_dir,  args.pattern)))
    right_files = sorted(glob.glob(os.path.join(args.right_dir, args.pattern)))
    if not left_files or not right_files:
        raise SystemExit("No images found; check --left_dir/--right_dir and --pattern")

    # Build (file, ts_ns) lists and align by timestamp (closest neighbor)
    left_ts  = [(f, parse_ts_ns_from_filename(f))  for f in left_files]
    right_ts = [(f, parse_ts_ns_from_filename(f))  for f in right_files]

    # Two-pointer merge to pair closest timestamps
    pairs: List[Tuple[str,str,int]] = []
    j = 0
    for lf, lts in left_ts:
        # advance right index while closer
        while j+1 < len(right_ts) and abs(right_ts[j+1][1] - lts) <= abs(right_ts[j][1] - lts):
            j += 1
        rf, rts = right_ts[j]
        ts_ns = min(lts, rts)  # common stamp (or choose average)
        pairs.append((lf, rf, ts_ns))

    # Load Voyis calibration
    fx, fy, cx, cy, width, height, baseline = load_voyis_json(args.voyis_json)

    # Bag writer
    storage_options = StorageOptions(uri=args.bag, storage_id=args.storage)
    converter_options = ConverterOptions(input_serialization_format="cdr",
                                         output_serialization_format="cdr")
    writer = SequentialWriter()
    writer.open(storage_options, converter_options)

    for name, typ in [
        ("/visual_slam/image_0",       "sensor_msgs/msg/Image"),
        ("/visual_slam/image_1",       "sensor_msgs/msg/Image"),
        ("/visual_slam/camera_info_0", "sensor_msgs/msg/CameraInfo"),
        ("/visual_slam/camera_info_1", "sensor_msgs/msg/CameraInfo"),
    ]:
        writer.create_topic(TopicMetadata(name=name, type=typ, serialization_format="cdr"))

    # Write
    print(f"Writing {len(pairs)} pairs → {args.bag}")
    for lf, rf, ts in pairs:
        img_l = cv2.imread(lf, cv2.IMREAD_UNCHANGED)
        img_r = cv2.imread(rf, cv2.IMREAD_UNCHANGED)
        if img_l is None or img_r is None:
            print(f"[WARN] Failed to read {lf} or {rf}; skipping")
            continue
        img_l = to_encoding(img_l, args.encoding)
        img_r = to_encoding(img_r, args.encoding)

        msg_l = make_image_msg(img_l, args.encoding, ts, args.left_frame_id)
        msg_r = make_image_msg(img_r, args.encoding, ts, args.right_frame_id)

        # CameraInfo (left Tx=0; right Tx=-fx*baseline)
        ci0 = make_cam_info(fx, fy, cx, cy, width, height, args.left_frame_id,  ts, Tx=0.0)
        ci1 = make_cam_info(fx, fy, cx, cy, width, height, args.right_frame_id, ts, Tx=-(fx*baseline))

        writer.write("/visual_slam/image_0",       serialize_message(msg_l), ts)
        writer.write("/visual_slam/image_1",       serialize_message(msg_r), ts)
        writer.write("/visual_slam/camera_info_0", serialize_message(ci0),   ts)
        writer.write("/visual_slam/camera_info_1", serialize_message(ci1),   ts)
    
    duration = time.time() - start_time
    print(f"Finished in {duration:.2f} seconds")

if __name__ == "__main__":
    main()
