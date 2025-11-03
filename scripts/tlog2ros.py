#!/usr/bin/env python3
"""
tlog2ros.py

Convert ArduSub .tlog files to ROS2 bag files for underwater vehicle navigation.
- Extracts IMU data (attitude) from ATTITUDE messages
- Extracts depth data from pressure sensors as PoseWithCovarianceStamped
- Extracts temperature data from SCALED_PRESSURE2 messages
- Creates separate ROS2 bag files per dive in /rhody/nav/sensors/blueos namespace
- Filters for underwater segments (depth >= 10m, duration >= 10 minutes)

Usage:
    python3 tlog2ros.py
    (Interactive prompts for folder and dive location)
"""

from __future__ import annotations
import csv, sys, re, math, os, shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import rclpy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3Stamped
from sensor_msgs.msg import Imu, Temperature
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from scipy.spatial.transform import Rotation as R
import numpy as np

import matplotlib.pyplot as plt
from pymavlink import mavutil  # type: ignore

# ───────── constants ─────────
DEPTH_MIN   = 10.0       # m
MIN_DIVE_S  = 600        # s (10 minutes)
UTC         = timezone.utc

# ROS2 configuration
NAMESPACE = "rhody/nav/sensors/blueos"
FRAME_ID = "blueos_imu_link"

# Covariance estimates for IMU data (conservative estimates)
IMU_ORIENTATION_COV = [0.01, 0.0, 0.0,    # roll
                      0.0, 0.01, 0.0,     # pitch  
                      0.0, 0.0, 0.05]     # yaw (less accurate)

IMU_ANGULAR_VEL_COV = [0.001, 0.0, 0.0,
                      0.0, 0.001, 0.0,
                      0.0, 0.0, 0.001]

IMU_LINEAR_ACC_COV = [0.1, 0.0, 0.0,
                     0.0, 0.1, 0.0,
                     0.0, 0.0, 0.1]

# Depth covariance (pressure sensor accuracy)
DEPTH_COV = [1e6, 0.0, 0.0, 0.0, 0.0, 0.0,    # x (unused)
            0.0, 1e6, 0.0, 0.0, 0.0, 0.0,     # y (unused)
            0.0, 0.0, 0.25, 0.0, 0.0, 0.0,    # z (depth) - 0.5m std dev
            0.0, 0.0, 0.0, 1e6, 0.0, 0.0,     # roll (unused)
            0.0, 0.0, 0.0, 0.0, 1e6, 0.0,     # pitch (unused)
            0.0, 0.0, 0.0, 0.0, 0.0, 1e6]     # yaw (unused)

def sanitize(tag: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]", "", tag)
    if not clean:
        raise ValueError("Dive location must contain letters/numbers.")
    return clean.upper()

def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def make_ros_time(epoch_time: float) -> tuple[Time, int]:
    """Convert epoch time to ROS Time message and nanosecond timestamp."""
    sec = int(epoch_time)
    nanosec = int((epoch_time - sec) * 1e9)
    t = Time()
    t.sec = sec
    t.nanosec = nanosec
    return t, sec * int(1e9) + nanosec

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (radians) to quaternion (x, y, z, w)."""
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return tuple(r.as_quat())

def create_imu_message(epoch_time: float, roll: float, pitch: float, yaw: float) -> Imu:
    """Create IMU message from attitude data."""
    ros_time, _ = make_ros_time(epoch_time)
    
    msg = Imu()
    msg.header.stamp = ros_time
    msg.header.frame_id = FRAME_ID
    
    # Convert degrees to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # Set orientation from Euler angles
    qx, qy, qz, qw = euler_to_quaternion(roll_rad, pitch_rad, yaw_rad)
    msg.orientation.x = qx
    msg.orientation.y = qy
    msg.orientation.z = qz
    msg.orientation.w = qw
    
    # Set covariances (no angular velocity or linear acceleration from ArduSub attitude)
    msg.orientation_covariance = IMU_ORIENTATION_COV
    msg.angular_velocity_covariance[0] = -1.0  # Mark as unknown
    msg.linear_acceleration_covariance[0] = -1.0  # Mark as unknown
    
    return msg

def create_depth_message(epoch_time: float, depth: float) -> PoseWithCovarianceStamped:
    """Create depth message as PoseWithCovarianceStamped."""
    ros_time, _ = make_ros_time(epoch_time)
    
    msg = PoseWithCovarianceStamped()
    msg.header.stamp = ros_time
    msg.header.frame_id = "utm_local"  # Global reference frame
    
    # Set depth as negative Z (ENU convention: positive Z is up)
    msg.pose.pose.position.x = 0.0  # Unknown horizontal position
    msg.pose.pose.position.y = 0.0  # Unknown horizontal position
    msg.pose.pose.position.z = -depth  # Depth as negative Z
    
    # Identity quaternion for orientation (unknown)
    msg.pose.pose.orientation.x = 0.0
    msg.pose.pose.orientation.y = 0.0
    msg.pose.pose.orientation.z = 0.0
    msg.pose.pose.orientation.w = 1.0
    
    # Set covariance
    msg.pose.covariance = DEPTH_COV
    
    return msg

def create_temperature_message(epoch_time: float, temp_celsius: float) -> Temperature:
    """Create temperature message."""
    ros_time, _ = make_ros_time(epoch_time)
    
    msg = Temperature()
    msg.header.stamp = ros_time
    msg.header.frame_id = FRAME_ID
    msg.temperature = temp_celsius
    msg.variance = 1.0  # 1°C variance estimate
    
    return msg

def create_rosbag_for_dive(dive_data: List[Dict], dive_id: int, output_dir: Path) -> None:
    """Create a ROS2 bag file for a single dive."""
    rclpy.init()
    
    # Create bag directory name
    dive_label = f"blueos_dive_{dive_id:03d}"
    bag_dir = output_dir / f"{dive_label}_bag"
    
    if bag_dir.exists():
        shutil.rmtree(bag_dir)
    
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=str(bag_dir), storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )
    
    # Create topics
    topics = [
        (f"{NAMESPACE}/imu", 'sensor_msgs/msg/Imu'),
        (f"{NAMESPACE}/depth", 'geometry_msgs/msg/PoseWithCovarianceStamped'),
        (f"{NAMESPACE}/temperature", 'sensor_msgs/msg/Temperature')
    ]
    
    for topic_name, topic_type in topics:
        writer.create_topic(TopicMetadata(
            name=topic_name,
            type=topic_type,
            serialization_format='cdr'
        ))
    
    print(f"🔄 Creating ROS2 bag for dive {dive_id} with {len(dive_data)} messages...")
    
    # Process each data point
    for data_point in dive_data:
        epoch_time = data_point["unix_time_us"] / 1e6
        _, timestamp_ns = make_ros_time(epoch_time)
        
        # Create and write IMU message
        imu_msg = create_imu_message(
            epoch_time,
            data_point["roll_deg"],
            data_point["pitch_deg"], 
            data_point["heading_deg"]
        )
        writer.write(f"{NAMESPACE}/imu", serialize_message(imu_msg), timestamp_ns)
        
        # Create and write depth message
        depth_msg = create_depth_message(epoch_time, data_point["depth_m"])
        writer.write(f"{NAMESPACE}/depth", serialize_message(depth_msg), timestamp_ns)
        
        # Create and write temperature message (if available)
        if data_point["water_temp_c"] is not None:
            temp_msg = create_temperature_message(epoch_time, data_point["water_temp_c"])
            writer.write(f"{NAMESPACE}/temperature", serialize_message(temp_msg), timestamp_ns)
    
    rclpy.shutdown()
    print(f"✅ ROS2 bag created: {bag_dir}")

def boot_offset(tlog: Path) -> float | None:
    m = mavutil.mavlink_connection(str(tlog))
    while True:
        msg = m.recv_match(type="SYSTEM_TIME", blocking=False)
        if msg is None:
            break
        if msg.time_unix_usec:
            m.close()
            return msg.time_unix_usec/1e6 - msg.time_boot_ms/1e3
    m.close()
    return None

def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs) if xs else float("nan")

def main() -> None:
    folder = Path(input("Folder with .tlog files: ").strip()).expanduser()
    if not folder.is_dir():
        sys.exit("Folder not found.")
    tag = sanitize(input("Dive location (one word): "))

    tlogs = sorted(folder.glob("*.tlog"))
    if not tlogs:
        sys.exit("No .tlog files.")

    all_rows: List[Dict] = []
    global_ctr = Counter()

    # parse all .tlogs
    for tlog in tlogs:
        print(f"Reading {tlog.name} …")
        off = boot_offset(tlog)
        m = mavutil.mavlink_connection(str(tlog))
        m.wait_heartbeat()

        p0_air = depth = temp = None
        while (msg := m.recv_match(blocking=False)):
            if not hasattr(msg, "time_boot_ms"):
                continue
            mt = msg.get_type()

            if mt == "SCALED_PRESSURE":
                p0_air = msg.press_abs * 100
            elif mt == "SCALED_PRESSURE2" and p0_air:
                depth = (msg.press_abs * 100 - p0_air) / (1000 * 9.80665)
                depth = max(0.0, depth)
                temp = msg.temperature / 100

            elif mt == "ATTITUDE":
                global_ctr["att_total"] += 1
                if depth is None:
                    global_ctr["no_depth"] += 1
                    continue
                if depth < DEPTH_MIN:
                    global_ctr["shallow"] += 1
                    continue

                tb = msg.time_boot_ms
                if off is not None:
                    epoch = off + tb/1000
                elif getattr(msg, "_timestamp", 0):
                    epoch = msg._timestamp
                else:
                    epoch = tb/1000.0

                tsu = int(epoch * 1e6)
                all_rows.append({
                    "unix_time_us":   tsu,
                    "timestamp_utc":  iso_utc(epoch),
                    "roll_deg":       round(math.degrees(msg.roll), 3),
                    "pitch_deg":      round(math.degrees(msg.pitch), 3),
                    "heading_deg":    round((math.degrees(msg.yaw)+360) % 360, 3),
                    "depth_m":        round(depth, 2),
                    "water_temp_c":   round(temp, 2) if temp is not None else None,
                    "dive_num":       0
                })
                global_ctr["deep_rows"] += 1

        m.close()

    if not all_rows:
        sys.exit("No deep attitude rows found.")

    # ───── Raw stats ─────
    print("\n=== Raw processing stats ===")
    print(f"  ATTITUDE msgs total: {global_ctr['att_total']}")
    print(f"  Dropped no-depth   : {global_ctr['no_depth']}")
    print(f"  Dropped shallow    : {global_ctr['shallow']}")
    print(f"  Kept deep rows     : {global_ctr['deep_rows']}")

    # segment by gaps >10s
    all_rows.sort(key=lambda r: r["unix_time_us"])
    segments = []
    start = 0
    for i in range(1, len(all_rows)):
        if all_rows[i]["unix_time_us"] - all_rows[i-1]["unix_time_us"] > 10_000_000:
            segments.append((start, i-1))
            start = i
    segments.append((start, len(all_rows)-1))

    # build summary and kept-rows
    summary = []
    kept_rows: List[Dict] = []
    dive_id = 0

    for seg_idx, (s, e) in enumerate(segments, start=1):
        t0 = all_rows[s]["unix_time_us"]
        t1 = all_rows[e]["unix_time_us"]
        dur_s = (t1 - t0) / 1e6
        orig_count = e - s + 1

        if dur_s >= MIN_DIVE_S:
            dive_id += 1
            status = "kept"
            seg = all_rows[s:e+1]
            for r in seg:
                r["dive_num"] = dive_id
            kept_rows.extend(seg)
            depths = [r["depth_m"] for r in seg]
            rows_kept = len(seg)
            mean_dep = mean(depths)
            max_dep = max(depths)
            dive_label = f"BLUEOS_DIVE_{dive_id:03d}"
        else:
            status = "dropped"
            rows_kept = 0
            mean_dep = ""
            max_dep = ""
            dive_label = ""

        summary.append({
            "segment_idx":   seg_idx,
            "start_utc":     iso_utc(t0/1e6),
            "end_utc":       iso_utc(t1/1e6),
            "duration_min":  f"{dur_s/60:.1f}",
            "status":        status,
            "reason":        f"{dur_s/60:.1f}min<{MIN_DIVE_S/60:.1f}min" if status=="dropped" else "",
            "dive_id":       dive_label,
            "rows_original": orig_count,
            "rows_kept":     rows_kept,
            "mean_depth_m":  mean_dep,
            "max_depth_m":   max_dep
        })

    total_segs = len(summary)
    kept_segs  = sum(1 for x in summary if x["status"]=="kept")
    drop_segs  = total_segs - kept_segs
    print(f"\nSegments: {total_segs}, kept: {kept_segs}, dropped: {drop_segs}")

    # write segment summary CSV
    sum_csv = folder.parent / f"{tag.lower()}_dive_summary.csv"
    fields = [
        "segment_idx","start_utc","end_utc","duration_min",
        "status","reason","dive_id",
        "rows_original","rows_kept",
        "mean_depth_m","max_depth_m"
    ]
    with sum_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f"\nSegment summary written to: {sum_csv}")

    # write kept-rows CSV
    if not kept_rows:
        sys.exit("No dives kept.")
    kept_csv = folder.parent / f"{tag.lower()}_rovlog.csv"
    with kept_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept_rows[0].keys()))
        w.writeheader()
        w.writerows(kept_rows)
    print(f"Kept rows CSV written to: {kept_csv}")

    # create ROS2 bags for each dive
    by_dive = defaultdict(list)
    for r in kept_rows:
        by_dive[r["dive_num"]].append(r)

    print(f"\n🚀 Creating ROS2 bag files...")
    ros_output_dir = folder.parent / "ros"
    ros_output_dir.mkdir(exist_ok=True)
    
    for dive_id, dive_data in by_dive.items():
        create_rosbag_for_dive(dive_data, dive_id, ros_output_dir)

    print(f"\n📊 Summary:")
    print(f"   Total dives processed: {len(by_dive)}")
    print(f"   ROS2 bags created in: {ros_output_dir}")
    print(f"   Topics per bag:")
    print(f"     - {NAMESPACE}/imu (sensor_msgs/msg/Imu)")
    print(f"     - {NAMESPACE}/depth (geometry_msgs/msg/PoseWithCovarianceStamped)")
    print(f"     - {NAMESPACE}/temperature (sensor_msgs/msg/Temperature)")

    # per-dive depth plots
    for did, sub in by_dive.items():
        times = [datetime.fromtimestamp(r["unix_time_us"]/1e6, tz=UTC) for r in sub]
        deps  = [r["depth_m"] for r in sub]
        t0 = times[0]
        dur_min   = (times[-1] - t0).total_seconds() / 60
        date_str  = t0.strftime("%Y%m%d")      # for filename
        date_label= t0.strftime("%Y-%m-%d")    # for title
        label     = f"BLUEOS_DIVE_{did:03d}"

        plt.figure()
        plt.plot(times, deps)
        plt.title(f"{label} – {date_label} – {dur_min:.1f} min")
        plt.xlabel("UTC")
        plt.ylabel("Depth (m)")
        plt.gca().invert_yaxis()
        plt.gcf().autofmt_xdate()

        outfn = folder.parent / f"{label}_{date_str}_depth.png"
        plt.savefig(outfn, dpi=150)
        plt.close()

    print(f"\n📈 Depth plots saved as BLUEOS_DIVE_###_YYYYMMDD_depth.png")

if __name__ == "__main__":
    main()