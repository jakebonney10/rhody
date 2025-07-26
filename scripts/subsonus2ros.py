import pandas as pd
import numpy as np
import os
import shutil
import rclpy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from scipy.spatial.transform import Rotation as R

"""
subsonus2ros.py

This script processes raw data from Subsonus USBL ANPP logs and converts it into ROS 2 bag files for use in robot 
localization. The Subsonus sensor provides acoustic positioning data in the NED (North-East-Down) coordinate frame, 
which is transformed into the ENU (East-North-Up) coordinate frame expected by ROS 2.

Usage:
1. Place the raw Subsonus USBL ANPP log file (`State.csv`) in the same directory as the script.
2. Run the script: `python subsonus2ros.py`.
3. The script will generate a ROS 2 bag file in the `subsonus_bag` directory.

Coordinate Frame Transformation:
- The Subsonus sensor uses the NED coordinate frame, while ROS 2 expects ENU.
- The script uses the `ned_to_enu` function to transform both quaternion orientations and vector data from NED to ENU.

Output:
- ROS 2 bag files containing:
    - `NavSatFix` messages for latitude, longitude, altitude, and position covariance.
    - `Imu` messages for orientation, angular velocity, and covariance matrices.
    - `Odometry` messages for position, orientation, and velocity in the ENU frame.

Notes:
- Ensure ROS 2 is installed and configured properly before running the script.
- The generated ROS 2 bag files can be used with robot localization tools such as `robot_localization` or `nav2`.

"""

# NOTE: Subsonus Sensor Coordinate Frame is NED, ROS2 robot localization expects ENU [https://docs.advancednavigation.com/subsonus/SensorCoordinate.htm]
frame_id = "usbl_link"
namespace = "rhody/nav/sensors/subsonus_usbl"
output_dir = "subsonus_bag"
fn_state = "RemoteSubsonusState_2.csv"
fn_track = "RemoteTrack_2.csv"
fn_raw = "RemoteRawSensors_2.csv"

def euler_to_quaternion(roll, pitch, yaw):
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
    )

def ned_to_enu(quat_ned=None, vec_ned=None):
    results = []
    if quat_ned is not None:
        q_ned = R.from_quat(quat_ned)
        ned_to_enu_rot = R.from_euler('x', np.pi) * R.from_euler('z', np.pi)
        q_enu = (ned_to_enu_rot * q_ned).as_quat()
        results.append(q_enu)
    if vec_ned is not None:
        vec_enu = np.array([vec_ned[1], vec_ned[0], -vec_ned[2]])
        vec_enu[0] = vec_ned[1]
        vec_enu[1] = vec_ned[0]
        vec_enu[2] = -vec_ned[2]
        results.append(vec_enu)
    return tuple(results) if len(results) > 1 else results[0]

def make_ros_time(unix_sec, micros):
    total_time = float(unix_sec) + float(micros) * 1e-6
    sec = int(total_time)
    nanosec = int((total_time - sec) * 1e9)
    t = Time()
    t.sec = sec
    t.nanosec = nanosec
    return t, sec * int(1e9) + nanosec

def main():
    rclpy.init()
    df_state = pd.read_csv(fn_state)
    df_state.columns = df_state.columns.str.strip()
    df_track = pd.read_csv(fn_track)
    df_track.columns = df_track.columns.str.strip()
    df_raw = pd.read_csv(fn_raw)
    df_raw.columns = df_raw.columns.str.strip()

    numeric_fields = [
        "Latitude Error", "Longitude Error", "Height Error",
        "Roll Error", "Pitch Error", "Heading Error",
        "Angular Velocity X", "Angular Velocity Y", "Angular Velocity Z"
    ]

    bad_rows = df_state[df_state[numeric_fields].isnull().any(axis=1)]
    if not bad_rows.empty:
        print("⚠️ Dropping rows with invalid values:")
        print(bad_rows[["Human Timestamp"] + numeric_fields])

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=output_dir, storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    )

    # Register topics
    writer.create_topic(TopicMetadata(
        name=namespace + '/fix',
        type='sensor_msgs/msg/NavSatFix',
        serialization_format='cdr'
    ))

    writer.create_topic(TopicMetadata(
        name='/ins/fix',
        type='sensor_msgs/msg/NavSatFix',
        serialization_format='cdr'
    ))
    writer.create_topic(TopicMetadata(
        name='/ins/imu',
        type='sensor_msgs/msg/Imu',
        serialization_format='cdr'
    ))
    writer.create_topic(TopicMetadata(
        name='/ins/velocity',
        type='geometry_msgs/msg/TwistWithCovarianceStamped',
        serialization_format='cdr'
    ))

    # ---- NavSatFix from Remote Track data (accomms fix)
    for _, row in df_track.iterrows():
        ros_time, timestamp = make_ros_time(row["Unix Time"], row["Microseconds"])

        fix = NavSatFix()
        fix.header.stamp = ros_time
        fix.header.frame_id = frame_id
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = 0  # USBL, not GNSS

        fix.latitude = row["Remote Latitude"]
        fix.longitude = row["Remote Longitude"]
        fix.altitude = row["Remote Height"]  # Or use another height field if preferred

        try:
            fix.position_covariance = [
                float(row["Remote Latitude Standard Deviation"])**2, 0.0, 0.0,
                0.0, float(row["Remote Longitude Standard Deviation"])**2, 0.0,
                0.0, 0.0, float(row["Remote Height standard deviation"])**2
            ]
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        except Exception as e:
            print(f"⚠️ NavSatFix covariance formatting error:\n{row}\n{e}")
            continue

        writer.write(namespace + '/fix', serialize_message(fix), timestamp)

    # ---- NavSatFix, IMU, and Velocity from Remote Subsonus State INS data
    for _, row in df_state.iterrows():
        ros_time, timestamp = make_ros_time(row["Unix Time"], row["Microseconds"])

        # ---- NavSatFix
        fix = NavSatFix()
        fix.header.stamp = ros_time
        fix.header.frame_id = frame_id
        fix.status.status = NavSatStatus.STATUS_FIX
        fix.status.service = 0
        fix.latitude = row["Latitude"]
        fix.longitude = row["Longitude"]
        fix.altitude = row["Height"]

        try:
            fix.position_covariance = [
                float(row["Latitude Error"])**2, 0.0, 0.0,
                0.0, float(row["Longitude Error"])**2, 0.0,
                0.0, 0.0, float(row["Height Error"])**2
            ]
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        except Exception as e:
            print(f"⚠️ NavSatFix covariance formatting error:\n{row}\n{e}")
            continue

        writer.write('/ins/fix', serialize_message(fix), timestamp)

        # ---- IMU
        imu = Imu()
        imu.header.stamp = ros_time
        imu.header.frame_id = frame_id
        qx_ned, qy_ned, qz_ned, qw_ned = euler_to_quaternion(row["Roll"], row["Pitch"], row["Heading"])
        q_enu = ned_to_enu(quat_ned=(qx_ned, qy_ned, qz_ned, qw_ned))
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = q_enu

        try:
            imu.orientation_covariance = [
                float(row["Roll Error"])**2, 0.0, 0.0,
                0.0, float(row["Pitch Error"])**2, 0.0,
                0.0, 0.0, float(row["Heading Error"])**2
            ]
        except Exception as e:
            print(f"⚠️ IMU covariance formatting error:\n{row}\n{e}")
            continue

        angvel_ned = [row["Angular Velocity X"], row["Angular Velocity Y"], row["Angular Velocity Z"]]
        angvel_enu = ned_to_enu(vec_ned=angvel_ned)
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = angvel_enu
        imu.angular_velocity_covariance = [0.01] * 9
        imu.linear_acceleration_covariance = [-1.0] * 9  # unknown

        writer.write('/ins/imu', serialize_message(imu), timestamp)

        # ---- Velocity (TwistWithCovarianceStamped)
        twist = TwistWithCovarianceStamped()
        twist.header.stamp = ros_time
        twist.header.frame_id = frame_id

        lin_vel_ned = [row["Velocity North"], row["Velocity East"], row["Velocity Down"]]
        lin_vel_enu = ned_to_enu(vec_ned=lin_vel_ned)
        ang_vel_enu = ned_to_enu(vec_ned=angvel_ned)

        twist.twist.twist.linear.x, twist.twist.twist.linear.y, twist.twist.twist.linear.z = lin_vel_enu
        twist.twist.twist.angular.x, twist.twist.twist.angular.y, twist.twist.twist.angular.z = ang_vel_enu
        twist.twist.covariance = [0.05] * 36  # Conservative guess

        writer.write('/ins/velocity', serialize_message(twist), timestamp)


    print(f"✅ ROS 2 bag created in: {output_dir}")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
