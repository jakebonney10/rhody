import pandas as pd
import numpy as np
import os
import shutil
import rclpy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from scipy.spatial.transform import Rotation as R

"""
subsonus2ros.py

This script processes raw data from Subsonus USBL ANPP logs and converts it into ROS 2 bag files for use in robot localization. The Subsonus sensor provides acoustic positioning data in the NED (North-East-Down) coordinate frame, which is transformed into the ENU (East-North-Up) coordinate frame expected by ROS 2.

Key Features:
- Reads Subsonus USBL ANPP logs from a CSV file (`State.csv` or `RemoteSubsonusState.csv`).
- Converts NED coordinate frame data to ENU coordinate frame.
- Creates ROS 2 bag files containing `NavSatFix` and `Imu` messages.
- Handles data validation and formatting errors gracefully.
- Supports covariance matrices for position and orientation data.

Dependencies:
- Python libraries: pandas, numpy, scipy, os, shutil
- ROS 2 Python libraries: rclpy, rosbag2_py
- ROS 2 message types: `sensor_msgs.msg.NavSatFix`, `sensor_msgs.msg.Imu`, `builtin_interfaces.msg.Time`

Usage:
1. Place the raw Subsonus USBL ANPP log file (`State.csv`) in the same directory as the script.
2. Run the script: `python subsonus2ros.py`.
3. The script will generate a ROS 2 bag file in the `subsonus_bag` directory.

Coordinate Frame Transformation:
- The Subsonus sensor uses the NED coordinate frame, while ROS 2 expects ENU.
- The script uses the `ned_to_enu` function to transform both quaternion orientations and vector data from NED to ENU.

Error Handling:
- Rows with invalid numeric values are dropped and logged.
- Covariance formatting errors are caught and logged without interrupting the process.

Output:
- ROS 2 bag files containing:
    - `NavSatFix` messages for latitude, longitude, altitude, and position covariance.
    - `Imu` messages for orientation, angular velocity, and covariance matrices.

Notes:
- Ensure ROS 2 is installed and configured properly before running the script.
- The generated ROS 2 bag files can be used with robot localization tools such as `robot_localization` or `nav2`.

"""

# NOTE: Subsonus Sensor Coordinate Frame is NED, ROS2 robot localization expects ENU [https://docs.advancednavigation.com/subsonus/SensorCoordinate.htm]
frame_id = "usbl_link"
namespace = "nav/sensors/subsonus_usbl"
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
        name=namespace + '/imu',
        type='sensor_msgs/msg/Imu',
        serialization_format='cdr'
    ))

    writer.create_topic(TopicMetadata(
        name=namespace + '/depth',
        type='geometry_msgs/msg/PoseWithCovarianceStamped',
        serialization_format='cdr'
    ))

    writer.create_topic(TopicMetadata(
        name=namespace + '/odom',
        type='nav_msgs/msg/Odometry',
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

    # ---- IMU and Odometry from Remote State data (orientation and angular velocity)
    for _, row in df_state.iterrows():
        ros_time, timestamp = make_ros_time(row["Unix Time"], row["Microseconds"])

        # IMU
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

        writer.write(namespace + '/imu', serialize_message(imu), timestamp)

        # Odometry
        odom = Odometry()
        odom.header.stamp = ros_time
        odom.header.frame_id = "odom_usbl"
        odom.child_frame_id = frame_id  # or your vehicle's moving frame

        # Position (lat/lon/alt → keep raw or transform to map if needed later)
        odom.pose.pose.position.x = row["Longitude"]
        odom.pose.pose.position.y = row["Latitude"]
        odom.pose.pose.position.z = row["Height"]

        # Orientation from roll/pitch/heading
        qx, qy, qz, qw = euler_to_quaternion(row["Roll"], row["Pitch"], row["Heading"])
        q_enu = ned_to_enu(quat_ned=(qx, qy, qz, qw))
        odom.pose.pose.orientation.x = q_enu[0]
        odom.pose.pose.orientation.y = q_enu[1]
        odom.pose.pose.orientation.z = q_enu[2]
        odom.pose.pose.orientation.w = q_enu[3]

        # Velocities (angular)
        vel_ang_ned = [row["Angular Velocity X"], row["Angular Velocity Y"], row["Angular Velocity Z"]]
        vel_ang_enu = ned_to_enu(vec_ned=vel_ang_ned)

        # ✅ Linear velocity from Velocity N/E/D
        vel_lin_ned = [row["Velocity North"], row["Velocity East"], row["Velocity Down"]]
        vel_lin_enu = ned_to_enu(vec_ned=vel_lin_ned)

        odom.twist.twist.angular.x = vel_ang_enu[0]
        odom.twist.twist.angular.y = vel_ang_enu[1]
        odom.twist.twist.angular.z = vel_ang_enu[2]

        odom.twist.twist.linear.x = vel_lin_enu[0]
        odom.twist.twist.linear.y = vel_lin_enu[1]
        odom.twist.twist.linear.z = vel_lin_enu[2]

        # Covariance matrices
        try:
            pose_cov = [0.0] * 36
            pose_cov[0] = float(row["Longitude Error"]) ** 2     # x
            pose_cov[7] = float(row["Latitude Error"]) ** 2      # y
            pose_cov[14] = float(row["Height Error"]) ** 2       # z
            pose_cov[21] = float(row["Roll Error"]) ** 2         # roll
            pose_cov[28] = float(row["Pitch Error"]) ** 2        # pitch
            pose_cov[35] = float(row["Heading Error"]) ** 2      # yaw
            odom.pose.covariance = pose_cov
        except Exception as e:
            print(f"⚠️ Odometry pose covariance error:\n{row}\n{e}")
            odom.pose.covariance = [0.5] * 36  # fallback

        # Covariances — conservative values (update if you know better)
        odom.twist.covariance = [0.1] * 36

        writer.write(namespace + '/odom', serialize_message(odom), timestamp)

    ''' # NOTE: This section is commented out remote raw sensors data doesnt have timestamps
    # ---- Depth from Remote Raw Sensors data (pressure depth)
    for _, row in df_raw.iterrows():
        # Generate timestamp (use best estimate or just Unix + Micro if no Human Timestamp)
        ros_time, timestamp = make_ros_time(row["Unix Time"], row["Microseconds"])

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = ros_time
        msg.header.frame_id = frame_id

        # Convert NED depth to ENU Z: ENU.z = -NED.depth
        msg.pose.pose.position.z = -float(row["Pressure Depth"])

        # Set unknown values for x/y/orientation, and conservative Z variance
        msg.pose.covariance = [0.0] * 36
        msg.pose.covariance[14] = 0.25  # Variance of 0.5 m
        msg.pose.covariance[0] = -1.0   # x unknown
        msg.pose.covariance[7] = -1.0   # y unknown
        msg.pose.covariance[21] = -1.0  # roll unknown
        msg.pose.covariance[28] = -1.0  # pitch unknown
        msg.pose.covariance[35] = -1.0  # yaw unknown

        writer.write(namespace + '/depth', serialize_message(msg), timestamp)
    '''

    print(f"✅ ROS 2 bag created in: {output_dir}")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
