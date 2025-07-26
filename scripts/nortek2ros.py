import os
import shutil
import pandas as pd
import numpy as np
import rclpy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from scipy.spatial.transform import Rotation as R

"""
nortek2ros.py

This script processes raw Nortek Nucleus DVL logs (Bottom Track, IMU, and INS data) 
and converts them into ROS 2 bag files for use with robot localization systems. 
The Nortek sensor data, originally in the NED (North-East-Down) coordinate frame, 
is transformed into the ENU (East-North-Up) frame as required by ROS 2 conventions.

Usage:
- Place the raw Nortek CSV files ("Bottom Track.csv", "IMU.csv", "INS.csv") in the script's directory.
- Run the script to generate a ROS 2 bag file in the specified output directory.

Coordinate Frame Conversion:
- NED (North-East-Down) → ENU (East-North-Up) for both vectors and quaternions.

Output:
- ROS 2 bag files containing:
    - `TwistWithCovarianceStamped` messages for velocity data.
    - `Imu` messages for orientation and angular velocity data.
    - `PoseWithCovarianceStamped` messages for depth data.
    - (Optional) `PoseWithCovarianceStamped` messages for INS data.

Note:
- INS data conversion is commented out but can be enabled if needed.
- Ensure the Nortek CSV files are formatted correctly with appropriate column names.
"""

# NOTE: Nortek Sensor Coordinate Frame is NED, ROS2 robot localization expects ENU [https://support.nortekgroup.com/hc/en-us/article_attachments/17223428270620]
frame_id = "dvl_link"
namespace = "rhody/nav/sensors/nortek_dvl"
output_dir = "nortek_dvl_bag"

# NOTE: Conservative estimates for covariances of Bottom Track based off of manual (squared std dev)
vx_std = 0.006  # Horizontal (X, Y): ±0.3% of velocity ±0.003 m/s
vy_std = 0.006  
vz_std = 0.02   # Vertical (Z): ±1% of velocity ±0.01 m/s

# NOTE: Invalid bottom track velocities are defined as this value
invalid_vel = -32.76

# NOTE: Gyro X, Y, Z values are reported in radians per second (°/s) and ROS2 REP 103 recommends radians per second (rad/s)
deg2rad = np.pi / 180.0

def make_ros_time(iso_time):
    dt = pd.to_datetime(iso_time)
    sec = int(dt.timestamp())
    nanosec = int((dt.timestamp() - sec) * 1e9)
    t = Time()
    t.sec = sec
    t.nanosec = nanosec
    return t, sec * int(1e9) + nanosec

def ned_to_enu(quat_ned=None, vec_ned=None):
    """
    Convert NED to ENU.
    Args:
        quat_ned (tuple/list/np.array): (x, y, z, w) quaternion in NED.
        vec_ned (tuple/list/np.array): (x, y, z) vector in NED.
    Returns:
        tuple:
            - quat_enu (x, y, z, w) if quat_ned is provided
            - vec_enu (x, y, z) if vec_ned is provided
    """
    results = []

    if quat_ned is not None:
        q_ned = R.from_quat(quat_ned)
        ned_to_enu_rot = R.from_euler('x', np.pi) * R.from_euler('z', np.pi)
        q_enu = (ned_to_enu_rot * q_ned).as_quat()
        results.append(q_enu)

    if vec_ned is not None:
        # x stays the same, y and z flip
        vec_enu = np.array([vec_ned[1], vec_ned[0], -vec_ned[2]])  # NED (x=North, y=East, z=Down) → ENU
        vec_enu[0] = vec_ned[1]  # ENU.x = East = NED.y
        vec_enu[1] = vec_ned[0]  # ENU.y = North = NED.x
        vec_enu[2] = -vec_ned[2] # ENU.z = -Down = Up
        results.append(vec_enu)

    return tuple(results) if len(results) > 1 else results[0]

def main():
    rclpy.init()

    # Load the CSVs
    bottom_df = pd.read_csv("Bottom Track.csv", sep=";")
    imu_df = pd.read_csv("IMU.csv", sep=";")
    ins_df = pd.read_csv("INS.csv", sep=";")

    # Output bag
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # ⚠️ Deletes previous bag!
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=output_dir, storage_id="sqlite3"),
                ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))

    writer.create_topic(TopicMetadata(
        name=namespace + '/velocity',
        type='geometry_msgs/msg/TwistWithCovarianceStamped',
        serialization_format='cdr'
    ))

    writer.create_topic(TopicMetadata(
        name=namespace + '/imu',
        type='sensor_msgs/msg/Imu',
        serialization_format='cdr'
    ))

    writer.create_topic(TopicMetadata(
        name=namespace + '/imu_ahrs',
        type='sensor_msgs/msg/Imu',
        serialization_format='cdr'
    ))

    writer.create_topic(TopicMetadata(
        name=namespace + '/depth',
        type='geometry_msgs/msg/PoseWithCovarianceStamped',
        serialization_format='cdr'
    ))

    # writer.create_topic(TopicMetadata(
    #     name=namespace + '/pose',
    #     type='geometry_msgs/msg/PoseWithCovarianceStamped',
    #     serialization_format='cdr'
    # ))

    # ---- Bottom Track → TwistStamped ----
    for _, row in bottom_df.iterrows():
        if row['velocityX'] <= invalid_vel:  # Ignore invalid velocities
            continue
        t, ts = make_ros_time(row['dateTime'])
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = t
        msg.header.frame_id = frame_id
        vel_ned = [row['velocityX'], row['velocityY'], row['velocityZ']]
        vel_enu = ned_to_enu(vec_ned=vel_ned)
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = vel_enu

        # Add reasonable covariances (example: tight trust in x/y, less in z, don't trust angular)
        msg.twist.covariance = [
            vx_std**2, 0.0,      0.0,      0.0, 0.0, 0.0,
            0.0,      vy_std**2, 0.0,      0.0, 0.0, 0.0,
            0.0,      0.0,      vz_std**2,  0.0, 0.0, 0.0,
            0.0,      0.0,      0.0,     -1.0, 0.0, 0.0,
            0.0,      0.0,      0.0,      0.0, -1.0, 0.0,
            0.0,      0.0,      0.0,      0.0, 0.0, -1.0
        ]

        writer.write(namespace + '/velocity', serialize_message(msg), ts)

    # ---- IMU → Imu ---- ang vel and acc in NED, convert to ENU
    for _, row in imu_df.iterrows():
        t, ts = make_ros_time(row['dateTime'])
        msg = Imu()
        msg.header.stamp = t
        msg.header.frame_id = frame_id
        gyro_ned = [row['gyroX'], row['gyroY'], row['gyroZ']]  
        gyro_enu = ned_to_enu(vec_ned=gyro_ned) # Convert NED to ENU
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro_enu
        acc_ned = [row['accelerometerX'], row['accelerometerY'], row['accelerometerZ']]
        acc_enu = ned_to_enu(vec_ned=acc_ned)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = acc_enu
        msg.orientation_covariance[0] = -1.0  # unknown
        writer.write(namespace + '/imu', serialize_message(msg), ts)

    # ---- IMU AHRS → Imu ---- Convert quaternion from NED to ENU
    for _, row in ins_df.iterrows():
        t, ts = make_ros_time(row['dateTime'])
        msg = Imu()
        msg.header.stamp = t
        msg.header.frame_id = frame_id

        # Extract quaternion in NED frame
        q_ned = [
            float(row['ahrsDataQuaternionX']),
            float(row['ahrsDataQuaternionY']),
            float(row['ahrsDataQuaternionZ']),
            float(row['ahrsDataQuaternionW'])
        ]

        # Convert to ENU
        q_enu = ned_to_enu(quat_ned=q_ned)
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q_enu

        # Set unknown covariances for angular vel and acc
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0

        writer.write(namespace + '/imu_ahrs', serialize_message(msg), ts)

    # ---- INS → PoseWithCovarianceStamped DEPTH----
    for _, row in ins_df.iterrows():
        t, ts = make_ros_time(row['dateTime'])
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = t
        msg.header.frame_id = frame_id

        # Only set Z = depth (positive down), flip to negative up for ENU
        msg.pose.pose.position.z = -float(row['depth'])  # ENU Z = -NED Depth

        # Set unknown values for x, y, and orientation
        msg.pose.covariance = [0.0] * 36
        msg.pose.covariance[14] = 0.25  # z variance (0.5m std dev as example)
        msg.pose.covariance[0] = -1.0   # x unknown
        msg.pose.covariance[7] = -1.0   # y unknown
        msg.pose.covariance[21] = -1.0  # roll unknown
        msg.pose.covariance[28] = -1.0  # pitch unknown
        msg.pose.covariance[35] = -1.0  # yaw unknown

        writer.write(namespace + '/depth', serialize_message(msg), ts)


    # # ---- INS → PoseWithCovarianceStamped ----
    # for _, row in ins_df.iterrows():
    #     t, ts = make_ros_time(row['dateTime'])
    #     msg = PoseWithCovarianceStamped()
    #     msg.header.stamp = t
    #     msg.header.frame_id = frame_id
    #     msg.pose.pose.position.x = float(row['latitude'])  # WARN: Lat and Lon need to be swapped for UTM to use this Pose msg
    #     msg.pose.pose.position.y = float(row['longitude'])
    #     msg.pose.pose.position.z = float(row['altitude'])
    #     q_ned = [
    #         float(row['ahrsDataQuaternionX']),
    #         float(row['ahrsDataQuaternionY']),
    #         float(row['ahrsDataQuaternionZ']),
    #         float(row['ahrsDataQuaternionW'])
    #     ]
    #     q_enu = ned_to_enu(quat_ned=q_ned)
    #     msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = q_enu
    #     msg.pose.covariance = [0.5] * 36  # placeholder
    #     writer.write(namespace + '/pose', serialize_message(msg), ts)

    print("✅ ROS 2 bag created at:", output_dir)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
