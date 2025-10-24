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
"""

# NOTE: Subsonus Sensor Coordinate Frame is NED, ROS2 robot localization expects ENU [https://docs.advancednavigation.com/subsonus/SensorCoordinate.htm]
frame_id = "usbl_link"
namespace = "rhody/nav/sensors/subsonus_usbl"
output_dir = "subsonus_bag"
fn_state = "RemoteSubsonusState_2.csv"
fn_track = "RemoteTrack_2.csv"
fn_raw = "RemoteRawSensors_2.csv"

# NED -> ENU (use Rotation so vectors and quats stay consistent)
P_NED_to_ENU = R.from_matrix([[0, 1, 0],
                              [1, 0, 0],
                              [0, 0,-1]])

def ned_vec_to_enu(v):
    """Map NED vector to ENU: [E,N,U] = [Y, X, -Z]."""
    return P_NED_to_ENU.apply(np.asarray(v, float)).astype(float)

def ned_quat_to_enu(q_ned_xyzw):
    """Map a quaternion (xyzw) that orients body w.r.t. NED into ENU."""
    return (P_NED_to_ENU * R.from_quat(q_ned_xyzw)).as_quat()

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
        name=namespace + '/depth',
        type='geometry_msgs/msg/PoseWithCovarianceStamped',
        serialization_format='cdr'
    ))

    writer.create_topic(TopicMetadata(
        name=namespace + '/ins/fix',
        type='sensor_msgs/msg/NavSatFix',
        serialization_format='cdr'
    ))
    writer.create_topic(TopicMetadata(
        name=namespace + '/ins/imu',
        type='sensor_msgs/msg/Imu',
        serialization_format='cdr'
    ))
    writer.create_topic(TopicMetadata(
        name=namespace + '/ins/velocity',
        type='geometry_msgs/msg/TwistWithCovarianceStamped',
        serialization_format='cdr'
    ))

    # ---- NavSatFix and Depth from Remote Track data (accomms fix)
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

        # ---- Depth as PoseWithCovarianceStamped
        depth_msg = PoseWithCovarianceStamped()
        depth_msg.header.stamp = ros_time
        depth_msg.header.frame_id = "utm_local"
        depth_msg.pose.pose.position.z = -row["Remote Depth"]  # Depth is positive down, so negate for ENU

        # Set unknown values for x, y, and orientation
        depth_msg.pose.covariance = [0.0] * 36
        depth_msg.pose.covariance[14] = 1.5  # z variance (manual advertises 1.5 m accuracy))
        depth_msg.pose.covariance[0] = 1.0    # x unused
        depth_msg.pose.covariance[7] = 1.0    # y unused
        depth_msg.pose.covariance[21] = 1.0   # roll unused
        depth_msg.pose.covariance[28] = 1.0   # pitch unused
        depth_msg.pose.covariance[35] = 1.0   # yaw unused

        writer.write(namespace + '/depth', serialize_message(depth_msg), timestamp)

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

        writer.write(namespace + '/ins/fix', serialize_message(fix), timestamp)

        # ---- IMU
        imu = Imu()
        imu.header.stamp = ros_time
        imu.header.frame_id = frame_id
        # q_ned = R.from_euler('xyz', [row["Roll"], row["Pitch"], row["Heading"]], degrees=True).as_quat()
        # q_enu = ned_quat_to_enu(q_ned)
        
        # Convert to ENU
        x_enu = row["Roll"]
        y_enu = -row["Pitch"]
        z_enu = (90 - row["Heading"]) % 360
        q_enu = R.from_euler('xyz', [x_enu, y_enu, z_enu], degrees=True).as_quat()
        
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = q_enu

        try:
            roll_sd_rad    = np.deg2rad(float(row["Roll Error"]))
            pitch_sd_rad   = np.deg2rad(float(row["Pitch Error"]))
            heading_sd_rad = np.deg2rad(float(row["Heading Error"]))
            imu.orientation_covariance = [
                roll_sd_rad**2,   0.0,              0.0,
                0.0,              pitch_sd_rad**2,  0.0,
                0.0,              0.0,              heading_sd_rad**2
            ]
        except Exception as e:
            print(f"⚠️ IMU covariance formatting error:\n{row}\n{e}")
            continue

        angvel_ned = [row["Angular Velocity X"], row["Angular Velocity Y"], row["Angular Velocity Z"]]
        ang_vel_enu = [row["Angular Velocity X"], -row["Angular Velocity Y"], -row["Angular Velocity Z"]]
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = ang_vel_enu
        ang_sd = np.deg2rad(1.0)  # example placeholder: 1 deg/s SD
        imu.angular_velocity_covariance = [
            ang_sd**2, 0.0,       0.0,
            0.0,       ang_sd**2, 0.0,
            0.0,       0.0,       ang_sd**2
        ]
        imu.linear_acceleration_covariance = [
            -1.0, 0.0,  0.0,
            0.0, 0.0,  0.0,
            0.0, 0.0,  0.0
        ]

        writer.write(namespace + '/ins/imu', serialize_message(imu), timestamp)

        # ---- Velocity (TwistWithCovarianceStamped)
        twist = TwistWithCovarianceStamped()
        twist.header.stamp = ros_time
        twist.header.frame_id = frame_id

        lin_vel_ned = [row["Velocity North"], row["Velocity East"], row["Velocity Down"]]
        #lin_vel_enu = ned_vec_to_enu(lin_vel_ned)
        lin_vel_enu = [row["Velocity East"], row["Velocity North"], -row["Velocity Down"]]
        #ang_vel_enu = ned_vec_to_enu(angvel_ned)

        twist.twist.twist.linear.x, twist.twist.twist.linear.y, twist.twist.twist.linear.z = lin_vel_enu
        twist.twist.twist.angular.x, twist.twist.twist.angular.y, twist.twist.twist.angular.z = ang_vel_enu
        lin_sd = 0.05  # conservative guess 5 cm/s SD
        twist.twist.covariance = [
            float(lin_sd**2), 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, float(lin_sd**2), 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, float(lin_sd**2), 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, float(ang_sd**2), 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, float(ang_sd**2), 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, float(ang_sd**2),
        ]

        writer.write(namespace + '/ins/velocity', serialize_message(twist), timestamp)

    print(f"✅ ROS 2 bag created in: {output_dir}")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
