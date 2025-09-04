from csv import writer
import os
import shutil
import pandas as pd
import numpy as np
import rclpy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped, Vector3Stamped
from sensor_msgs.msg import Imu, MagneticField
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from scipy.spatial.transform import Rotation as R
#from tf_transformations import euler_from_quaternion, quaternion_from_euler 

"""
nortek2ros.py

This script processes raw Nortek Nucleus DVL logs (Bottom Track, IMU, and INS data) 
and converts them into ROS 2 bag files for use with robot localization systems. 
The Nortek sensor data, originally in the NED (North-East-Down) coordinate frame, 
is transformed into the ENU (East-North-Up) frame as required by ROS 2 conventions.
"""

# NOTE: Nortek Sensor Coordinate Frame is NED, ROS2 robot localization expects ENU [https://support.nortekgroup.com/hc/en-us/article_attachments/17223428270620]
frame_id = "dvl_link"
namespace = "rhody/nav/sensors/nortek_dvl"
output_dir = "nortek_dvl_bag"
start_time  = pd.to_datetime("2025-05-28 12:38:12-04:00")  # EDT is UTC-4
end_time    = pd.to_datetime("2025-05-28 13:38:12-04:00")

# NOTE: Conservative estimates for covariances of Bottom Track based off of manual (squared std dev)
vx_std = 0.006  # Horizontal (X, Y): ±0.3% of velocity ±0.003 m/s
vy_std = 0.006  
vz_std = 0.02   # Vertical (Z): ±1% of velocity ±0.01 m/s

# NOTE: Invalid bottom track velocities are defined as this value
invalid_vel = -32.76

# NOTE: Gyro X, Y, Z values are reported in radians per second (°/s) and ROS2 REP 103 recommends radians per second (rad/s)
deg2rad = np.pi / 180.0

# NOTE: Magnetometer.csv is in Gauss, set to 1e-4; if it's Tesla, leave 1.0
MAG_UNIT_SCALE = 1e-4  # <-- change to 1.0 if your values are already in Tesla

# ---- Covariance estimates based on Nortek Nucleus specs ----
ahrs_roll_pitch_std = np.deg2rad(0.3)      # ~0.3 deg → radians
ahrs_yaw_std       = np.deg2rad(1.0)      # ~1 deg → radians

imu_ang_vel_std    = 0.01                 # rad/s
imu_lin_acc_std    = 0.1                  # m/s^2

# Precompute covariance matrices
ahrs_orientation_cov = [
    ahrs_roll_pitch_std**2, 0.0, 0.0,
    0.0, ahrs_roll_pitch_std**2, 0.0,
    0.0, 0.0, ahrs_yaw_std**2
]

imu_ang_vel_cov = [
    imu_ang_vel_std**2, 0.0, 0.0,
    0.0, imu_ang_vel_std**2, 0.0,
    0.0, 0.0, imu_ang_vel_std**2
]

imu_lin_acc_cov = [
    imu_lin_acc_std**2, 0.0, 0.0,
    0.0, imu_lin_acc_std**2, 0.0,
    0.0, 0.0, imu_lin_acc_std**2
]

def make_ros_time(iso_time):
    dt = pd.to_datetime(iso_time)
    sec = int(dt.timestamp())
    nanosec = int((dt.timestamp() - sec) * 1e9)
    t = Time()
    t.sec = sec
    t.nanosec = nanosec
    return t, sec * int(1e9) + nanosec

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


def main():
    rclpy.init()

    # Load and filter the CSVs
    bottom_df = pd.read_csv("Bottom Track.csv", sep=";", parse_dates=["dateTime"])
    bottom_df = bottom_df[bottom_df["dateTime"] >= start_time]
    bottom_df = bottom_df[bottom_df["dateTime"] <= end_time]

    imu_df = pd.read_csv("IMU.csv", sep=";", parse_dates=["dateTime"])
    imu_df = imu_df[imu_df["dateTime"] >= start_time]
    imu_df = imu_df[imu_df["dateTime"] <= end_time]

    ins_df = pd.read_csv("INS.csv", sep=";", parse_dates=["dateTime"])
    ins_df = ins_df[ins_df["dateTime"] >= start_time]
    ins_df = ins_df[ins_df["dateTime"] <= end_time]

    mag_df = pd.read_csv("Magnetometer.csv", sep=";", parse_dates=["dateTime"])
    mag_df = mag_df[mag_df["dateTime"] >= start_time]
    mag_df = mag_df[mag_df["dateTime"] <= end_time]

    print("✅ Loaded CSV files between cutoff:", start_time, "to", end_time)

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

    writer.create_topic(TopicMetadata(
        name=namespace + '/magnetic_field',
        type='sensor_msgs/msg/MagneticField',
        serialization_format='cdr'
    ))

    writer.create_topic(TopicMetadata(
        name=namespace + '/rpy_ned_deg',  # or rpy_ned_deg 
        type='geometry_msgs/msg/Vector3Stamped',
        serialization_format='cdr'
    ))

    # ---- Bottom Track → TwistStamped ----
    for _, row in bottom_df.iterrows():
        if row['velocityX'] <= invalid_vel:  # Ignore invalid velocities
            continue
        t, ts = make_ros_time(row['dateTime'])
        msg = TwistWithCovarianceStamped()
        msg.header.stamp = t
        msg.header.frame_id = frame_id
        vel_ned = [row['velocityX'], row['velocityY'], -row['velocityZ']]
        vel_enu = ned_vec_to_enu(vel_ned)
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = vel_ned

        # Add reasonable covariances (example: tight trust in x/y, less in z, don't trust angular so 1e6)
        msg.twist.covariance = [
            vx_std**2, 0.0,       0.0,       0.0, 0.0, 0.0,
            0.0,       vy_std**2, 0.0,       0.0, 0.0, 0.0,
            0.0,       0.0,       vz_std**2, 0.0, 0.0, 0.0,
            0.0,       0.0,       0.0,       1e6, 0.0, 0.0,
            0.0,       0.0,       0.0,       0.0, 1e6, 0.0,
            0.0,       0.0,       0.0,       0.0, 0.0, 1e6
        ]

        writer.write(namespace + '/velocity', serialize_message(msg), ts)

    # ---- IMU → Imu ---- ang vel and acc in NED, convert to ENU
    for _, row in imu_df.iterrows():
        t, ts = make_ros_time(row['dateTime'])
        msg = Imu()
        msg.header.stamp = t
        msg.header.frame_id = frame_id
        gyro_ned = [row['gyroX'], row['gyroY'], -row['gyroZ']]  
        gyro_enu = ned_vec_to_enu(gyro_ned) # Convert NED to ENU
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro_ned
        acc_ned = [row['accelerometerX'], row['accelerometerY'], -row['accelerometerZ']]
        acc_enu = ned_vec_to_enu(acc_ned)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = acc_ned
        msg.orientation_covariance[0] = -1.0  # unknown
        msg.angular_velocity_covariance = imu_ang_vel_cov
        msg.linear_acceleration_covariance = imu_lin_acc_cov
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

        # Euler angles from AHRS
        roll_deg   = float(row['ahrsDataRoll'])              # radians (if in deg, convert)
        pitch_deg  = float(row['ahrsDataPitch'])
        heading_deg = float(row['ahrsDataHeading'])      # Nortek AHRS heading is degrees
        roll_rad = np.deg2rad(roll_deg)
        pitch_rad = np.deg2rad(pitch_deg)
        heading_rad = np.deg2rad(heading_deg)
        rpy_ned = Vector3Stamped()
        rpy_ned.header.stamp = t
        rpy_ned.header.frame_id = frame_id
        rpy_ned.vector.x = roll_deg
        rpy_ned.vector.y = pitch_deg
        rpy_ned.vector.z = heading_deg
        writer.write(namespace + '/rpy_ned_deg', serialize_message(rpy_ned), ts)

        # Convert to ENU
        x_enu = pitch_deg
        y_enu = roll_deg
        z_enu = (90 - heading_deg) % 360
        q_enu = R.from_euler('xyz', [roll_deg, -pitch_deg, heading_deg], degrees=True).as_quat()

        # q_enu = ned_quat_to_enu(q_ned)
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q_enu

        # Set unknown covariances for angular vel and acc
        msg.orientation_covariance = ahrs_orientation_cov
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0

        writer.write(namespace + '/imu_ahrs', serialize_message(msg), ts)

    # ---- INS → PoseWithCovarianceStamped DEPTH----
    for _, row in ins_df.iterrows():
        t, ts = make_ros_time(row['dateTime'])
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = t
        msg.header.frame_id = "odom"

        # Only set Z = depth (positive down), flip to negative up for ENU
        msg.pose.pose.position.z = -float(row['depth'])  # ENU Z = -NED Depth
        # Not fusing orientation here → use identity quaternion
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        # Set unknown values for x, y, and orientation
        msg.pose.covariance = [0.0] * 36
        msg.pose.covariance[14] = 0.25  # z variance (0.5m std dev as example)
        msg.pose.covariance[0] = 1.0    # x unused
        msg.pose.covariance[7] = 1.0    # y unused
        msg.pose.covariance[21] = 1.0   # roll unused
        msg.pose.covariance[28] = 1.0   # pitch unused
        msg.pose.covariance[35] = 1.0   # yaw unused

        writer.write(namespace + '/depth', serialize_message(msg), ts)

    # ---- Magnetometer → MagneticField ----
    # CSV columns: dateTime, magnetometerX, magnetometerY, magnetometerZ
    # NED [N,E,D] -> ENU [E,N,U] and convert units to Tesla if needed
    mag_var = (30e-9)**2  # ~30 nT std dev (example); adjust if you have specs
    mag_cov = [mag_var, 0.0, 0.0,
            0.0, mag_var, 0.0,
            0.0, 0.0, mag_var]

    for _, row in mag_df.iterrows():
        t, ts = make_ros_time(row['dateTime'])

        # raw in NED:
        mag_ned = np.array([
            float(row['magnetometerX']),
            float(row['magnetometerY']),
            -float(row['magnetometerZ'])
        ], dtype=float)

        # NED -> ENU
        mag_enu = ned_vec_to_enu(mag_ned)

        # Apply unit scale to Tesla
        mag_enu *= MAG_UNIT_SCALE

        msg = MagneticField()
        msg.header.stamp = t
        msg.header.frame_id = frame_id
        msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z = mag_ned
        msg.magnetic_field_covariance = mag_cov  # set to zeros if unknown

        writer.write(namespace + '/magnetic_field', serialize_message(msg), ts)

    print("✅ ROS 2 bag created at:", output_dir)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
