#!/usr/bin/env python3
"""
bag2csv.py

Convert ROS2 bag topics to CSV files for data analysis.
Supports common message types with automatic field extraction.

Usage:
    python3 bag2csv.py --bag input.db3 --topic /sensor/imu --output imu_data.csv
    python3 bag2csv.py --bag input.db3 --topic /gps/fix --config config.yaml
    python3 bag2csv.py --bag input.db3 --list-topics  # Show available topics
"""

import os
import sys
import argparse
import csv
import yaml
from pathlib import Path
from datetime import datetime
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from builtin_interfaces.msg import Time

# Supported message type handlers
MESSAGE_HANDLERS = {}

def register_handler(msg_type):
    """Decorator to register message type handlers."""
    def decorator(func):
        MESSAGE_HANDLERS[msg_type] = func
        return func
    return decorator

def ros_time_to_timestamp(ros_time):
    """Convert ROS Time to Unix timestamp."""
    return ros_time.sec + ros_time.nanosec * 1e-9

def ros_time_to_iso(ros_time):
    """Convert ROS Time to ISO format string."""
    timestamp = ros_time_to_timestamp(ros_time)
    return datetime.fromtimestamp(timestamp).isoformat()

def flatten_message(msg, prefix="", max_depth=3, current_depth=0):
    """
    Recursively flatten a ROS message into a dictionary.
    Handles nested messages up to max_depth to avoid infinite recursion.
    """
    result = {}
    
    if current_depth >= max_depth:
        result[prefix.rstrip('.')] = str(msg)
        return result
    
    # Handle common built-in types
    if hasattr(msg, '__slots__'):
        for slot in msg.__slots__:
            if hasattr(msg, slot):
                value = getattr(msg, slot)
                key = f"{prefix}{slot}" if prefix else slot
                
                # Handle nested messages
                if hasattr(value, '__slots__'):
                    nested = flatten_message(value, f"{key}.", max_depth, current_depth + 1)
                    result.update(nested)
                # Handle lists/arrays
                elif isinstance(value, (list, tuple)):
                    if len(value) > 0:
                        # For numeric arrays, create indexed columns
                        if isinstance(value[0], (int, float)):
                            for i, item in enumerate(value):
                                result[f"{key}[{i}]"] = item
                        # For arrays of messages, flatten first few elements
                        elif hasattr(value[0], '__slots__'):
                            for i, item in enumerate(value[:5]):  # Limit to first 5 elements
                                nested = flatten_message(item, f"{key}[{i}].", max_depth, current_depth + 1)
                                result.update(nested)
                        else:
                            result[f"{key}_length"] = len(value)
                            result[f"{key}_str"] = str(value)
                    else:
                        result[f"{key}_length"] = 0
                # Handle primitive types
                else:
                    result[key] = value
    else:
        # Fallback for non-slotted objects
        result[prefix.rstrip('.')] = str(msg)
    
    return result

@register_handler('sensor_msgs/msg/Imu')
def handle_imu(msg, timestamp):
    """Handle IMU messages."""
    return {
        'timestamp': timestamp,
        'time_iso': ros_time_to_iso(msg.header.stamp),
        'frame_id': msg.header.frame_id,
        'orientation_x': msg.orientation.x,
        'orientation_y': msg.orientation.y,
        'orientation_z': msg.orientation.z,
        'orientation_w': msg.orientation.w,
        'angular_velocity_x': msg.angular_velocity.x,
        'angular_velocity_y': msg.angular_velocity.y,
        'angular_velocity_z': msg.angular_velocity.z,
        'linear_acceleration_x': msg.linear_acceleration.x,
        'linear_acceleration_y': msg.linear_acceleration.y,
        'linear_acceleration_z': msg.linear_acceleration.z,
    }

@register_handler('geometry_msgs/msg/TwistWithCovarianceStamped')
def handle_twist_with_covariance(msg, timestamp):
    """Handle TwistWithCovarianceStamped messages."""
    return {
        'timestamp': timestamp,
        'time_iso': ros_time_to_iso(msg.header.stamp),
        'frame_id': msg.header.frame_id,
        'linear_x': msg.twist.twist.linear.x,
        'linear_y': msg.twist.twist.linear.y,
        'linear_z': msg.twist.twist.linear.z,
        'angular_x': msg.twist.twist.angular.x,
        'angular_y': msg.twist.twist.angular.y,
        'angular_z': msg.twist.twist.angular.z,
    }

@register_handler('geometry_msgs/msg/PoseWithCovarianceStamped')
def handle_pose_with_covariance(msg, timestamp):
    """Handle PoseWithCovarianceStamped messages."""
    return {
        'timestamp': timestamp,
        'time_iso': ros_time_to_iso(msg.header.stamp),
        'frame_id': msg.header.frame_id,
        'position_x': msg.pose.pose.position.x,
        'position_y': msg.pose.pose.position.y,
        'position_z': msg.pose.pose.position.z,
        'orientation_x': msg.pose.pose.orientation.x,
        'orientation_y': msg.pose.pose.orientation.y,
        'orientation_z': msg.pose.pose.orientation.z,
        'orientation_w': msg.pose.pose.orientation.w,
    }

@register_handler('nav_msgs/msg/Odometry')
def handle_odometry(msg, timestamp):
    """Handle Odometry messages."""
    return {
        'timestamp': timestamp,
        'time_iso': ros_time_to_iso(msg.header.stamp),
        'frame_id': msg.header.frame_id,
        'child_frame_id': msg.child_frame_id,
        # Position
        'position_x': msg.pose.pose.position.x,
        'position_y': msg.pose.pose.position.y,
        'position_z': msg.pose.pose.position.z,
        # Orientation
        'orientation_x': msg.pose.pose.orientation.x,
        'orientation_y': msg.pose.pose.orientation.y,
        'orientation_z': msg.pose.pose.orientation.z,
        'orientation_w': msg.pose.pose.orientation.w,
        # Linear velocity
        'linear_velocity_x': msg.twist.twist.linear.x,
        'linear_velocity_y': msg.twist.twist.linear.y,
        'linear_velocity_z': msg.twist.twist.linear.z,
        # Angular velocity
        'angular_velocity_x': msg.twist.twist.angular.x,
        'angular_velocity_y': msg.twist.twist.angular.y,
        'angular_velocity_z': msg.twist.twist.angular.z,
        # Pose covariance (6x6 matrix flattened)
        'pose_cov_xx': msg.pose.covariance[0],
        'pose_cov_xy': msg.pose.covariance[1],
        'pose_cov_xz': msg.pose.covariance[2],
        'pose_cov_xroll': msg.pose.covariance[3],
        'pose_cov_xpitch': msg.pose.covariance[4],
        'pose_cov_xyaw': msg.pose.covariance[5],
        'pose_cov_yy': msg.pose.covariance[7],
        'pose_cov_yz': msg.pose.covariance[8],
        'pose_cov_yroll': msg.pose.covariance[9],
        'pose_cov_ypitch': msg.pose.covariance[10],
        'pose_cov_yyaw': msg.pose.covariance[11],
        'pose_cov_zz': msg.pose.covariance[14],
        'pose_cov_zroll': msg.pose.covariance[15],
        'pose_cov_zpitch': msg.pose.covariance[16],
        'pose_cov_zyaw': msg.pose.covariance[17],
        # Twist covariance (6x6 matrix - key diagonal elements)
        'twist_cov_vx': msg.twist.covariance[0],
        'twist_cov_vy': msg.twist.covariance[7],
        'twist_cov_vz': msg.twist.covariance[14],
        'twist_cov_wx': msg.twist.covariance[21],
        'twist_cov_wy': msg.twist.covariance[28],
        'twist_cov_wz': msg.twist.covariance[35],
    }

@register_handler('sensor_msgs/msg/NavSatFix')
def handle_navsatfix(msg, timestamp):
    """Handle GPS NavSatFix messages."""
    return {
        'timestamp': timestamp,
        'time_iso': ros_time_to_iso(msg.header.stamp),
        'frame_id': msg.header.frame_id,
        'latitude': msg.latitude,
        'longitude': msg.longitude,
        'altitude': msg.altitude,
        'status': msg.status.status,
        'service': msg.status.service,
        'position_covariance_type': msg.position_covariance_type,
    }

@register_handler('sensor_msgs/msg/MagneticField')
def handle_magnetic_field(msg, timestamp):
    """Handle MagneticField messages."""
    return {
        'timestamp': timestamp,
        'time_iso': ros_time_to_iso(msg.header.stamp),
        'frame_id': msg.header.frame_id,
        'magnetic_field_x': msg.magnetic_field.x,
        'magnetic_field_y': msg.magnetic_field.y,
        'magnetic_field_z': msg.magnetic_field.z,
    }

def generic_message_handler(msg, timestamp):
    """Generic handler for any message type using flattening."""
    result = {'timestamp': timestamp}
    
    # Add header timestamp if available
    if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
        result['time_iso'] = ros_time_to_iso(msg.header.stamp)
        result['frame_id'] = getattr(msg.header, 'frame_id', '')
    
    # Flatten the message
    flattened = flatten_message(msg)
    result.update(flattened)
    
    return result

def load_config_from_yaml(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from: {config_file}")
        return config
    except Exception as e:
        print(f"❌ Error loading config file {config_file}: {e}")
        return None

def list_bag_topics(bag_path):
    """List all topics in the bag file."""
    reader = SequentialReader()
    try:
        reader.open(
            StorageOptions(uri=bag_path, storage_id='sqlite3'),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        )
    except Exception as e:
        print(f"❌ Could not open bag {bag_path}: {e}")
        return []
    
    topics = reader.get_all_topics_and_types()
    print(f"\n📋 Topics in {bag_path}:")
    print(f"{'Topic Name':<50} {'Message Type':<40} {'Count'}")
    print("-" * 100)
    
    topic_counts = {}
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
    
    for topic in topics:
        count = topic_counts.get(topic.name, 0)
        print(f"{topic.name:<50} {topic.type:<40} {count}")
    
    return topics

def bag_to_csv(bag_path, topic_name, output_csv, start_time=None, end_time=None):
    """
    Convert a specific topic from a ROS2 bag to CSV.
    
    Args:
        bag_path: Path to the bag file
        topic_name: Name of the topic to extract
        output_csv: Output CSV file path
        start_time: Start time filter (datetime object)
        end_time: End time filter (datetime object)
    """
    reader = SequentialReader()
    
    try:
        reader.open(
            StorageOptions(uri=bag_path, storage_id='sqlite3'),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        )
    except Exception as e:
        print(f"❌ Could not open bag {bag_path}: {e}")
        return False
    
    # Find the topic
    topics = reader.get_all_topics_and_types()
    topic_info = None
    for topic in topics:
        if topic.name == topic_name:
            topic_info = topic
            break
    
    if not topic_info:
        print(f"❌ Topic '{topic_name}' not found in bag file")
        print("Available topics:")
        for topic in topics:
            print(f"  - {topic.name}")
        return False
    
    print(f"🔄 Converting topic '{topic_name}' ({topic_info.type}) to CSV...")
    
    # Get message class
    try:
        msg_class = get_message(topic_info.type)
    except Exception as e:
        print(f"❌ Could not get message class for {topic_info.type}: {e}")
        return False
    
    # Get handler for this message type
    handler = MESSAGE_HANDLERS.get(topic_info.type, generic_message_handler)
    
    # Convert time filters to nanoseconds
    start_timestamp_ns = int(start_time.timestamp() * 1e9) if start_time else None
    end_timestamp_ns = int(end_time.timestamp() * 1e9) if end_time else None
    
    # Process messages
    rows = []
    message_count = 0
    filtered_count = 0
    
    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        
        if topic != topic_name:
            continue
        
        message_count += 1
        
        # Apply time filtering
        if start_timestamp_ns and timestamp_ns < start_timestamp_ns:
            filtered_count += 1
            continue
        if end_timestamp_ns and timestamp_ns > end_timestamp_ns:
            filtered_count += 1
            continue
        
        try:
            # Deserialize message
            msg = deserialize_message(data, msg_class)
            
            # Convert timestamp to seconds
            timestamp = timestamp_ns * 1e-9
            
            # Extract data using handler
            row_data = handler(msg, timestamp)
            rows.append(row_data)
            
        except Exception as e:
            print(f"⚠️  Error processing message {message_count}: {e}")
            continue
    
    if not rows:
        print(f"❌ No data extracted from topic '{topic_name}'")
        return False
    
    # Write to CSV
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ CSV file created: {output_csv}")
        print(f"📊 Summary:")
        print(f"   Messages processed: {len(rows)}")
        print(f"   Total messages in topic: {message_count}")
        if filtered_count > 0:
            print(f"   Messages filtered: {filtered_count}")
        print(f"   Fields extracted: {len(fieldnames)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error writing CSV file: {e}")
        return False

def parse_time_string(time_str):
    """Parse time string to datetime object."""
    if not time_str:
        return None
    
    formats = [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    try:
        import pandas as pd
        return pd.to_datetime(time_str)
    except:
        pass
    
    raise ValueError(f"Could not parse time string: {time_str}")

def main():
    parser = argparse.ArgumentParser(description='Convert ROS2 bag topics to CSV files')
    parser.add_argument('--bag', '-b', type=str, required=True,
                       help='Path to ROS2 bag file (.db3)')
    parser.add_argument('--topic', '-t', type=str,
                       help='Topic name to extract')
    parser.add_argument('--output', '-o', type=str,
                       help='Output CSV file path (default: <topic_name>.csv)')
    parser.add_argument('--start-time', '-s', type=str,
                       help='Start time for filtering')
    parser.add_argument('--end-time', '-e', type=str,
                       help='End time for filtering')
    parser.add_argument('--config', '-c', type=str,
                       help='YAML configuration file')
    parser.add_argument('--list-topics', '-l', action='store_true',
                       help='List all topics in the bag file')
    
    args = parser.parse_args()
    
    # Validate bag file
    if not os.path.exists(args.bag):
        print(f"❌ Bag file not found: {args.bag}")
        return 1
    
    # List topics if requested
    if args.list_topics:
        list_bag_topics(args.bag)
        return 0
    
    # Load config if provided
    config = {}
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        if yaml_config is None:
            return 1
        config = yaml_config
    
    # Get parameters (command line overrides YAML)
    topic = args.topic or config.get('topic')
    output = args.output or config.get('output')
    start_time_str = args.start_time or config.get('start_time')
    end_time_str = args.end_time or config.get('end_time')
    
    if not topic:
        print("❌ Topic name is required. Use --topic or specify in config file.")
        print("Use --list-topics to see available topics.")
        return 1
    
    # Default output filename
    if not output:
        topic_name = topic.replace('/', '_').lstrip('_')
        output = f"{topic_name}.csv"
    
    # Parse time filters
    start_time = None
    end_time = None
    
    if start_time_str:
        try:
            start_time = parse_time_string(start_time_str)
        except ValueError as e:
            print(f"❌ Error parsing start time: {e}")
            return 1
    
    if end_time_str:
        try:
            end_time = parse_time_string(end_time_str)
        except ValueError as e:
            print(f"❌ Error parsing end time: {e}")
            return 1
    
    # Convert bag to CSV
    success = bag_to_csv(args.bag, topic, output, start_time, end_time)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())