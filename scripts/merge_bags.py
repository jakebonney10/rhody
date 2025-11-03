import os
import shutil
import tempfile
import argparse
import glob
import yaml
from pathlib import Path
from datetime import datetime
import rclpy
from rclpy.serialization import deserialize_message, serialize_message
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from builtin_interfaces.msg import Time
from rosidl_runtime_py.utilities import get_message

# William Elgin
input_bags = [
    "/media/bonnaroo/Seagate/20250527_William_Elgin/DVL/Data0012_300414/nortek_dvl_bag/nortek_dvl_bag_0.db3",
    "/media/bonnaroo/Seagate/20250527_William_Elgin/Subsonous/ANPP_LOG_000214_2025_05_27_16_07_33/subsonus_bag/subsonus_bag_0.db3"
]

def load_config_from_yaml(config_file):
    """
    Load configuration from YAML file.
    Returns a dictionary with the configuration.
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from: {config_file}")
        return config
    except Exception as e:
        print(f"❌ Error loading config file {config_file}: {e}")
        return None

def parse_time_string(time_str):
    """
    Parse time string in various formats to datetime object.
    Supports ISO format, common date formats.
    """
    if not time_str:
        return None
    
    # Common time formats to try
    formats = [
        "%Y-%m-%d %H:%M:%S%z",      # 2025-05-29 17:47:33-04:00
        "%Y-%m-%d %H:%M:%S",        # 2025-05-29 17:47:33
        "%Y-%m-%dT%H:%M:%S%z",      # 2025-05-29T17:47:33-04:00
        "%Y-%m-%dT%H:%M:%S",        # 2025-05-29T17:47:33
        "%Y-%m-%d",                 # 2025-05-29
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    
    # If all else fails, try pandas (if available)
    try:
        import pandas as pd
        return pd.to_datetime(time_str)
    except:
        pass
    
    raise ValueError(f"Could not parse time string: {time_str}")

def ros_time_to_timestamp(ros_time):
    """Convert ROS Time message to nanosecond timestamp."""
    return ros_time.sec * int(1e9) + ros_time.nanosec

def timestamp_to_ros_time(timestamp_ns):
    """Convert nanosecond timestamp to ROS Time message."""
    sec = timestamp_ns // int(1e9)
    nanosec = timestamp_ns % int(1e9)
    ros_time = Time()
    ros_time.sec = int(sec)
    ros_time.nanosec = int(nanosec)
    return ros_time

def datetime_to_timestamp_ns(dt):
    """Convert datetime to nanosecond timestamp."""
    return int(dt.timestamp() * 1e9)

def find_bag_files(directory):
    """
    Find all ROS2 bag database files (.db3) in a directory and its subdirectories.
    Returns a sorted list of bag file paths.
    """
    bag_files = []
    directory = Path(directory).resolve()
    
    # Look for .db3 files recursively
    for db3_file in directory.rglob("*.db3"):
        bag_files.append(str(db3_file))
    
    # Sort by modification time to process in chronological order
    bag_files.sort(key=lambda x: os.path.getmtime(x))
    
    print(f"Found {len(bag_files)} bag files in {directory}:")
    for bag_file in bag_files:
        print(f"  - {bag_file}")
    
    return bag_files

def merge_rosbags(input_bag_paths, output_bag_path, start_time=None, end_time=None):
    """
    Merge multiple ROS2 bag files into a single bag file with optional time filtering.
    
    Args:
        input_bag_paths: List of input bag file paths
        output_bag_path: Output bag file path
        start_time: datetime object for start time filter (optional)
        end_time: datetime object for end time filter (optional)
    """
    rclpy.init()

    if os.path.exists(output_bag_path):
        shutil.rmtree(output_bag_path)

    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=output_bag_path, storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )

    seen_topics = {}
    
    # Convert datetime filters to nanosecond timestamps
    start_timestamp_ns = datetime_to_timestamp_ns(start_time) if start_time else None
    end_timestamp_ns = datetime_to_timestamp_ns(end_time) if end_time else None
    
    if start_time or end_time:
        print(f"🕒 Time filtering enabled:")
        if start_time:
            print(f"   Start: {start_time}")
        if end_time:
            print(f"   End: {end_time}")

    total_messages = 0
    filtered_messages = 0

    for bag_path in input_bag_paths:
        print(f"Processing bag: {bag_path}")
        reader = SequentialReader()
        
        try:
            reader.open(
                StorageOptions(uri=bag_path, storage_id='sqlite3'),
                ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
            )
        except Exception as e:
            print(f"Warning: Could not open bag {bag_path}: {e}")
            continue
            
        topics = reader.get_all_topics_and_types()

        for topic in topics:
            if topic.name not in seen_topics:
                seen_topics[topic.name] = topic
                writer.create_topic(TopicMetadata(
                    name=topic.name,
                    type=topic.type,
                    serialization_format='cdr'
                ))

        message_count = 0
        bag_filtered_count = 0
        
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            total_messages += 1
            
            # Apply time filtering
            if start_timestamp_ns and timestamp_ns < start_timestamp_ns:
                bag_filtered_count += 1
                continue
            if end_timestamp_ns and timestamp_ns > end_timestamp_ns:
                bag_filtered_count += 1
                continue
            
            writer.write(topic, data, timestamp_ns)
            message_count += 1
        
        filtered_messages += bag_filtered_count
        print(f"  - Processed {message_count} messages")
        if bag_filtered_count > 0:
            print(f"  - Filtered out {bag_filtered_count} messages (time range)")

    rclpy.shutdown()
    
    print(f"✅ Merged bag created at: {output_bag_path}")
    print(f"📊 Summary:")
    print(f"   Total messages read: {total_messages}")
    print(f"   Messages written: {total_messages - filtered_messages}")
    if filtered_messages > 0:
        print(f"   Messages filtered: {filtered_messages}")

def main():
    parser = argparse.ArgumentParser(description='Merge ROS2 bag files with optional time filtering')
    parser.add_argument('--directory', '-d', type=str, 
                       help='Directory to search for bag files (merges all .db3 files found). Defaults to current directory if no bags specified.')
    parser.add_argument('--bags', '-b', nargs='+', type=str,
                       help='Specific bag file paths to merge')
    parser.add_argument('--output', '-o', type=str, default='merged_bag',
                       help='Output bag file path (default: merged_bag in current directory)')
    parser.add_argument('--start-time', '-s', type=str,
                       help='Start time for filtering (e.g., "2025-05-29 17:47:33-04:00" or "2025-05-29T17:47:33")')
    parser.add_argument('--end-time', '-e', type=str,
                       help='End time for filtering (e.g., "2025-05-29 18:47:33-04:00" or "2025-05-29T18:47:33")')
    parser.add_argument('--config', '-c', type=str,
                       help='YAML configuration file with options')
    
    args = parser.parse_args()
    
    # Load configuration from YAML if provided
    config = {}
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        if yaml_config is None:
            return 1
        config = yaml_config
    
    # Command line arguments override YAML config
    directory = args.directory or config.get('directory')
    bags = args.bags or config.get('bags')
    output = args.output if args.output != 'merged_bag' else config.get('output', 'merged_bag')
    start_time_str = args.start_time or config.get('start_time')
    end_time_str = args.end_time or config.get('end_time')
    
    # Parse time strings
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
    
    # Validate time range
    if start_time and end_time and start_time >= end_time:
        print(f"❌ Error: Start time must be before end time")
        return 1
    
    # Make sure output path is absolute and create the full path
    output_path = os.path.abspath(output)
    
    if directory:
        if not os.path.isdir(directory):
            print(f"Error: Directory {directory} does not exist")
            return
        input_bags = find_bag_files(directory)
        if not input_bags:
            print(f"No bag files found in {directory}")
            return
    elif bags:
        input_bags = bags
        # Verify all input bags exist
        for bag in input_bags:
            if not os.path.exists(bag):
                print(f"Error: Bag file {bag} does not exist")
                return
    else:
        # Default to current directory if no directory or bags specified
        current_dir = "."
        print(f"No directory or bags specified, defaulting to current directory: {os.path.abspath(current_dir)}")
        input_bags = find_bag_files(current_dir)
        if not input_bags:
            print(f"No bag files found in current directory")
            print("Use --directory or --bags to specify input files")
            return
    
    print(f"Output will be saved to: {output_path}")
    merge_rosbags(input_bags, output_path, start_time, end_time)

if __name__ == "__main__":
    main()

# Legacy hardcoded examples (uncomment and modify as needed):