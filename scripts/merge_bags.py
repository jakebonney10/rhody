import os
import shutil
import tempfile
import rclpy
from rclpy.serialization import deserialize_message, serialize_message
from rosbag2_py import SequentialReader, SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from builtin_interfaces.msg import Time
from rosidl_runtime_py.utilities import get_message

# Unknown dive 5
# input_bags = [
#     "/home/bonnaroo/Desktop/20250528_Unknown_005_NAV_Data/SUBSONUS/ANPP_LOG_000231_2025_05_28_16_38_12/subsonus_bag/subsonus_bag_0.db3",
#     "/home/bonnaroo/Desktop/20250528_Unknown_005_NAV_Data/SUBSONUS/ANPP_LOG_000233_2025_05_28_18_57_05/subsonus_bag/subsonus_bag_0.db3",
#     "/home/bonnaroo/Desktop/20250528_Unknown_005_NAV_Data/DVL/Data0014_300414/nortek_dvl_bag/nortek_dvl_bag_0.db3"
# ]

# Adiramled
# input_bags = [
#     "/media/bonnaroo/Seagate/20250529_Adiramled/DVL/Data0017_300414/nortek_dvl_bag/nortek_dvl_bag_0.db3",
#     "/media/bonnaroo/Seagate/20250529_Adiramled/DVL/Data0017_300414_2/nortek_dvl_bag/nortek_dvl_bag_0.db3",
#     "/media/bonnaroo/Seagate/20250529_Adiramled/Subsonus/ANPP_LOG_000112_2025_05_29_12_52_07/subsonus_bag/subsonus_bag_0.db3",
#     "/media/bonnaroo/Seagate/20250529_Adiramled/Subsonus/ANPP_LOG_000113_2025_05_29_15_09_45/subsonus_bag/subsonus_bag_0.db3",
#     "/media/bonnaroo/Seagate/20250529_Adiramled/Subsonus/ANPP_LOG_000114_2025_05_29_17_23_08/subsonus_bag/subsonus_bag_0.db3",
#     "/media/bonnaroo/Seagate/20250529_Adiramled/Subsonus/ANPP_LOG_000115_2025_05_29_19_36_18/subsonus_bag/subsonus_bag_0.db3",
#     "/media/bonnaroo/Seagate/20250529_Adiramled/Subsonus/ANPP_LOG_000116_2025_05_29_21_49_02/subsonus_bag/subsonus_bag_0.db3"
# ]

# William Elgin
input_bags = [
    "/media/bonnaroo/Seagate/20250527_William_Elgin/DVL/Data0012_300414/nortek_dvl_bag/nortek_dvl_bag_0.db3",
    "/media/bonnaroo/Seagate/20250527_William_Elgin/Subsonous/ANPP_LOG_000214_2025_05_27_16_07_33/subsonus_bag/subsonus_bag_0.db3"
]


output_bag = "/home/bonnaroo/Desktop/merged_bag_20250527_william_elgin.db3"

def merge_rosbags(input_bag_paths, output_bag_path):
    rclpy.init()

    if os.path.exists(output_bag_path):
        shutil.rmtree(output_bag_path)

    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=output_bag_path, storage_id='sqlite3'),
        ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    )

    seen_topics = {}

    for bag_path in input_bag_paths:
        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=bag_path, storage_id='sqlite3'),
            ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        )
        topics = reader.get_all_topics_and_types()

        for topic in topics:
            if topic.name not in seen_topics:
                seen_topics[topic.name] = topic
                writer.create_topic(TopicMetadata(
                    name=topic.name,
                    type=topic.type,
                    serialization_format='cdr'
                ))

        while reader.has_next():
            topic, data, t = reader.read_next()
            writer.write(topic, data, t)

    rclpy.shutdown()
    print(f"✅ Merged bag created at: {output_bag_path}")

merge_rosbags(input_bags, output_bag)