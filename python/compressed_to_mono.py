#!/usr/bin/env python3
"""Decode a CompressedImage topic straight to raw mono8.

`image_transport republish` preserves the source encoding, so republishing a
colour JPEG yields rgb8 raw -- 1.49 MB per 704x704 frame. With the default
net.core.rmem_max of 208 KB, fragmented UDP reassembly drops most of those and
the consumer sees ~0.75 Hz instead of 5 Hz.

cuVSLAM only wants luminance anyway, so decoding directly to mono8 (495 KB at
704x704) keeps the stereo pair inside the transport's comfortable range while
the colour compressed topic stays untouched for visualisation.

    ros2 run rhody compressed_to_mono.py --ros-args \
        -r in/compressed:=/stereo/left/image_raw/compressed \
        -r out:=/stereo/left/image_mono
"""

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment diagnostic
    raise SystemExit(
        f"cannot import cv2: {exc}\n\n"
        "A user-local numpy in ~/.local usually shadows the one apt's "
        "python3-opencv was built against. Re-run with user site-packages "
        "disabled:\n"
        "    PYTHONNOUSERSITE=1 ros2 run rhody compressed_to_mono.py ...\n"
        "voyis_slam_demo.launch.py already sets this for you."
    ) from exc

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class CompressedToMono(Node):
    def __init__(self):
        super().__init__('compressed_to_mono')
        self.declare_parameter('queue_size', 5)
        queue = self.get_parameter('queue_size').value
        self.pub = self.create_publisher(Image, 'out', queue)
        self.sub = self.create_subscription(
            CompressedImage, 'in/compressed', self.on_image, queue)
        self.failures = 0

    def on_image(self, msg):
        buf = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self.failures += 1
            if self.failures <= 3:
                self.get_logger().warn(f'failed to decode frame (format {msg.format!r})')
            return

        out = Image()
        out.header = msg.header
        out.height, out.width = img.shape[:2]
        out.encoding = 'mono8'
        out.is_bigendian = 0
        out.step = img.shape[1]
        out.data = np.ascontiguousarray(img).tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CompressedToMono()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
