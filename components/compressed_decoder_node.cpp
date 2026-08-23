// Decode a CompressedImage topic to raw, as a composable node.
//
// This is the C++ twin of python/compressed_to_mono.py, and it exists for one
// reason: nvblox wants a colour image, and a raw rgb8 704x704 frame is 1.49 MB.
// With net.core.rmem_max at its 212992 B default, fragmented UDP reassembly
// shreds frames that size and the consumer sees well under 1 Hz (see the
// throughput notes in README.md). Composed into the nvblox container, the rgb8
// frame never reaches the wire -- only the ~150 KB JPEG the bag already
// publishes does.
//
// The Python script stays as-is for the SLAM-only path, where mono8 is small
// enough to survive the transport and a separate process is simpler to reason
// about.
//
//     ComposableNode(
//         package='rhody', plugin='rhody::CompressedDecoderNode',
//         parameters=[{'encoding': 'rgb8'}],
//         remappings=[('in/compressed', '/stereo/left/image_raw/compressed'),
//                     ('out', '/nvblox/left/image_color')])

#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace rhody
{

class CompressedDecoderNode : public rclcpp::Node
{
public:
  explicit CompressedDecoderNode(const rclcpp::NodeOptions & options)
  : Node("compressed_decoder", options)
  {
    encoding_ = declare_parameter<std::string>("encoding", "rgb8");
    const int queue = declare_parameter<int>("queue_size", 5);

    if (encoding_ != "rgb8" && encoding_ != "bgr8" && encoding_ != "mono8") {
      RCLCPP_ERROR(get_logger(), "unsupported encoding '%s' (rgb8|bgr8|mono8)", encoding_.c_str());
      throw std::runtime_error("unsupported encoding: " + encoding_);
    }

    pub_ = create_publisher<sensor_msgs::msg::Image>("out", queue);
    sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      "in/compressed", queue,
      [this](const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {onImage(msg);});
  }

private:
  void onImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr & msg)
  {
    const bool mono = encoding_ == "mono8";
    const cv::Mat buf(1, static_cast<int>(msg->data.size()), CV_8UC1,
      const_cast<uint8_t *>(msg->data.data()));
    cv::Mat img = cv::imdecode(buf, mono ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);

    if (img.empty()) {
      if (++failures_ <= 3) {
        RCLCPP_WARN(get_logger(), "failed to decode frame (format '%s')", msg->format.c_str());
      }
      return;
    }

    // imdecode hands back BGR regardless of what the JPEG was tagged as, so the
    // rgb8 case needs an explicit swap -- publishing BGR bytes labelled rgb8
    // would leave nvblox's mesh looking plausible but with red and blue
    // exchanged, which is easy to miss on a blue-green seabed.
    if (encoding_ == "rgb8") {
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    }

    auto out = std::make_unique<sensor_msgs::msg::Image>();
    out->header = msg->header;
    out->height = static_cast<uint32_t>(img.rows);
    out->width = static_cast<uint32_t>(img.cols);
    out->encoding = encoding_;
    out->is_bigendian = 0;
    out->step = static_cast<uint32_t>(img.cols * img.elemSize());
    out->data.assign(img.datastart, img.dataend);
    pub_->publish(std::move(out));
  }

  std::string encoding_;
  int failures_{0};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr sub_;
};

}  // namespace rhody

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(rhody::CompressedDecoderNode)
