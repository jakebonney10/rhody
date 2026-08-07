# rhody

run `ros2 run tf2_tools view_frames` to generate a diagram of the TF tree. 

For static transform to view tf run `ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map rhody/base_link`.

---

## Voyis + Visual SLAM demo

Replays a Voyis survey bag through Isaac ROS Visual SLAM with the Rhody 2 model
in the loop, for RViz and Foxglove.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch rhody voyis_slam_demo.launch.py bag:=$HOME/bags/demo_wreck_pass
```

(The launch file expands `~` itself, but `bag:=~/...` is *not* tilde-expanded by
the shell — the `~` is not at the start of the word — so prefer `$HOME`.)

Arguments: `rate` (default 1.0), `loop` (default true), `rviz` (true),
`foxglove` (true, serves `ws://localhost:8765`).

### What it wires up

```
bag play ──> stereo/{left,right}/image_raw/compressed ──> RViz / Foxglove (colour)
                    │
                    └─ compressed_to_mono.py ─> image_mono ─> cuVSLAM ─> odom->base_link TF
robot_state_publisher (rhody2.urdf.xacro) ──> base_link -> voyis_*_optical + robot model
```

Because cuVSLAM publishes `odom -> base_link`, the whole Rhody model flies
through the map on the SLAM solution.

### Building a demo bag

The bag must be built with `--no-tf-static` — the URDF owns every transform
below `base_link`, and a bag publishing them too would fight
`robot_state_publisher` for the same edges of the TF tree.

```bash
ros2 run voyis_discovery_driver voyis_images_to_bag.py <export_dir> \
    -o ~/bags/demo_wreck_pass \
    --start "2025-05-27 12:57:13" --end "2025-05-27 12:59:44" \
    --scale 0.25 --encoding rgb8 --no-tf-static --no-gt
```

### Gotchas this launch file already handles

- **GXF extensions are not on the loader path.** The apt `isaac_ros_gxf` ships
  its extensions under `share/isaac_ros_gxf/gxf/lib/<component>/` but its ament
  hook only adds `<prefix>/lib`, so `visual_slam` dies with
  `libgxf_serialization.so: cannot open shared object file`.
- **`image_jitter_threshold_ms` is a frame-to-frame delta cap**, not a jitter
  tolerance. It must exceed the 200 ms survey period or every frame is rejected.
- **Colour raw does not fit through the default transport.** `net.core.rmem_max`
  defaults to 208 KB; an rgb8 704×704 raw frame is 1.49 MB and fragmented UDP
  reassembly drops most of them (cuVSLAM saw 0.75 Hz instead of 5 Hz).
  `compressed_to_mono.py` decodes straight to mono8 (495 KB), which is all
  cuVSLAM uses, and leaves the colour compressed topic alone for display.
- **The cuVSLAM `vis/*` clouds are silent by default** — they need
  `enable_slam_visualization`, `enable_landmarks_view` and
  `enable_observations_view`.
- **RViz infers image transport from the topic name**, so the Image displays
  point at `.../image_raw/compressed` directly. There is no transport dropdown.

### Foxglove

With `foxglove:=true`, connect Foxglove Studio to `ws://localhost:8765`
(Open connection → Foxglove WebSocket). Useful panels:

- **3D** — set the frame to `odom`; it picks the robot model up from
  `/robot_description` and the trajectory from `/visual_slam/tracking/slam_path`.
- **Image** ×2 on `/stereo/left/image_raw/compressed` and the right equivalent.

### Known workspace issues

**`gemini_sonar_driver`'s vendored ffmpeg breaks OpenCV workspace-wide.**
`install/gemini_sonar_driver/lib` contains `libavcodec.so.58` and
`libavutil.so.56` from the Tritech SDK, and colcon puts that directory on the
global `LD_LIBRARY_PATH`. They shadow the system ffmpeg, so the system
`libchromaprint.so.1` that OpenCV pulls in fails with:

```
undefined symbol: av_rdft_calc, version LIBAVCODEC_58
```

After `source install/setup.bash`, *any* node that imports `cv2` — or uses
`compressed_image_transport`, including RViz — dies. Reproduce with
`source install/setup.bash && python3 -c "import cv2"`.

This launch file works around it by stripping directories containing
`libavcodec.so*` from `LD_LIBRARY_PATH` for the processes that need OpenCV. The
real fix belongs in `gemini_sonar_driver`: install the vendored SDK libraries
into a private subdirectory reached via the `$ORIGIN` RPATH so they never reach
the global search path.

**Stale `template_pkg` in the install space.** `source install/setup.bash`
prints `not found: ".../install/template_pkg/share/template_pkg/local_setup.bash"`
from a package that no longer exists. Harmless but noisy;
`rm -rf install/template_pkg` clears it.
