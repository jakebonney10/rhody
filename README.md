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

`--no-tf-static` is recommended but no longer required: the launch replays with
`--remap /tf_static:=/bag_tf_static`, so `robot_state_publisher` owns every edge
below `base_link` regardless of what the bag carries. See the `/tf_static`
gotcha below for what used to happen.

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
- **A bag's `/tf_static` silently beats the URDF.** `voyis_images_to_bag.py`
  defaults `--base-to-left` to identity, and `william_elgin_704` was built
  without `--no-tf-static`. When both publish, the bag wins the race and
  `base_link -> voyis_left_optical` resolves to *identity* — the optical frame
  rotation vanishes, so the camera points along `base_link` +z (straight up)
  instead of forward. cuVSLAM still tracks happily and nothing logs an error;
  the only symptom is that reconstructions land above the vehicle. Verify with:

  ```bash
  ros2 run tf2_ros tf2_echo base_link voyis_left_optical
  # want xyz +0.3054 +0.0849 +0.0908, quat -0.5 +0.5 -0.5 +0.5
  ```

  The launch now diverts the bag's copy to `/bag_tf_static`.

---

## Dense reconstruction with nvblox

`nvblox:=true` adds an Isaac ROS nvblox TSDF reconstruction on top of the same
stereo pair, turning the sparse SLAM landmarks into an actual surface.

```bash
sudo apt install ros-humble-nvblox-ros ros-humble-nvblox-rviz-plugin \
                 ros-humble-isaac-ros-stereo-image-proc
ros2 launch rhody voyis_slam_demo.launch.py bag:=$HOME/bags/william_elgin_704 nvblox:=true
```

Install `nvblox_ros`, **not** the `isaac_ros_nvblox` metapackage. The metapackage
depends on `nvblox_examples_bringup`, whose ESS and people-segmentation demos
pull `gxf_isaac_triton` → `libnvinfer10`; TensorRT is not installable from the
configured apt sources and none of it is needed here.

`voyis_nvblox.launch.py` can also be run on its own against the live driver — it
assumes only that the stereo topics and the TF chain already exist.

```
/stereo/{left,right}/image_mono ──> DisparityNode (VPI SGM)
                                        │ disparity
                                        v
                                    DisparityToDepthNode
                                        │ depth (32FC1)
                                        v
                                      nvblox ──> /nvblox_node/mesh
```

All three are composed into one `component_container_mt` with intra-process
comms deliberately: a 32FC1 depth image is 1.98 MB at 704×704 and would hit the
same `net.core.rmem_max` wall that pushed the stereo pair to mono8. Composed,
disparity and depth never reach the wire.

### Sizing the voxels

From the bag's own calibration (`william_elgin_704.log`) at `--scale 0.25`:
`f = 473.92 px`, `B = 0.169697 m`, so `f·B = 80.42 px·m` and `depth = 80.42 / d`.

| altitude | disparity | depth error per px of disparity |
|---|---|---|
| 2 m | 40.2 px | 0.05 m |
| 4 m | 20.1 px | 0.20 m |
| 6 m | 13.4 px | 0.45 m |

The 5 cm default voxel is matched to roughly 2 m altitude, and past ~5 m one
pixel of disparity error already exceeds a voxel — hence
`projective_integrator_max_integration_distance_m: 5.0`. For a survey flown
higher than ~3 m, raise `voxel_size` rather than the integration distance.

`max_disparity: 128` bounds the *near* field at 0.63 m, which is what stops SGM
painting backscatter and particulate drifting in front of the dome into the map
as solid surface.

### Exporting the map

```bash
ros2 service call /nvblox_node/save_ply nvblox_msgs/srv/FilePath \
    "{file_path: $HOME/bags/william_elgin_704.ply}"
```

### Colour

```bash
ros2 launch rhody voyis_slam_demo.launch.py bag:=$HOME/bags/demo_wreck_long \
    nvblox:=true color:=true
```

`color:=true` adds a `rhody::CompressedDecoderNode` to the container that
decodes the left eye's JPEG straight to rgb8, and sets `use_color: true`. Depth
and colour are already registered to each other — both are the left rectified
view, sharing `voyis_left_optical` and one `camera_info`.

**Only on the rgb8 bags** (`demo_wreck_pass`, `demo_wreck_long`). The
`william_elgin` exports were built `--encoding mono8`, where this integrates a
grey colour layer for nothing.

The decode is a composable node rather than a third `compressed_to_mono.py`
process for the usual reason: raw rgb8 at 704×704 is 1.49 MB and does not
survive the default transport. In-container, only the bag's ~150 KB JPEG is on
the wire, and it runs at the full survey rate with no `net.core.rmem_max`
tuning.

`cv::imdecode` returns BGR whatever the JPEG was tagged as, so the node swaps to
RGB explicitly. Getting that wrong is easy to miss — a red/blue swap still looks
like a plausible seabed. Verify against an independent decode rather than by
eye:

```bash
ros2 topic echo /nvblox/left/image_color --field encoding   # expect rgb8
```

### Gotchas

- **The nvblox mesh display needs `scripts/install_rviz_glsl150.sh` first.**
  `ros-humble-rviz-rendering` 11.2.23 deleted
  `ogre_media/materials/{glsl150,scripts150}`
  ([isaac_ros_nvblox#159](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox/issues/159)),
  and `nvblox_rviz_plugin` still asks for them — it registers those directories
  as an Ogre group `rviz_rendering_150` and looks up the geometry program
  `rviz/glsl150/box.geom` the first time it draws a mesh block. Without them
  RViz does not warn, it aborts:

  ```
  ItemIdentityException: Unable to locate geometry program called rviz/glsl150/box.geom
  terminate called after throwing an instance of 'Ogre::ItemIdentityException'
  ```

  It only fires once mesh geometry is on screen, so a session with the mesh
  off-camera survives and one with it in view dies — which reads as intermittent
  rather than deterministic. The script restores both directories from rviz
  11.2.22, the last release that shipped them. Additive only; re-run after any
  `apt upgrade` touching `rviz_rendering`, since no newer version brings them
  back. `./install_rviz_glsl150.sh --check` reports status without changing
  anything.
- **`loop:=true` throws the map away each lap.** Looping restarts the bag's
  clock, `tf2_buffer` clears on the jump back in time, and nvblox starts a fresh
  map — so a looping demo shows a map that never covers more than one pass, and
  RViz flashes `Global Status: Error` while TF re-establishes. Use `loop:=false`
  for anything you intend to `save_ply`, and note the launch shuts everything
  down when the bag ends (`on_exit=Shutdown`), so save before the last frame.
- **The mesh is RViz only.** Foxglove cannot render `nvblox_msgs/msg/Mesh` —
  there is no schema for it. For a Foxglove view use
  `/nvblox_node/tsdf_layer_marker` (a `visualization_msgs` cube marker of the
  TSDF) or `/nvblox_node/static_esdf_pointcloud` (a `PointCloud2`).
- **Two defaults silently cap the map at ~10 m and are both wrong for a
  survey.** `map_clearing_radius_m: 5.0` deletes blocks further than 5 m from
  `base_link`, and `decay_tsdf_rate_hz: 5.0` decays TSDF weights by
  `tsdf_decay_factor: 0.95` until they drop under
  `static_mapper.tsdf_decayed_weight_threshold: 0.001`, at which point
  `decay_integrator_deallocate_decayed_blocks` frees them — about 30 s after the
  camera leaves. At ~0.34 m/s that is a map permanently trailing the vehicle by
  ~10 m no matter how long the bag runs, with no warning logged. Both are
  disabled in `nvblox_voyis.yaml`. Symptom to watch for: `save_ply` returns a
  roughly constant vertex count while `ros2 topic echo
  /visual_slam/tracking/odometry` shows the vehicle tens of metres away from the
  map's bounding box.
- **A full survey at 5 cm voxels does not fit in 8 GB.** With clearing and decay
  both off — which is what you want for a survey — the map grows without bound
  and the RTX 4070's 8 GB runs out. Measured: ~10 minutes of `william_elgin_704`
  reached 131072 blocks and the container died with

  ```
  CUDA error = 2 ... 'cudaMallocAsync(...)'. Error string: out of memory.
  ```

  It is a hard crash of the whole component container, not a graceful degrade,
  and cuVSLAM keeps running afterwards so the launch does not exit. For a full
  48-minute survey either raise `voxel_size` to 0.10 (8x fewer voxels) or map
  one pass at a time and `save_ply` between them.
- **nvblox integrates in `odom`, not `map`.** cuVSLAM's `odom` is smooth;
  `map` jumps on every loop closure and nvblox cannot deform a map it has
  already integrated. The reconstruction drifts rather than tearing.
- **`nvblox_ros` loads GXF**, so it needs the same loader-path fix as
  `visual_slam`. Both get it from `launch/isaac_env.py`.
- **VPI SGM cannot take `mono8`.** NITROS advertises `nitros_image_mono8` as an
  accepted format for `DisparityNode`, but that is only the ROS-side type
  adaptation table — the VPI wrapper underneath cannot map an 8-bit grayscale
  GXF buffer and kills the whole container on the first frame with
  `GXF_INVALID_DATA_FORMAT` out of `vpi::VideoFormatToImageFormat`. Two
  `ImageFormatConverterNode`s expand mono8 to rgb8 on the GPU inside the
  container, so the wire still only carries the 495 KB mono8 images.
- **NITROS nodes reject `use_intra_process_comms`.** They create transient-local
  publishers internally, and rclcpp's intra-process path allows only volatile
  durability, so the component constructor throws. NITROS negotiates its own
  CUDA-buffer handover inside a container anyway, which is better than the
  rclcpp path since the data never leaves the GPU.
- **Integrator parameters live under `static_mapper.`.** Setting
  `projective_integrator_max_integration_distance_m` bare is silently ignored
  and the node keeps its 7 m default. The only symptom is a mesh built from
  garbage far-field depth. Check what actually applied:

  ```bash
  ros2 param get /nvblox_node static_mapper.projective_integrator_max_integration_distance_m
  # "Parameter not set" means you got the namespace wrong
  ```

  Top-level nvblox params (`voxel_size`, `global_frame`, `esdf_mode`,
  `max_back_projection_distance`) take no prefix.
- **SGM leaves a far-field garbage tail.** Textureless seabed yields near-zero
  disparity, so the depth image runs out to hundreds of metres even though the
  median sits at a plausible 3–4 m. That is what
  `static_mapper.projective_integrator_max_integration_distance_m: 5.0` is for —
  the tail exists in the depth image but is never integrated.

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
