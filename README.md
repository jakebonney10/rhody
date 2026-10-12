# rhody

run `ros2 run tf2_tools view_frames` to generate a diagram of the TF tree.

## TF tree

`base_link` **is the USBL**, and follows REP-103 (x forward, y port, z up).

```
base_link (= USBL)
├─ usbl_link          identity
├─ nucleus_dvl        Nortek Nucleus 1000
├─ sprintnav_link     SPRINT-Nav Mini      *** PROVISIONAL, NOT MEASURED ***
├─ gemini_fls         Gemini 1200ik FLS, 5 deg down
├─ voyis_left_link  ──┬─ voyis_left_optical
│                     └─ voyis_imu         (identity placeholder)
└─ voyis_right_link ─── voyis_right_optical
```

The vehicle is published by `rhody-description.service` (`robot_state_publisher`
on `rhody2.urdf.xacro`, global namespace). Without it the only TF on the graph is
mavros's own `map_ned`/`odom_ned`/`base_link_frd` aliases, so no sensor frame
resolves and **recorded bags contain no extrinsics at all** — worth checking
before a dive, since nothing warns you.

Link names match what each driver actually publishes as its `frame_id`
(`nucleus_dvl`, `gemini_fls`), and the `sensor_fov.yaml` sections and
`meshes/fov/*.stl` prefixes, so there is one name per sensor across the repo.
The old generic `dvl_link`/`fls_link` are gone: with two DVLs aboard, "the DVL"
is ambiguous.

### Sensor frame policy (REP-145)

Every sensor driver on Rhody follows [REP-145](https://www.ros.org/reps/rep-0145.html),
and the rule is checked against each driver below. Stated once so the next
driver gets built the same way:

> **Body-frame measurements are published exactly as the device reports them,
> in the device's own axes as given in its manual.** `frame_id` names that
> frame, and the rotation from it to `base_link` is declared in
> `rhody2.urdf.xacro` — never applied inside the driver. (REP-145: *"all data
> should be published by the driver as it is reported by the device. Any
> subsequent modifications to the data (e.g. filtering, transformations) should
> be delegated to a downstream consumer."*)
>
> Three things are **not** covered by "as reported", because TF cannot reach
> them: the **world reference of an orientation** is converted to ENU in the
> driver (robot_localization rejects NED); **world-referenced quantities**
> (depth, fixes, INS velocity-over-ground) are ENU per REP-105; and **SI
> units, invalid-value sentinels, and each message's own contract** (e.g.
> `Dvl.msg` mandates FRD, covariance fields are variances) remain the
> driver's job.

What that means per sensor, and the frame each one is declared in:

| sensor | body vectors | declared axes | orientation world side |
|---|---|---|---|
| Nucleus (`nucleus_dvl`) | as reported | FRD, 180° roll from `base_link` | n/a (no std orientation published) |
| SPRINT-Nav (`sprintnav_link`) | as reported | FRD, 180° roll | NED→ENU in driver |
| Subsonus (`usbl_link`) | as reported | FRD, 180° roll (device set to *Subsea* alignment) | NED→ENU in driver |
| Voyis (`voyis_*_optical`) | as reported | optical, per REP-103 suffix frame | — |
| Gemini (`gemini_fls`) | sonar angles, consumer-interpreted | 5° down pitch | — |
| mavros | converts to ENU/FLU itself | `base_link_frd`/`map_ned` aliases | ENU |

mavros is upstream and the one accepted exception: it converts in the driver
and publishes `_frd`/`_ned` alias frames, which is also where those suffix
names come from (REP-103 itself defines only `_optical` and `_ned`).

The cost this policy accepts, with eyes open: data is only correct when the
frame is declared **and** `rhody-description.service` is running. Raw FRD
stamped `nucleus_dvl` with no URDF loaded is wrong and nothing flags it. That
service existing is part of the policy, not an optional extra.

### Why base_link is the USBL

All hull offsets were originally measured in Shapr3D *relative to the Nucleus
DVL*, and `base_link` sat there (as on Rhody 1). The Nucleus is a temporary test
fit for the two-DVL comparison, and a `base_link` defined by a sensor that gets
removed leaves every other extrinsic referred to a point no longer on the
vehicle. So `base_link` moved to the USBL, which is permanent.

The original measurements are kept verbatim in `rhody2.urdf.xacro` and shifted
into the USBL frame by a single pair of properties (`usbl_to_dvl_x/z`), so
re-measuring the USBL↔DVL offset is a one-line change. The move is a pure
translation: every sensor-to-sensor transform is unchanged, which is also why
the FOV meshes and the coverage-overlap figures did not need regenerating.

### Known gaps

- **`sprintnav_link` is provisional and not measured.** It is placed 8 in
  (0.2032 m) aft of the Nucleus on the centerline, with z assumed equal (both
  are bottom-mounted) and orientation assumed aligned. This bounds the two-DVL
  comparison: the heads are separated by a lever arm `r`, so under rotation they
  legitimately differ by `omega x r` — ~3.5 cm/s at 0.2 m and 10 deg/s, the same
  order as DVL accuracy. Measure it before trusting any DVL-vs-DVL result.
- **`map` and `odom` are disconnected islands.** `gps2map.launch.xml` publishes
  `utm_local -> map -> odom` and the EKF/`mru_transform` would publish
  `odom -> base_link`, but none of those run as services, so under the service
  deployment the vehicle has geometry but no localisation.
- adnav now publishes into `usbl_link` — `adnav_driver` gained a read-only
  `frame_id` parameter (it was hardcoded to `imu_link`, a link rhody2 does not
  declare, so no lookup against Subsonus data could resolve). Set in
  `config/adnav_tcp_client.yaml`. The driver default is unchanged, so this is
  the only config that needs it. **Not yet pinned**: the fix is committed on the
  submodule's `devel` branch but the superproject still records the previous
  commit.
- `voyis_imu` is still an identity placeholder pending the factory extrinsics.
- `rhody_description.launch.py` still defaults to `rhody.urdf.xacro`, the
  **Rhody 1** description. The service passes `model:=` explicitly; do the same
  when launching it by hand.

---

## Sensor field-of-view visualisation

Draws the coverage volume of every sensor on the Rhody 2 model: the two Voyis
stereo frustums, the Gemini FLS fan, the Water Linked 3D sonar wedge, and the
Nucleus DVL's three slanted beams plus altimeter.

Live, in RViz:

```bash
ros2 launch rhody sensor_fov.launch.py
```

Each volume is its own link, so `RobotModel > Links` gives you a checkbox per
sensor. `TF` is in the config but off by default — turn it on to check the
extrinsics against the volumes.

For a figure (offscreen, no display needed, writes PNG + PDF):

```bash
python3 src/rhody/scripts/render_fov_figure.py
# -> src/rhody/docs/figures/rhody2_sensor_fov.{png,pdf}
```

The figure is laid out at the width it will occupy on the page, so the point
sizes in it are the point sizes you get at
`\includegraphics[width=\textwidth]`. It defaults to a 6.5 in text block —
if yours differs, say so and the type and caption re-flow to match:

```bash
python3 src/rhody/scripts/render_fov_figure.py --width-in 5.5
```

Sizing type on an oversized canvas and letting LaTeX shrink it is what makes
figure text illegible; no amount of bumping the font size fixes it while the
canvas is wide. The run also prints a ready-to-paste `figure` environment with
the provenance caption, which is deliberately *not* burned into the image.

It also prints the pairwise coverage overlap, computed analytically rather than
from the meshes:

```
  voyis_stereo    & gemini_fls         1.12 m³   ( 20.8% of voyis_stereo,  57.7% of gemini_fls)
  voyis_stereo    & waterlinked_3d     2.22 m³   ( 41.1% of voyis_stereo,  77.5% of waterlinked_3d)
  gemini_fls      & waterlinked_3d     1.34 m³   ( 68.9% of gemini_fls,  46.9% of waterlinked_3d)
```

### How the pieces fit

| File | Owns |
|---|---|
| `config/sensor_fov.yaml` | FOV angles, ranges, colours, and the provenance of every number |
| `scripts/gen_fov_meshes.py` | bakes that YAML into `meshes/fov/*.stl` |
| `urdf/sensor_fov.xacro` | hangs those meshes off the sensor links, included only when `fov:=true`. Also owns the DVL beam *rotations* via its own `dvl_slant` property, which duplicates `slant_deg` in the YAML — keep the two in step |
| `scripts/render_fov_figure.py` | offscreen VTK render + overlap numbers |

Extrinsics are never duplicated: the FOV solids are generated in each sensor's
own frame with the apex at the origin, so they attach with an identity origin
and inherit the real mounting pose from the joints in `rhody2.urdf.xacro`. The
picture therefore cannot drift away from the calibration — if the URDF is
wrong, the figure is wrong the same way.

After editing `config/sensor_fov.yaml`, regenerate:

```bash
python3 src/rhody/scripts/gen_fov_meshes.py
```

### Caveats worth knowing before quoting any of this

- **Volumes are truncated at 2 m**, not at each sensor's true range (see
  `display_range_m`). Drawn to scale the Gemini's 50 m fan makes the vehicle a
  dot. True ranges are in the legend and in the YAML.
- **The Water Linked Sonar 3D-15 is not installed.** Both its 90° × 40° FOV and
  its mounting pose are placeholders — the FLS mount mirrored to starboard. Any
  quantitative claim about FLS/3D-sonar overlap depends on replacing that with
  measured offsets.
- **The DVL beam geometry is now from the datasheet**, not assumed: 20° slant,
  3.4° beamwidth, 1 MHz (Nucleus 1000 datasheet, rev 03/23/2025, under
  Hardware). The previous 25°/3.0° placeholders came from a Nortek web figure
  because `docs/N3015-033` gives no beam angles. Three slanted beams plus a
  dedicated vertical altimeter transducer is confirmed by both documents and
  the driver. The altimeter's own beamwidth is still assumed — the datasheet
  quotes one beam width and does not break the vertical beam out separately.
  Changing this did not move the overlap numbers below: the DVL needles are not
  part of that pairwise comparison.
- Voyis FOV (73.2° square) is derived, not assumed — from the factory intrinsics
  `f = 1895.675 px` on a 2816² sensor in `Calibration_180114031/*.xml`.
- `fov:=true` is opt-in and off by default, so SLAM/nvblox/EKF pipelines see the
  same 9-link tree they always did.

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

**Stale `template_pkg` in the install space.** Fixed — this was the
`src/hydroport` submodule sitting on its upstream `master`, which is the package
*template* (`template_pkg`). The submodule now pins `devel`, which carries the
real `hydroport` package. If an old workspace still prints
`not found: ".../install/template_pkg/share/template_pkg/local_setup.bash"` on
`source install/setup.bash`, `rm -rf install/template_pkg build/template_pkg`
clears the leftovers.

---

## MCAP logging (hydroport)

Records every topic on the graph into split `.mcap` bags, as a user systemd
service alongside the sensor drivers. The recorder is **remotely operated**: the
service is always up, but it writes nothing until an operator turns it on.

```bash
# start recording
ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable \
  /rhody/data/logging_on std_msgs/msg/Bool "{data: true}"

# stop (closes and flushes the current bag)
ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable \
  /rhody/data/logging_on std_msgs/msg/Bool "{data: false}"
```

Both the control topic and the `/rhody/data/bag_recorder/logging_on` state
echo are `transient_local`, so the setting is latched — the recorder picks it up
even if it (re)starts after the command was sent, and a late-joining Foxglove
sees the current state.

Status is on the global `/diagnostics` topic at 1 Hz, next to the other rhody
drivers:

| Level | Meaning |
|---|---|
| `OK` | `Recording N topics` — plus the topic count and output path as key/values |
| `WARN` | `Not recording — logging OFF` (idle is deliberately a WARN, so a dive with the logger off is visible) |
| `ERROR` | A setting is blocking recording — unwritable `logging.directory`, bad prefix. The node stays alive; fix the parameter and it retries |

Bags land in `~/data/rhody_mcap/rhody_<UTC ISO stamp>/`, split every 900 s, with
zstd chunk compression inside the mcap. Override the location per-run with
`log_dir:=`, or live with `ros2 param set /rhody/data/bag_recorder
logging.directory /some/path` — changing it while recording closes the current
bag and opens a new one at the new path.

| File | Owns |
|---|---|
| `config/hydroport_mcap.yaml` | prefix, ignore regex, discovery period, split length, compression preset |
| `launch/hydroport_mcap.launch.py` | namespace, and the output directory (from `$HOME`, so it follows the service user) |
| `systemd_services/rhody-hydroport.service` | the unit |

### Why `bag_recorder` and not `mcap_logger`

hydroport ships two mcap-writing nodes and rhody runs the newer one.
`mcap_logger` builds its own generic subscriptions on a fixed `KeepLast(10)`
RELIABLE profile, with `/tf` and `/tf_static` hand-special-cased. Any publisher
offering BEST_EFFORT reliability or TRANSIENT_LOCAL durability — the Gemini and
Voyis image streams — is then QoS-incompatible and **silently records nothing**;
no error is logged, the topic simply never appears in the bag.
`bag_recorder` wraps `rosbag2_transport::Recorder`, the class behind
`ros2 bag record`, which queries each publisher's offered QoS and adapts the
subscription to match (`Rosbag2QoS::adapt_request_to_offers`).

### Gotchas this setup already handles

- **The submodule's default branch is useless.** `SeawardScience/hydroport`'s
  `master` is the upstream package *template* — `template_pkg`, no `hydroport`
  package at all. The logger nodes live on `devel`, which is what
  `.gitmodules` pins (`branch = devel`) and what the recorded commit points at.
  This is also what used to produce the stale `install/template_pkg`
  `local_setup.bash` warning on `source install/setup.bash`.
- **Stopping the service must go through systemd, not a bare kill.**
  `kill -TERM` on the `ros2 launch` process alone kills the launcher and
  *orphans* the recorder, which keeps writing forever and never closes the bag —
  leaving a directory with an `.mcap` chunk and no `metadata.yaml`, which
  `ros2 bag info` cannot open. `systemctl --user stop` signals the whole
  cgroup, so the node itself gets SIGTERM, unwinds, and flushes; measured
  under a second. `TimeoutStopSec=60` in the unit is headroom for that flush,
  since a SIGKILL mid-chunk truncates the bag.
- **An unfinalized bag is not necessarily lost** — `ros2 bag reindex <dir> -s mcap`
  rebuilds `metadata.yaml` from the chunks.
- **`compression_format: ""` in `metadata.yaml` is correct.** `zstd_fast` is an
  *mcap storage* preset — zstd is applied per chunk inside the file, which is
  not rosbag2's file/message compression layer. Confirm with
  `strings <bag>.mcap | grep -m1 zstd`, not from the metadata.
- **Recording everything is the default and it adds up.** The ignore regex only
  drops hidden `/__*` topics and `/parameter_events`; the sonar and camera
  streams are recorded in full. 900 s splits bound the loss from a power cut
  and keep files copyable between dives, but they do not bound total volume —
  budget disk per dive, or add the heavy topics to `topic_ignore_regex`.

### Kernel UDP buffers

`install.sh` installs `system/60-rhody-udp.conf` into `/etc/sysctl.d/`, raising
`net.core.rmem_default` to 16 MB and `rmem_max` to 64 MB (and the same for
`wmem`). Fast DDS sizes each socket from the kernel default unless an XML
profile says otherwise, so with the stock 208 KB the recorder's data socket
overflowed: a 2.4 MB point cloud or 2 MB sonar image arrives as ~40
back-to-back 64 KB datagrams over loopback (same-host traffic over the
512 KB shared-memory segment limit falls back to UDP), and every overflow is a
lost message on a best-effort topic or a retransmit on a reliable one.

Measured 2026-10-12 in the tank, 60 s windows with a scratch recorder on the
full topic set: 28,560 datagrams dropped on the recorder socket and 152
messages reported lost on the transport layer before; 0 and 0 after. Host-wide
`Udp: RcvbufErrors` went from 39,611 to 10 over the same windows.

- A process only picks the new default up when it creates its sockets, so
  after changing the file restart the ROS services (or reboot).
- Check for drops with `grep Udp: /proc/net/snmp` (the `RcvbufErrors`
  column) and per-socket in the last column of `/proc/net/udp`.
- The Voyis camera data port (17913) still drops at ~0.5/s after the change;
  the Discovery SDK appears to size that socket itself. That is device-side
  traffic, not DDS.

---

## Nortek Nucleus 1000 DVL

Runs as a user systemd service in **listen-only** mode: the driver attaches to
the Nucleus read-only streaming port (9002), parses whatever is already being
streamed, and never writes to the socket. Configuration and start/stop stay with
whatever client owns the command port (9000) — normally the Nortek software.

```bash
systemctl --user status rhody-nucleus.service
journalctl --user-unit rhody-nucleus.service -f    # note: --user-unit, not --user -u
```

DVL data is published three ways: `dvl` and `water_track_dvl`
(`marine_acoustic_msgs/Dvl`, the cross-vendor message sprintnav also uses, with
the per-beam ranges/velocities/quality), `bottom_lock_velocity_common` and
`water_track_velocity_common` (`TwistWithCovarianceStamped`), and the raw
Nortek-shaped `*_packets`. **The Dvl and the Twist deliberately disagree on
axes**: `Dvl.msg` documents its own FRD convention, which is what the Nucleus
reports natively, while the Twist must match its REP-103 frame and so has y and
z negated. sprintnav splits it the same way — do not "fix" one to match the
other.

Topics are at `/rhody/nav/sensors/nucleus_node/*`, alongside `sprintnav/` and
`adnav/` in the same namespace. The `*_packets` topics are the raw Nortek
messages (custom types from the driver's `interfaces` package); the `*_common`
topics are the `sensor_msgs`/`geometry_msgs` equivalents — `imu_common`,
`magnetic_common`, `pressure_common`, `temperature_common`, `sound_speed_common`,
`depth_odometry_common`, `bottom_lock_velocity_common`,
`water_track_velocity_common`, `altimeter_common` and `altimeter_beam{1,2,3}_common`.

### Diagnostics

On the `feature/streaming-only-reconnect` branch the driver publishes to the
global `/diagnostics` like the other sensors (hardware ID `nucleus_1000`).
`Connection` and `Decoder` mirror `sprintnav_driver`/`adnav_driver`; `Bottom
Track` is the DVL-specific one, since bottom lock is what says whether the
device is usable for navigation.

| Task | OK | WARN | ERROR / STALE |
|---|---|---|---|
| `Connection` | `receiving` | `connected, no data - is measurement started?` | `not connected`, or `data timeout` in full-control mode |
| `Decoder` | `decoding N packets` | `N unrecognised packet(s)` — firmware/parser mismatch | STALE until the first packet |
| `Bottom Track` | `bottom lock` | `no bottom lock` (off-bottom/out of range) | STALE until the first bottom-track packet |

`Connection` also carries the endpoint, seconds since last data, connect count
and the reconnect/keepalive settings. Note that in listen-only mode a silence is
a **WARN, not an ERROR** — an external client owns start/stop, so an idle device
is not a fault. For the same reason the task's staleness threshold does not reuse
`data_timeout_s`, which is `0.0` in that mode.

| File | Owns |
|---|---|
| `config/nucleus_listen.yaml` | IP, stream port 9002, frames, fluid density, reconnect/keepalive |
| `launch/nucleus_listen.launch.py` | namespace `rhody/nav/sensors` |
| `systemd_services/rhody-nucleus.service` | the unit, and the reachability retry loop |
| `systemd_services/rhody_nucleus.sh` | the pre-launch port probe |

`config/nucleus.yaml` + `launch/nucleus.launch.py` are the full-control variant,
kept for manual use: they log into port 9000, push the `SETIMU`/`SETAHRS`/
`SETBT`/`SETTRIG` block and auto-start measurement. Nothing runs them by default
— the service deliberately does not own the device.

### Gotchas this setup already handles

- **The driver now reconnects, but only on the workspace branch.** Upstream
  `auto_connect` is a single attempt during node init: on failure the node logs
  an error, *keeps running* with no connection, and never retries, and a link
  that drops later is never noticed because the read loop just keeps timing out.
  Neither case exits, so `Restart=always` could not paper over it. The submodule
  branch `feature/streaming-only-reconnect` moves (re)connection into the packet
  thread — see `reconnect_interval_s`, `tcp_keepalive_idle_s` and
  `data_timeout_s` in the config. **This is not yet merged or pinned**: the
  superproject still records the pre-reconnect commit, pending a hardware test.
- **`rhody_nucleus.sh` probes the port before launching.** With reconnect in the
  driver this is no longer load-bearing, but it is kept as cheap boot ordering:
  it exits non-zero while the Nucleus is unreachable so systemd retries. A
  service sitting in `activating (auto-restart)` with the device powered off is
  **working as intended**, not failing.
- **`data_timeout_s` is off in listen-only mode on purpose.** In that mode an
  external client owns start/stop, so a long silence just means nobody has
  started measurement — it is not a dead link. Enabling the timeout there makes
  the node reconnect every few seconds forever against a perfectly healthy
  device (measured: 5 reconnects in 23 s at `data_timeout_s: 3.0`). Link loss is
  detected by TCP keepalive plus an `SO_ERROR` / `MSG_PEEK` EOF check instead,
  which is what `tcp_keepalive_idle_s` tunes. `nucleus.yaml` (full control) does
  enable the timeout, because there the node starts measurement itself and
  silence really is a fault.
- **The `nucleus_driver` Python library is not installed by colcon.** The ROS
  node does `from nucleus_driver import NucleusDriver`, but that library is the
  plain-Python package next to it in the submodule (`nucleus_driver/nucleus_driver/`),
  which is pip-shaped, not ament-shaped. This image has no pip at all and is PEP
  668 externally managed, so `install.sh` instead drops a `.pth` file naming the
  library's `src/` into the user site-packages directory. No sudo, no pip, and it
  follows the submodule checkout. Without it the node dies immediately with
  `ModuleNotFoundError: No module named 'nucleus_driver'`. Only `pyserial` is
  actually needed at runtime (present); `cmd2`, also in `requirements.txt`, is
  used solely by the library's console entry point.
- **`sensor_configs: []` crashes the node on startup.** The driver declares that
  parameter by handing rclpy a bare `ParameterValue` as the default, which rclpy
  rejects, so the parameter gets its type *only* from the params file — and an
  empty YAML sequence carries no type. The result is
  `ParameterUninitializedException: sensor_configs` before the node ever
  connects. `nucleus_listen.yaml` uses `['']`, the smallest value that types the
  array; listen-only mode skips `sensor_configs` anyway, so nothing is sent and
  the node just logs "sensor_configs is set but will be ignored in listen-only
  mode". Expect that warning.
