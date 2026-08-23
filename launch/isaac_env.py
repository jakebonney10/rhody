"""Environment fixes every Isaac ROS / OpenCV process in this workspace needs.

Two unrelated problems, both of which bite anything that loads GXF or OpenCV:

1. The apt isaac_ros_gxf keeps its GXF extensions under
   share/isaac_ros_gxf/gxf/lib/<component>/ but only puts <prefix>/lib on the
   loader path, so isaac_ros_visual_slam and nvblox_ros cannot find
   libgxf_serialization.so. Every directory under that root has to go on
   LD_LIBRARY_PATH.

2. gemini_sonar_driver vendors its SDK's ffmpeg (libavcodec.so.58,
   libavutil.so.56) into install/gemini_sonar_driver/lib, which colcon puts on
   the global LD_LIBRARY_PATH. Those shadow the system ffmpeg, so the system
   libchromaprint that OpenCV pulls in fails with

       undefined symbol: av_rdft_calc, version LIBAVCODEC_58

   and every OpenCV consumer in the workspace dies. The real fix belongs in
   gemini_sonar_driver: install the vendored SDK libs into a private
   subdirectory and reach them via the $ORIGIN RPATH instead of the search path.

Both are scoped per-process through `Node(additional_env=...)` rather than
mutating the whole launch, so unrelated nodes keep the environment they were
started with.

ros2 launch loads launch files by path rather than importing them as a package,
so callers reach this module by putting their own directory on sys.path:

    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    from isaac_env import isaac_env, opencv_env
"""

import glob
import os

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)


def _gxf_extension_dirs():
    try:
        gxf_root = os.path.join(get_package_share_directory('isaac_ros_gxf'), 'gxf', 'lib')
    except PackageNotFoundError:
        return []
    return sorted(d for d in glob.glob(f'{gxf_root}/*') if os.path.isdir(d))


def without_vendored_ffmpeg(path):
    """Drop every LD_LIBRARY_PATH entry that carries its own libavcodec."""
    keep = []
    for entry in path.split(':'):
        if entry and not glob.glob(os.path.join(entry, 'libavcodec.so*')):
            keep.append(entry)
    return ':'.join(keep)


def clean_ld_library_path():
    return without_vendored_ffmpeg(os.environ.get('LD_LIBRARY_PATH', ''))


def isaac_env():
    """Environment for a process that loads GXF: visual_slam, nvblox, NITROS."""
    return {'LD_LIBRARY_PATH': ':'.join(_gxf_extension_dirs() + [clean_ld_library_path()])}


def opencv_env():
    """Environment for a Python process that imports cv2.

    PYTHONNOUSERSITE additionally guards against a user-local numpy in ~/.local
    shadowing the one apt's python3-opencv was built against.
    """
    return {'LD_LIBRARY_PATH': clean_ld_library_path(), 'PYTHONNOUSERSITE': '1'}


def rviz_env():
    """Environment for rviz2, which links OpenCV through the image displays."""
    return {'LD_LIBRARY_PATH': clean_ld_library_path()}
