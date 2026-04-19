import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory, get_package_prefix


def generate_launch_description():
    pkg_share = get_package_share_directory('trajectory_viz')

    rviz_config = os.path.join(pkg_share, 'config', 'display.rviz')

    # Wrapper script that strips snap library paths before launching rviz2.
    # Needed when running from a VS Code snap terminal — see scripts/snap_clean_exec.sh.
    wrapper = os.path.join(pkg_share, 'scripts', 'snap_clean_exec.sh')
    rviz2_bin = os.path.join(get_package_prefix('rviz2'), 'lib', 'rviz2', 'rviz2')

    params_file = os.path.join(pkg_share, 'config', 'pallet_poses.yaml')

    return LaunchDescription([
        Node(
            package='trajectory_viz',
            executable='simulation_node',
            name='simulation_node',
            output='screen',
            parameters=[params_file],
        ),
        ExecuteProcess(
            cmd=['bash', wrapper, rviz2_bin, '-d', rviz_config],
            output='screen',
        ),
    ])
