from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'trajectory_viz'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.rviz')),
        (os.path.join('share', package_name, 'meshes'),
            [f for f in glob('meshes/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'scripts'),
            glob('scripts/*.sh')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@email.com',
    description='RViz2 simulation to validate forklift to pallet trajectory pipeline',
    license='MIT',
    entry_points={
        'console_scripts': [
            'simulation_node = trajectory_viz.simulation_node:main',
            'animation_control = trajectory_viz.animation_control:main',
        ],
    },
)
