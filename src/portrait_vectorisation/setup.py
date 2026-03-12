from setuptools import setup

package_name = 'portrait_vectorisation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maksim',
    maintainer_email='maksim.lovchev@student.uts.edu.au',
    description='Camera publisher for ROS 2 Humble.',
    license='Apache License 2.0',
        extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_publisher = portrait_vectorisation.camera_publisher:main',
        ],
    },
)