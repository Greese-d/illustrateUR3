from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'illustrateur3_gui'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Main Launch for whole system
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='caleb',
    maintainer_email='caleb.chadwick1@icloud.com',
    description='GUI package for the illustrateUR3 selfie drawing robot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gui_main = illustrateur3_gui.main:main',
        ],
    },
)