from setuptools import setup

package_name = 'gesture_vision'

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
    maintainer='caleb',
    maintainer_email='caleb@todo.todo',
    description='Webcam publisher + hand gesture recognition for ROS 2 Humble.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webcam_publisher = gesture_vision.webcam_publisher:main',
            'gesture_recognizer = gesture_vision.gesture_recognizer:main',
            'gesture_recognizer_nocvbridge = gesture_vision.gesture_recognizer_nocvbridge:main',
        ],
    },
)
