from glob import glob

from setuptools import find_packages, setup

package_name = "lerobot_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=[]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    python_requires=">=3.12",
    install_requires=[
        "setuptools",
        "lerobot~=0.5.0",
        "numpy>=2.0,<2.3.0",
        "opencv-python-headless>=4.8.0,<5.0.0",
        "pyyaml>=6.0",
    ],
    zip_safe=True,
    maintainer="Patrick Hsu",
    maintainer_email="patrick.hsu@anvil.bot",
    description="LeRobot model inference and control for YAM robot arms",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = lerobot_control.inference_node:main",
            "mock_controller_node = lerobot_control.test.fake_hardware.fake_hardware_node:main",
            "mcap_player_node = lerobot_control.mcap_player_node:main",
            "eval_recorder_node = lerobot_control.eval_recorder_node:main",
        ],
    },
)
