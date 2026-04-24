from glob import glob

from setuptools import find_packages, setup

package_name = "neuracore_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=[]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
    ],
    python_requires=">=3.10",
    install_requires=[
        "setuptools",
        "numpy>=2.0,<2.3.0",
        "opencv-python-headless>=4.8.0,<5.0.0",
    ],
    zip_safe=True,
    maintainer="Daniel Pino",
    maintainer_email="daniel@anvil.bot",
    description="Local Neuracore policy inference for OpenArm follower arms",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inference_node = neuracore_control.inference_node:main",
        ],
    },
)
