# Neuracore local inference image.
# Separate from the lerobot image so torch/transformers/diffusers pins don't
# fight each other.
FROM ros:jazzy-ros-base

ARG NEURACORE_VERSION=11.0.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# neuracore[ml] pulls torch 2.7.1, transformers, diffusers, etc.
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed \
    numpy packaging "neuracore[ml]==${NEURACORE_VERSION}" && \
    pip3 install --break-system-packages --no-cache-dir \
        "setuptools<80" \
        "opencv-python-headless>=4.8.0,<5.0.0"

# Neuracore's policy loader shells out to `pip install -r requirements.txt`
# against the algorithm bundle inside the .nc.zip at runtime. On Debian 12+
# that hits PEP 668 (externally-managed). This env var lets pip proceed.
# Placed after the heavy pip layer so edits here don't invalidate its cache.
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY ros2/src/neuracore_control /workspace/src/neuracore_control
COPY configs /workspace/configs

RUN . /opt/ros/jazzy/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install

COPY docker/neuracore/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "neuracore_control", "inference.launch.py"]
