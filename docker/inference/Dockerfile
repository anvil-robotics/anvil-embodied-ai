# Use ROS2 Jazzy as base
FROM ros:jazzy-ros-base

ARG LEROBOT_VERSION=0.4.4
ARG LEROBOT_EXTRAS=""

# Set environment
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace
WORKDIR /workspace

# Base lerobot install — cached as long as version doesn't change
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed \
    numpy packaging "lerobot==${LEROBOT_VERSION}" && \
    pip3 install --break-system-packages --no-cache-dir "setuptools<80"

# Extras — separate layer, only rebuilt when LEROBOT_EXTRAS changes
RUN if [ -n "${LEROBOT_EXTRAS}" ]; then \
        pip3 install --break-system-packages --no-cache-dir \
            "lerobot[${LEROBOT_EXTRAS}]==${LEROBOT_VERSION}"; \
    fi

# Copy ROS2 package (changes often → late layer)
COPY ros2/src/lerobot_control /workspace/src/lerobot_control

# Copy configs (lerobot_control + cyclonedds)
COPY configs /workspace/configs

# Build ROS2 workspace
RUN . /opt/ros/jazzy/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install

# Set entrypoint
COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "lerobot_control", "inference.launch.py"]
