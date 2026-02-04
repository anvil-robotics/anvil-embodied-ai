# Use ROS2 Jazzy as base
FROM ros:jazzy-ros-base

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DOMAIN_ID=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace
WORKDIR /workspace

# Copy ROS2 package
COPY ros2/src/lerobot_control /workspace/src/lerobot_control

# Copy configs
COPY configs/lerobot_control /workspace/configs

# Install Python dependencies
RUN cd /workspace/src/lerobot_control && \
    pip3 install -r requirements.txt || true

# Build ROS2 workspace
RUN . /opt/ros/jazzy/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install

# Set entrypoint
COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "lerobot_control", "inference.launch.py"]
