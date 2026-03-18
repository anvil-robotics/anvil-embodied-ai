# Use ROS2 Jazzy as base
FROM ros:jazzy-ros-base

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

# Install Python dependencies FIRST (rarely changes → cached)
# lerobot pulls torch, opencv, etc. — use --ignore-installed to bypass Debian system pkgs
# Default: ACT + Diffusion. For VLA policies, set LEROBOT_EXTRAS build arg.
# Examples: "pi" (Pi0/Pi0-Fast), "groot", "xvla", "smolvla", or "pi,groot,xvla,smolvla"
ARG LEROBOT_EXTRAS=""
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed "numpy>=2.0.0,<2.3.0" "packaging>=24.2,<26.0" && \
    if [ -z "$LEROBOT_EXTRAS" ]; then \
        pip3 install --break-system-packages --no-cache-dir --ignore-installed "lerobot>=0.5.0"; \
    else \
        pip3 install --break-system-packages --no-cache-dir --ignore-installed "lerobot[$LEROBOT_EXTRAS]>=0.5.0"; \
    fi && \
    pip3 install --break-system-packages --no-cache-dir "setuptools<80"

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
