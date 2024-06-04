# 从NVIDIA L4T基础镜像开始，与JetPack 4.6兼容的版本
FROM nvcr.io/nvidia/l4t-base:r32.6.1

# 避免在安装过程中被询问配置选项
ARG DEBIAN_FRONTEND=noninteractive

# 安装依赖
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
    gnupg2 \
    lsb-release \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    ca-certificates

# 设置环境变量
ENV ROS_DISTRO=melodic

# 安装ROS Melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
    
RUN apt-get update && apt-get install -y -o Dpkg::Options::="--force-overwrite" ros-melodic-desktop-full

RUN apt-get update && apt-get install -y python-rosdep python-catkin-pkg python-rosinstall python-rosinstall-generator python-wstool build-essential python3-setuptools

# 初始化 rosdep
RUN rosdep init && rosdep update

# 环境设置
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# 将你的ROS工作空间复制到容器中
COPY limo_ws /root/limo_ws

# 安装Python依赖
# 更换为Python3的rosdep、rosinstall等工具
RUN python -m pip install -U rosdep rosinstall_generator wstool rosinstall
RUN pip install -r /root/limo_ws/src/limo_policy/src/requirements.txt

# 安装cuDNN
COPY cudnn/include/cudnn*.h /usr/local/cuda/include/
COPY cudnn/lib64/libcudnn* /usr/local/cuda/lib64/
# 设置库文件路径环境变量，以便运行时能找到cuDNN库
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 安装tensorrt
WORKDIR /root
COPY tensorrt-8.2.1.8-cp38-none-linux_aarch64.whl /root/
RUN python -m pip install tensorrt-8.2.1.8-cp38-none-linux_aarch64.whl

# 安装ROS包的依赖并构建ROS包
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /root/limo_ws; rosdep install --from-paths src --ignore-src -r -y; catkin_make'

RUN chmod +x /root/limo_ws/src/limo_policy/src/*

# 清理安装包和缓存以减小镜像大小
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置容器启动时执行的命令
CMD ["/bin/bash", "-c", "source /opt/ros/melodic/setup.bash && \
                          cd /root/limo_ws && \
                          catkin_make && \
                          source devel/setup.bash && \
                          python src/limo_policy/src/onnx2engine.py && \
                          roslaunch limo_policy launch_test.launch"]
