FROM ubuntu:22.04 AS base

SHELL ["/bin/bash", "-c"]

ENV project=meeting-sdk-linux-sample
ENV cwd=/tmp/$project

WORKDIR $cwd

ARG DEBIAN_FRONTEND=noninteractive

#  Install Dependencies
RUN apt-get update  \
    && apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gdb \
    git \
    gfortran \
    libopencv-dev \
    libdbus-1-3 \
    libgbm1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    libssl-dev \
    libx11-dev \
    libx11-xcb1 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-xtest0 \
    libgl1-mesa-dri \
    libxfixes3 \
    linux-libc-dev \
    pkgconf \
    tar \
    unzip \
    zip

# Install ALSA
RUN apt-get install -y libasound2 libasound2-plugins alsa alsa-utils alsa-oss

# Install necessary compilers
RUN apt-get update && apt-get install -y \
    gcc-12 g++-12 clang-15

# Set GCC 12, G++ 12, and Clang 15 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100

RUN apt-get install -y pulseaudio pulseaudio-utils

# Copy the entire project into the container
COPY . /tmp/meeting-sdk-linux-sample

FROM base AS deps

ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

RUN apt-get update && apt-get install -y autoconf automake autoconf-archive ninja-build

# Set Environment Variables for Vcpkg to Use GCC 12
ENV CXX=/usr/bin/g++-12
ENV CC=/usr/bin/gcc-12

WORKDIR /opt
RUN git clone https://github.com/Microsoft/vcpkg.git \
    && cd vcpkg \
    && git checkout 23b33f5a010e3d67132fa3c34ab6cd0009bb9296 \
    && ./bootstrap-vcpkg.sh -disableMetrics \
    && ln -s /opt/vcpkg/vcpkg /usr/local/bin/vcpkg \
    && vcpkg install vcpkg-cmake boost-system boost-asio boost-log \
    && vcpkg install ada-url cli11 jwt-cpp websocketpp

FROM deps AS build

WORKDIR $cwd

# Set CMake to use vcpkg toolchain
ENV CMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake
ENV VCPKG_ROOT=/opt/vcpkg

ENTRYPOINT ["/tini", "--", "./bin/entry.sh"]
