#!/bin/bash -e
#
# This is a build script for VisualSLAM.
#
# To install all dependencies as well, invoke:
#   ./build.sh --fresh-install
#
#   which will create:
#   - ./deps/densecrf
#   - ./deps/gSLICr
#   - ./deps/OpenNI2
#   - ./deps/Pangolin
#   - ./deps/opencv-3.1.0
#   - ./deps/opencv_contrib
#   - ./deps/boost (unless env BOOST_ROOT is defined)

cd $(dirname `realpath $0`)/..
mkdir -p deps
cd deps

if [ "${1:-}" = "--fresh-install" ]; then
  # add cuda and other necessary upgrades
  sudo apt-get install -y wget software-properties-common
  source /etc/lsb-release # fetch DISTRIB_CODENAME
  if [[ $DISTRIB_CODENAME == *"trusty"* ]] ; then
    # CUDA
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
    # g++ 4.9.4
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    # cmake 3.2.2
    sudo add-apt-repository -y ppa:george-edison55/cmake-3.x
    # openjdk 8
    sudo add-apt-repository -y ppa:openjdk-r/ppa
    sudo apt-get update > /dev/null
    sudo apt-get install -y cuda-7-5
  elif [[ $DISTRIB_CODENAME == *"vivid"* ]] ; then
    # CUDA
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5-18_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1504_7.5-18_amd64.deb
    rm cuda-repo-ubuntu1504_7.5-18_amd64.deb
    sudo apt-get update > /dev/null
    sudo apt-get install cuda-7-5
  elif [[ $DISTRIB_CODENAME == *"xenial"* ]]; then
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    rm cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo apt-get update > /dev/null
    sudo apt-get install -y cuda-8-0
  else
    echo "$DISTRIB_CODENAME is not yet supported"
    exit 1
  fi

  sudo apt-get install -y \
    build-essential \
    cmake \
    cmake-qt-gui \
    freeglut3-dev \
    g++-4.9 \
    gcc-4.9 \
    git \
    zeroc-ice35 \
    libeigen3-dev \
    libglew-dev \
    libjpeg-dev \
    libsuitesparse-dev \
    libudev-dev \
    libusb-1.0-0-dev \
    openjdk-8-jdk \
    unzip \
    zlib1g-dev

  if [[ $DISTRIB_CODENAME == *"trusty"* ]] ; then
     # switch to g++-4.9
     sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9
     # switch to java-1.8.0
     sudo update-java-alternatives -s java-1.8.0-openjdk-amd64
  fi

  git clone --depth=1 --branch devel https://github.com/martinruenz/Pangolin.git
  git clone --depth=1 https://github.com/occipital/OpenNI2.git
  git clone --depth=1 https://github.com/opencv/opencv_contrib.git
  git clone --depth=1 https://github.com/martinruenz/densecrf.git
  git clone --depth=1 https://github.com/carlren/gSLICr.git

  if [ ! -d opencv-3.1.0/install ]; then
    wget https://github.com/Itseez/opencv/archive/3.1.0.zip
    unzip 3.1.0.zip
    rm 3.1.0.zip
    cd opencv-3.1.0
    mkdir -p build
    cd build
    cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
      -DOPENCV_EXTRA_MODULES_PATH="`pwd`/../../opencv_contrib/modules" \
      \
      `# OpenCV: (building is not possible when DBUILD_opencv_video/_videoio is OFF?)` \
      -DBUILD_opencv_flann=ON  \
      -DWITH_CUDA=OFF  \
      -DBUILD_DOCS=OFF  \
      -DBUILD_TESTS=OFF  \
      -DBUILD_PERF_TESTS=OFF  \
      -DBUILD_opencv_java=OFF  \
      -DBUILD_opencv_python2=OFF  \
      -DBUILD_opencv_python3=OFF  \
      -DBUILD_opencv_features2d=ON  \
      -DBUILD_opencv_calib3d=ON  \
      -DBUILD_opencv_objdetect=ON  \
      -DBUILD_opencv_stitching=OFF  \
      -DBUILD_opencv_superres=OFF  \
      -DBUILD_opencv_shape=OFF  \
      -DWITH_1394=OFF  \
      -DWITH_GSTREAMER=OFF  \
      -DWITH_GPHOTO2=OFF  \
      -DWITH_MATLAB=OFF  \
      -DWITH_TIFF=OFF  \
      -DWITH_VTK=OFF  \
      \
      `# OpenCV-Contrib:` \
      -DBUILD_opencv_surface_matching=ON \
      -DBUILD_opencv_aruco=OFF \
      -DBUILD_opencv_bgsegm=OFF \
      -DBUILD_opencv_bioinspired=OFF \
      -DBUILD_opencv_ccalib=OFF \
      -DBUILD_opencv_contrib_world=OFF \
      -DBUILD_opencv_datasets=OFF \
      -DBUILD_opencv_dnn=OFF \
      -DBUILD_opencv_dpm=OFF \
      -DBUILD_opencv_face=OFF \
      -DBUILD_opencv_fuzzy=OFF \
      -DBUILD_opencv_line_descriptor=OFF \
      -DBUILD_opencv_matlab=OFF \
      -DBUILD_opencv_optflow=OFF \
      -DBUILD_opencv_plot=OFF \
      -DBUILD_opencv_reg=OFF \
      -DBUILD_opencv_rgbd=OFF \
      -DBUILD_opencv_saliency=OFF \
      -DBUILD_opencv_stereo=OFF \
      -DBUILD_opencv_structured_light=OFF \
      -DBUILD_opencv_text=OFF \
      -DBUILD_opencv_tracking=OFF \
      -DBUILD_opencv_xfeatures2d=OFF \
      -DBUILD_opencv_ximgproc=OFF \
      -DBUILD_opencv_xobjdetect=OFF \
      -DBUILD_opencv_xphoto=OFF \
      ..
    make -j8
    make install > /dev/null
    cd ../install
    OpenCV_DIR=$(pwd)
    cd ../..
  fi

  if [ -z "${BOOST_ROOT}" -a ! -d boost ]; then
    wget -O boost_1_62_0.tar.bz2 https://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.tar.bz2/download
    tar -xjvf boost_1_62_0.tar.bz2 > /dev/null
    rm boost_1_62_0.tar.bz2
    cd boost_1_62_0
    mkdir -p ../boost
    ./bootstrap.sh --prefix=../boost
    ./b2 --prefix=../boost --with-filesystem install > /dev/null
    cd ..
    rm -r boost_1_62_0
    BOOST_ROOT=$(pwd)/boost
  fi

fi

if [ -z "${OpenCV_DIR}" -a -d opencv-3.1.0/install ]; then
  OpenCV_DIR=$(pwd)/opencv-3.1.0/install
fi

if [ -z "${BOOST_ROOT}" -a -d boost ]; then
  BOOST_ROOT=$(pwd)/boost
fi

# build pangolin
cd Pangolin
git pull
mkdir -p build
cd build
cmake -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON ..
make -j8
Pangolin_DIR=$(pwd)
cd ../..

# build OpenNI2
cd OpenNI2
git pull
make -j8
cd ..

# build DenseCRF, see: http://graphics.stanford.edu/projects/drf/
cd densecrf
git pull
mkdir -p build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fPIC" \
  ..
make -j8
cd ../..

# build gSLICr, see: http://www.robots.ox.ac.uk/~victor/gslicr/
cd gSLICr
git pull
mkdir -p build
cd build
cmake \
  -DOpenCV_DIR="${OpenCV_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_HOST_COMPILER=/usr/bin/gcc-4.9 \
  -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -D_FORCE_INLINES" \
  ..
make -j8
cd ../..

# build VisualSLAM
cd ..
mkdir -p build
cd build
cmake \
  -DBOOST_ROOT="${BOOST_ROOT}" \
  -DOpenCV_DIR="${OpenCV_DIR}" \
  -DPangolin_DIR="${Pangolin_DIR}" \
  ..
make -j8
cd ..

