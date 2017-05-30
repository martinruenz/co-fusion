#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup and installation of all project dependencies; and project building.
"""
import argparse
import os
import platform
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='0.1')
    parser.add_argument(
        '-a', '--all', dest='shouldInstallAll', action='store_true',
        help="full fresh installation; all other options will be ignored; "
             "will force re-installation (as when using --force) "
    )
    parser.add_argument(
        '-s', '--source', dest='shouldInstallFromSource', action='store_true',
        help="install dependencies to be built from source (in folder '/deps')"
    )
    parser.add_argument(
        '-p', '--apt-get', dest='shouldInstallAptPackages',
        action='store_true', help="install required apt-get deb packages"
    )
    parser.add_argument(
        '-b', '--build', dest='shouldBuildProject', action='store_true',
        help="create '/build' folder, prepare CMake, and build project"
    )
    parser.add_argument(
        '-f', '--force', dest='force', action='store_true',
        help="force re-installation of already installed dependencies"
    )
    parser.add_argument(
        '-u', '--update', dest='shouldUpdateFromSource',
        action='store_true', help="Update dependencies built from source."
    )

    parser.set_defaults(shouldInstallAll=False)
    parser.set_defaults(shouldInstallFromSource=False)
    parser.set_defaults(shouldInstallAptPackages=False)
    parser.set_defaults(shouldUpdateFromSource=False)
    parser.set_defaults(shouldBuildProject=False)
    parser.set_defaults(force=False)

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def prepareCMakeAndBuild(buildPath, depsPath):
    executeCommand('mkdir -p ' + buildPath)
    os.chdir(buildPath)

    exportEnvironmentVariables(depsPath)

    BOOST_ROOT = os.environ['BOOST_ROOT']
    Pangolin_DIR = os.environ['Pangolin_DIR']
    OpenCV_DIR = os.environ['OpenCV_DIR']

    cmakeCmd = (
        'cmake -DBOOST_ROOT="{BOOST_ROOT}"'
        ' -DOpenCV_DIR="{OpenCV_DIR}"'
        ' -DPangolin_DIR="{Pangolin_DIR}" ..'
        .format(
            BOOST_ROOT=BOOST_ROOT,
            OpenCV_DIR=OpenCV_DIR,
            Pangolin_DIR=Pangolin_DIR
        )
    )

    print(cmakeCmd.split())

    executeCommands([
        cmakeCmd,
        'make -j8'])


def installDebPackages():
    distro = platform.dist()[2]

    executeCommand(
        'sudo apt-get install -y software-properties-common '
        ' python-software-properties'
    )

    if distro == 'trusty':
        cmds = (
            'sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test',
            'sudo add-apt-repository -y ppa:george-edison55/cmake-3.x',
            'sudo add-apt-repository -y ppa:openjdk-r/ppa',
            ('sudo add-apt-repository -y '
                'ppa:v-launchpad-jochen-sprickerhof-de/pcl')
        )

        executeCommands(cmds)

        executeAptGetUpdate()

    elif distro == 'vivid':
        executeAptGetUpdate()

    elif distro == 'xenial':
        executeAptGetUpdate()

    else:
        print(distro, ' is not yet supported')
        return

    executeCommand(
        'sudo apt-get install -y  wget '
        'git '
        'build-essential '
        'cmake '
        'cmake-qt-gui '
        'freeglut3-dev '
        'g++-4.9 '
        'gcc-4.9 '
        'git '
        'libeigen3-dev '
        'libglew-dev '
        'libjpeg-dev '
        'libsuitesparse-dev '
        'libudev-dev '
        'libusb-1.0-0-dev '
        'openjdk-8-jdk '
        'unzip '
        'zlib1g-dev '
        'tar '
        'libopencv-dev '
        'python-opencv '
        'libopenni-dev '
        'libopenni2-dev'
    )


def installFromSource(path):
    distro = platform.dist()[2]

    executeCommand("mkdir -p " + path)
    os.chdir(path)

    # OpenCV contrib
    executeCommand('git clone --depth=1 --branch 3.1.0 '
                   'https://github.com/opencv/opencv_contrib.git'
                   )

    openCVContribPath = os.path.join(path, 'opencv_contrib')
    os.chdir(openCVContribPath)

    executeCommand('git pull')

    os.chdir(path)

    # CUDA
    if distro == 'trusty':
        cmds = [
            ('wget http://developer.download.nvidia.com/compute/cuda'
                '/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5'
                '-18_amd64.deb'),
            'sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb',
            'rm cuda-repo-ubuntu1404_7.5-18_amd64.deb',
            'sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test',
            'sudo add-apt-repository -y ppa:george-edison55/cmake-3.x',
            'sudo add-apt-repository -y ppa:openjdk-r/ppa'
        ]

        executeCommands(cmds)

        executeAptGetUpdate()
        executeCommand('sudo apt-get install -y cuda-7-5')

    elif distro == 'vivid':
        cmds = [
            ('wget http://developer.download.nvidia.com/compute/cuda'
                '/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5'
                '-18_amd64.deb'),
            'sudo dpkg -i cuda-repo-ubuntu1504_7.5-18_amd64.deb',
            'rm cuda-repo-ubuntu1504_7.5-18_amd64.deb'
        ]

        executeCommands(cmds)
        executeAptGetUpdate()
        executeCommand('sudo apt-get install -y cuda-7-5')

    elif distro == 'xenial':
        cmds = [
            ('wget http://developer.download.nvidia.com/compute/cuda'
                '/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44'
                '-1_amd64.deb'),
            'sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb',
            'rm cuda-repo-ubuntu1604_8.0.44-1_amd64.deb'
        ]

        executeCommands(cmds)
        executeAptGetUpdate()
        executeCommand('sudo apt-get install -y cuda-8-0')

    else:
        print(distro, ' is not yet supported')
        return

    # Switching to alternative installations of g++ and java
    if distro == 'trusty':
        cmds = [
            ('sudo update-alternatives --install '
                '/usr/bin/gcc gcc '
                '/usr/bin/gcc-4.9 60 '
                '--slave /usr/bin/g++ g++ '
                '/usr/bin/g++-4.9'),
            'sudo update-java-alternatives -s java-1.8.0-openjdk-amd64'
        ]

        executeCommands(cmds)

    # OpenCV

    cmds = [
        'wget https://github.com/Itseez/opencv/archive/3.1.0.zip',
        'unzip 3.1.0.zip',
        'rm 3.1.0.zip'
    ]

    executeCommands(cmds)

    openCVPath = os.path.join(path, 'opencv-3.1.0')
    os.chdir(openCVPath)

    executeCommand('mkdir -p build')

    openCVBuildPath = os.path.join(openCVPath, 'build')
    os.chdir(openCVBuildPath)

    modulesPath = os.path.join(openCVContribPath, 'modules')
    installPath = os.path.join(openCVPath, 'install')
    cmds = [
        ('cmake'
            ' -DCMAKE_BUILD_TYPE=Release'
            ' -DCMAKE_INSTALL_PREFIX=' + installPath +
            ' -DOPENCV_EXTRA_MODULES_PATH=' + modulesPath +
            ' -DBUILD_opencv_flann=ON'
            ' -DWITH_CUDA=OFF'
            ' -DBUILD_DOCS=OFF'
            ' -DBUILD_TESTS=OFF'
            ' -DBUILD_PERF_TESTS=OFF'
            ' -DBUILD_opencv_java=OFF'
            ' -DBUILD_opencv_python2=OFF'
            ' -DBUILD_opencv_python3=OFF'
            ' -DBUILD_opencv_features2d=ON'
            ' -DBUILD_opencv_calib3d=ON'
            ' -DBUILD_opencv_objdetect=ON'
            ' -DBUILD_opencv_stitching=OFF'
            ' -DBUILD_opencv_superres=OFF'
            ' -DBUILD_opencv_shape=OFF'
            ' -DWITH_1394=OFF'
            ' -DWITH_GSTREAMER=OFF'
            ' -DWITH_GPHOTO2=OFF'
            ' -DWITH_MATLAB=OFF'
            ' -DWITH_TIFF=OFF'
            ' -DWITH_VTK=OFF'
            ' -DBUILD_opencv_surface_matching=ON'
            ' -DBUILD_opencv_aruco=OFF'
            ' -DBUILD_opencv_bgsegm=OFF'
            ' -DBUILD_opencv_bioinspired=OFF'
            ' -DBUILD_opencv_ccalib=OFF'
            ' -DBUILD_opencv_contrib_world=OFF'
            ' -DBUILD_opencv_datasets=OFF'
            ' -DBUILD_opencv_dnn=OFF'
            ' -DBUILD_opencv_dpm=OFF'
            ' -DBUILD_opencv_face=OFF'
            ' -DBUILD_opencv_fuzzy=OFF'
            ' -DBUILD_opencv_line_descriptor=OFF'
            ' -DBUILD_opencv_matlab=OFF'
            ' -DBUILD_opencv_optflow=OFF'
            ' -DBUILD_opencv_plot=OFF'
            ' -DBUILD_opencv_reg=OFF'
            ' -DBUILD_opencv_rgbd=OFF'
            ' -DBUILD_opencv_saliency=OFF'
            ' -DBUILD_opencv_stereo=OFF'
            ' -DBUILD_opencv_structured_light=OFF'
            ' -DBUILD_opencv_text=OFF'
            ' -DBUILD_opencv_tracking=OFF'
            ' -DBUILD_opencv_xfeatures2d=OFF'
            ' -DBUILD_opencv_ximgproc=OFF'
            ' -DBUILD_opencv_xobjdetect=OFF'
            ' -DBUILD_opencv_xphoto=OFF'
            ' ..'),
        'make -j8',
        'make install'
    ]

    executeCommands(cmds)

    os.chdir(path)

    # BOOST
    cmds = [
        ('wget -O boost_1_62_0.tar.bz2 '
            'https://sourceforge.net/projects/boost/files/boost/1.62.0'
            '/boost_1_62_0.tar.bz2/download'),
        'tar -xjf boost_1_62_0.tar.bz2',
        'rm boost_1_62_0.tar.bz2'
    ]

    executeCommands(cmds, stdout=subprocess.DEVNULL)

    boostPath = os.path.join(path, 'boost_1_62_0')
    os.chdir(boostPath)

    cmds = [
        'mkdir -p ../boost',
        './bootstrap.sh --prefix=../boost',
        './b2 --prefix=../boost --with-filesystem install'
    ]

    executeCommands(cmds, stdout=subprocess.DEVNULL)

    os.chdir(path)

    executeCommand('rm -r boost_1_62_0')

    os.chdir(path)

    # Pangolin

    cmds = [
        ('git clone --depth=1 --branch devel '
         'https://github.com/martinruenz/Pangolin.git'),
        'mkdir -p Pangolin/build'
    ]

    executeCommands(cmds)

    # Dense crf
    executeCommand(
        "git clone --depth=1 https://github.com/martinruenz/densecrf.git"
    )

    # gSLICr
    executeCommand('git clone --depth=1 https://github.com/carlren/gSLICr.git')

    # Update and install OpenCV contrib, Pangolin, Dense crf, gSLICr
    updateFromSource(path)


def updateFromSource(path):
    exportEnvironmentVariables(path)

    OpenCV_DIR = os.environ['OpenCV_DIR']

    CMAKE_CXX_FLAGS = (
        os.environ['CMAKE_CXX_FLAGS']
        if 'CMAKE_CXX_FLAGS' in os.environ
        else ""
    )

    # OpenCV contrib
    openCVContribPath = os.path.join(path, 'opencv_contrib')
    os.chdir(openCVContribPath)
    executeCommand('git pull')
    os.chdir(path)

    # Pangolin
    pangolinPath = os.path.join(path, 'Pangolin')
    os.chdir(pangolinPath)

    cmds = [
        'git pull',
        'mkdir -p build'
    ]

    executeCommands(cmds)

    pangolinBuildPath = os.path.join(pangolinPath, 'build')
    os.chdir(pangolinBuildPath)

    cmds = [
        'cmake -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON ..',
        'make -j8'
    ]

    executeCommands(cmds)

    os.chdir(path)

    # Dense crf
    densecrfPath = os.path.join(path, 'densecrf')
    os.chdir(densecrfPath)

    cmds = [
        'git pull',
        'mkdir -p build'
    ]

    executeCommands(cmds)

    densecrfBuildPath = os.path.join(densecrfPath, 'build')
    os.chdir(densecrfBuildPath)

    cmakeCmd = (
        'cmake -DCMAKE_BUILD_TYPE=Release '
        '-DCMAKE_CXX_FLAGS="{CMAKE_CXX_FLAGS} -fPIC" ..'
        .format(
            CMAKE_CXX_FLAGS=CMAKE_CXX_FLAGS
        )
    )

    executeCommand(cmakeCmd)

    executeCommand('make -j8')

    executeCommands(cmds)

    os.chdir(path)

    # gSLICr
    gSLICrPath = os.path.join(path, 'gSLICr')
    os.chdir(gSLICrPath)

    cmds = [
        'git pull',
        'mkdir -p build'
    ]

    executeCommands(cmds)

    gSLICrBuildPath = os.path.join(gSLICrPath, 'build')
    os.chdir(gSLICrBuildPath)

    cmakeCmd = (
        'cmake '
        '-DCMAKE_BUILD_TYPE=Release '
        '-DCUDA_HOST_COMPILER=/usr/bin/gcc-4.9 '
        '-DCMAKE_CXX_FLAGS="{CMAKE_CXX_FLAGS} -D_FORCE_INLINES" '
        '..'
        .format(
            OpenCV_DIR=OpenCV_DIR,
            CMAKE_CXX_FLAGS=CMAKE_CXX_FLAGS
        )
    )

    executeCommand(cmakeCmd)

    os.system('cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCUDA_HOST_COMPILER=/usr/bin/gcc-4.9 \
      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -D_FORCE_INLINES" \
      ..')

    cmds = [
        'make clean',
        'make -j8'
    ]

    executeCommands(cmds)

    os.chdir(path)


def exportEnvironmentVariables(depsPath):
    BOOST_ROOT = (
        os.environ['BOOST_ROOT']
        if 'BOOST_ROOT' in os.environ
        else os.path.join(depsPath, 'boost')
    )

    print(BOOST_ROOT)

    if os.path.exists(BOOST_ROOT):
        os.environ["BOOST_ROOT"] = BOOST_ROOT
        print('BOOST_ROOT set and exported')
    else:
        print('BOOST_ROOT path (in "./deps") not found. Variable not set and '
              'exported')

    Pangolin_DIR = (
        os.environ['Pangolin_DIR']
        if 'Pangolin_DIR' in os.environ
        else os.path.join(depsPath, 'Pangolin/build')
    )

    if os.path.exists(Pangolin_DIR):
        os.environ["Pangolin_DIR"] = Pangolin_DIR
        print('Pangolin_DIR set and exported')
    else:
        print('Pangolin_DIR path (in "./deps") not found.'
              ' Variable not set and exported')

    OpenCV_DIR = (
        os.environ['OpenCV_DIR']
        if 'OpenCV_DIR' in os.environ
        else os.path.join(depsPath, 'opencv-3.1.0/install')
    )

    if os.path.exists(OpenCV_DIR):
        os.environ["OpenCV_DIR"] = OpenCV_DIR
        print('OpenCV_DIR set and exported')
    else:
        print('OpenCV_DIR path (in "./deps") not found. Variable not set and '
              'exported')


def main(args):
    # ============================= Initialisation ============================

    # Getting parsed arguments
    shouldInstallAll = args.shouldInstallAll

    if shouldInstallAll:
        shouldInstallFromSource = True
        shouldInstallAptPackages = True
        shouldBuildProject = True
        force = True
        shouldUpdateFromSource = False

    else:
        shouldInstallFromSource = args.shouldInstallFromSource
        shouldInstallAptPackages = args.shouldInstallAptPackages
        shouldBuildProject = args.shouldBuildProject
        force = args.force
        shouldUpdateFromSource = args.shouldUpdateFromSource

    # Getting required paths
    previousPath = os.path.abspath(__file__)
    previousPath = os.path.dirname(previousPath)

    path = os.path.join(previousPath, '..')
    path = os.path.abspath(path)

    depsPath = os.path.join(path, 'deps')

    os.chdir(path)

    # ============================= Initialisation ============================
    print("  Co-fusion setup and installation   ")
    print("=====================================")
    print("Please enter SUDO password when asked")
    print()

    # ====================== Installation of Deb packages =====================
    if shouldInstallAptPackages:
        print("Installation of apt-get packages...")
        installDebPackages()

    # ======================== Installation from source =======================
    if shouldInstallFromSource:
        if (os.path.exists(depsPath)):
            if not force:
                print(
                    "Folder '/deps' already exists and dependencies won't be "
                    "re-installed"
                )
                print("Use option --force to re-install dependencies")
            else:
                print("Forcing re-installation of dependencies from source...")
                executeCommand("rm -rf deps")
                installFromSource(depsPath)
                os.chdir(path)
        else:
            print("Installation of dependencies from source...")
            installFromSource(depsPath)
            os.chdir(path)

    # =========================== Update from source ==========================
    if shouldUpdateFromSource:
        print("Updating and building dependencies from source...")
        updateFromSource(depsPath)
        os.chdir(path)

    # ============================== VSLAM CMake ==============================
    if shouldBuildProject:
        print(
            "Creating build folder, preparing CMake, and building project...")
        buildPath = os.path.join(path, 'build')
        prepareCMakeAndBuild(buildPath, depsPath)

    os.chdir(previousPath)


def executeAptGetUpdate():
    cmds = [
        'sudo apt-get clean',
        'sudo rm -rf /var/cache/apt/*',
        'sudo rm -rf /var/lib/apt/lists/*',
        'sudo apt-get update'
    ]

    executeCommands(
        cmds,
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL
    )


def executeCommand(command, stderr=None, stdout=None):
    """ Raises an error when a command fails. """
    print("$ {}".format(command))
    subprocess.check_call(command, stderr=stderr,
                          stdout=stdout, env=dict(os.environ), shell=True)


def executeCommands(commands, stderr=None, stdout=None):
    for command in commands:
        executeCommand(command, stderr=stderr, stdout=stdout)


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
