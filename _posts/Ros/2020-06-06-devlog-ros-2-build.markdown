---
layout: post
title:  "[ROS] Ros 작업폴더 생성 후 Build 해보기 "
subtitle:   "Ros Build"
categories: devlog
tags: ros
---

## Creating a woring environment for ROS

본격적으로 ROS를 시작하기 전에 working space를 생성해 봅시다.

ROS에서는 catkin 이라는 ROS 전용 빌드 시스템을 사용하고 있습니다. 
이를 사용하기위해서는 아래와 같이 catkin 작업 폴더 및 작업 폴더 초기화 설정을 해주어야 합니다.

```sh
mkdir -p ~/catkin_ws/src

cd ~/catkin_ws/src

catkin_init_workspace
```

꼭 이렇게 할 필요는 없습니다.
저는 쓰임에 맞추어서 디렉토리를 만든 후 그 디렉토리 안에 작업 디렉토리를 생성합니다.

가령 Rviz를 통해 라이다 센서가 잘 동작하는지 확인하고 싶다면

```sh
mkdir LIDAR && cd LIDAR

mkdir -p catkin_ws/src

cd catkin_ws/src

catkin_init_workspace
```

이런 식으로 작업 디렉토리를 생성할 수 있을 것 입니다.

작업 디렉토리를 성공적으로 생성하였으면 컴파일을 합니다. 

현재의 catkin 작업 폴더에는 src폴더 및 그 안의 CMakeLists.txt 이외에 아무런 파일이 없지만 테스트삼아 아래와 같이 “catking_make”명령어를 이용하여 빌드할 것 입니다.

주의할 점은 catkin_make 명령어를 이용한 빌드는 항상 catkin_ws 폴더 안에서 이루어 진다는 것 입니다.

```sh
cd ..

catkin_make
```

문제없이 빌드를 마치게 되면 아래와 같이 ls 명령어를 실행해 봅시다. 

유저가 직접 생성하였던 “src” 폴더 이외의 없었던 “build” 및 “devel”폴더가 새로 생성되었습니다. 

catkin 빌드 시스템의 빌드 관련 파일은 “build” 폴더에, 빌드 후 실행관련 파일은 “devel” 에 저장되게 됩니다.

```sh
ls

build  devel  src
```

마지막으로, catkin 빌드 시스템과 관련된 환경 파일을 불러오면 됩니다.

```sh
source ~/catkin_ws/devel/setup.bash
```

## 참조

- http://wiki.ros.org/catkin/Tutorials/create_a_workspace