---
layout: post
title:  "[ROS] Ubuntu 18.04에 Ros Melodic 설치하기 "
subtitle:   "Ros Installation"
categories: devlog
tags: ros
---

# Installation ROS on Ubuntu 18.04

이 글은 우분투18.04에서 Ros Melodic을 설치하는 내용을 다룹니다.

우분투에 ROS 소프트웨어 패키지 리파지토리와 키를 등록 합니다.

```sh
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
```

패키지 관리자를 업데이트 합니다.

```sh
sudo apt update
```

우분투 18.04에는 ROS melodic를 설치 할 수 있고, 우분투 16.04에는 ROS kinetic을 설치 할 수 있습니다. 2020년 2월 현재 ROS melodic가 모든 패키지를 지원하지는 않습니다.

```sh
# Ubuntu 18.04
sudo apt install ros-melodic-desktop-full
```

ROS 의존성을 자동으로 해결 하도록 합니다.

```sh
sudo rosdep init
```

출력되는 메시지를 따라서 업데이트 합니다.

```sh
rosdep update
```

ros의 다양한 패키지를 인스톨하는 프로그램 입니다. 빈번하게 사용할 정도로 유용한 툴인만큼 꼭 설치하도록 합시다.

```sh
sudo apt-get install python-rosinstall
```

환경 변수를 등록 합니다.

```sh
# melodic
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 참조

- http://wiki.ros.org/melodic/Installation/Ubuntu