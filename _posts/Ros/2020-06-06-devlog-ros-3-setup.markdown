---
layout: post
title:  "[ROS] Ros 환경 설정하기 "
subtitle:   "Ros Setup"
categories: devlog
tags: ros
---

# ROS setup

설치과정에서 사용된 아래 명렁어처럼 환경 설정 파일을 불러오는 것은 새로운 터미널 창을 열때마다 매번 실행해줘야 합니다. 

이러한 번거로운 작업을 없애기 위하여 새로운 터미널 창을 열때마다 정해진 환경 설정 파일을 읽어오도록 설정해주도록 합시다.

또한, ROS 네트워크 설정 및 자주 사용하는 명령어를 단축 명령어로 설정해 봅시다.

```sh
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

우선 gedit 프로그램과 같은 tool을 사용하여 bashrc 파일을 수정해야 합니다. 

아래의 명령어로 bashrc 파일을 불러옵니다.

```sh
gedit ~/.bashrc
```

저같은 경우에는

```sh
# Set ROS Melodic
source /opt/ros/melodic/setup.bash
source ~/EscortRobot/catkin_ws/devel/setup.bash
source ~/YDLIDAR/catkin_ws/devel/setup.bash
source ~/SLAM/catkin_ws/devel/setup.bash

# Set ROS Network
# TurtleBot 구동 시 HOSTNAME은 터틀봇 ip로 수정해야 함
export ROS_MASTER_URI=http://192.168.0.26:11311
export ROS_HOSTNAME=http://192.168.0.26


# set ROS alias command
alias cw='cd ~/EscortRobot/catkin_ws'
alias cs='cd ~/EscortRobot/catkin_ws/src'
alias cm='cd ~/EscortRobot/catkin_ws && catkin_make'
```

이렇게 수정하였습니다. 

## ROS 네트워크 설정
ROS_MASTER_URI와 ROS_HOSTNAME의 설정입니다. 

ROS는 네트워크를 이용해 노드 간에 메시지 통신을 하기 때문에 이 설정은 매우 중요합니다.

우선은 둘 다 자신의 네트워크 IP를 입력해주면 되는데, 차후에 마스터 PC가 따로 있고 로봇은 호스트 PC를 사용하는 경우(Turtlebot의 라즈베리파이) 이를 구분하여 입력하면 다른 컴퓨터 간의 통신이 가능하게 됩니다.

## 참조

- http://wiki.ros.org/melodic/Installation/Ubuntu