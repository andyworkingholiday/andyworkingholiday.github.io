---
layout: post
title:  "[Intel Realsense] D435, T265 Intel RealSense 카메라 사용해보기"
subtitle:   "Intel Realsense Setup"
categories: devlog
tags: linux
---

## Intel RealSense SDK 2.0 Installation


인텔 리얼센스는 인텔에서 개발한 3D 카메라 기술로 사용자와 사용자의 배경을 하나의 평면 이미지로 감지하는 이전 세대의 2D 카메라와 달리 3차원 공간 그 자체를 인지하고 깊이감을 이해할 수 있는 기술입니다.


인텔이라 하면 반도체 전문회사라 생각할 수도 있으나 최근 스마트폰과 그 외 모바일 디바이스 시장의 빠른 확장으로 상대적으로 PC시장이 침체되기 시작하자 사물인터넷으로 사업 분야를 확장하여 컴퓨팅 기술을 복합적으로 생활 영역 구석구석에 적용하려는 시도를 하였고 인텔 리얼센스는 그 일환이라고 할 수 있습니다.

마침 현재 근무하고 있는 ETRI에서 박사님께서 갖고 놀아보라고  인텔 리얼센스 카메라 2가지 기종인 D435, T265를  두개 다 주셨습니다. 
먼저 우분투에서 카메라가 잘 작동하는지 확인하고 추후에 라즈베리파이4, JETSON NANO 등 보드에 적용시킬 예정입니다.

이제 인텔이 만든 이 신기한 장난감을 갖고 놀아봅시다. 

설치 관련해서는 [인텔 사이트]에 있는 명령어를 그대로 따라하면 됩니다. 저는 우분투 18.04를 사용하고 있기 때문에 linux용 설치법을 갖고왔습니다.

```sh
#Add Intel server to the list of repositories :
echo 'deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main' | sudo tee /etc/apt/sources.list.d/realsense-public.list

/etc/apt/sources.list.d/realsense-public.list

#Register the server’s public key :
sudo apt-key adv --keyserver keys.gnupg.net --recv-key 6F3EFCDE

#Refresh the list of repositories and packages available :
sudo apt-get update
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils

#Developers shall install additional packages:
sudo apt-get install librealsense2-dev
sudo apt-get install librealsense2-dbg
```
<br/>
이후 

```sh
realsense-viewer
```

명령어를 통해 테스트를 해봅시다. 
<br/>


## Realsense Camera Test

먼저 D435 모델을 테스트 해보았습니다.

설치하자마자 realsense-viewer를 실행하시지 마시고 다시 usb를 재연결하시고 실행시키면 됩니다.
<br/>

![intel-435](https://i.ibb.co/GQwQYVd/435-1.png)

라즈베리파이 카메라 같은 걸 쓰다보니 확실히 화질이 좋은게 느껴졌습니다. 

Depth view를 활성화 시키면 열화상 카메라 처럼 보이는 화면이 나타나는데 열감지는 절대 아니고 파란색일 수록 가까운거리 이고 빨간색일 수록 먼 거리를 감지 합니다. 
<br/>
![intel-435-2](https://i.ibb.co/m5ZRfZs/435-2.png)

또한 우측 상단의 3D 버튼을 클릭하면  카메라의 2D 정보와 공간정보를 Overlay해서 3차원 텍스쳐 이미지를 만들어냅니다. 정말 대단합니다...
이후 T265 모델도 테스트 해 보았습니다.
 <br/>
 
![intel-265](https://i.ibb.co/NTzVL4Z/265-1.png)

T265는 마치 잡입할 때 쓰는 카메라처럼 렌즈가 휘어있고 흑백이였습니다. 
또한 Pose Estimation이 D435보다 훨씬 유연한 느낌을 받았습니다.
 <br/>
 
 ![intel-265](https://i.ibb.co/zn1pPhZ/265-2.png)
 
 3D버튼을 클릭하면 카메라의 이동에 따라 실선이 그려지는 것을 확인하였습니다.
 확실히 IMU센서가 위치 이동을 잘 잡아내는 느낌입니다. 하지만 이리저리 흔들다가 원래의 위치로 돌아오면 실제의 위치와는 다른 곳을 가르켰습니다. 
 
 <br/>
 <br/>
 <br/>
 추후에는  ROS에서 SLAM 연동 및 OpenCV를 통해서 객체인식을 해볼 예정입니다!
 
 

[인텔 사이트]: https://www.intelrealsense.com/sdk-2/
