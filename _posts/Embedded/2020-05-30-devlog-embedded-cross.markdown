---
layout: post
title:  "[Raspberry Pi] 라즈베리파이와 우분투에서의 크로스 컴파일 환경을 구축해보자"
subtitle:   "Cross Compile"
categories: devlog
tags: embedded
---

## Cross Compile이란?

과거부터 임베디드 리눅스를 개발할 경우 성능과 환경의 문제로 실제 개발은 호스트 PC에서 진행을 하고, 만들어진 바이너리를 타깃(임베디드 보드)에 올려 실행 및 테스트를 하곤 합니다.
이를 '크로스 컴파일'이라고 합니다. 

라즈베리파이에서 간단한 프로그램은 빌드가 가능하지만 프로그램의 덩치가 커지면 데스크탑에서 빌드후 소스를 라즈베리로 복사하여 실행하는 방법을 많이 이용합니다. 
<br/>
<br/>

아래 내용에서는 라즈베리파이 3와 Ubuntu 16.04환경에서 크로스컴파일 환경을 설정하고 프로젝트를 빌드하는 과정 예시를 소개합니다.

먼저 우분투에서 라즈베리파이 어플레이케이션의 컴파일을 위해서 크로스 컴파일러가 필요합니다. 
터미널에서 크로스 컴파일러를 다운로드 해봅시다. 

```sh
sudo apt-get update

#크로스 컴파일러 및 tool 설치
sudo apt-get install git gcc-arm-linux-gnueabihf make ncurses-dev build-essential
```
<br/>
라즈비안에서 컴파일러의 경우, 'arm-linux-gnueabihf'라는 arm용 gcc 컴파일러를 사용합니다.
라즈베리파이에서 돌아갈 프로그램(리눅스 커널 포함)은 arm에 맞는 바이너리로 빌드를 해야 하기 때문에 arm용 gcc가 필요합니다.
<br/>



이후 간단한 예제 프로그램을 작성해봅시다.

```sh
vi hello.c
```

```c
#include <stdio.h>

int main(void) {
    printf("Hello World!\n");
    return 0;
}
```
<br/>

라즈베리파이에서 사용할 수 있도록 크로스 컴파일을 진행합니다. 
```sh
arm-linux-gnueabihf-gcc -o hello hello.c
```
<br/>

라즈베리파이에서 실행이 잘되는 지 확인하기 위해서 라즈베리파이의 SD카드에 예제 프로그램을 복사해야 합니다.<br/>
먼저 가상 머신에 연결된 블록 디바이스 확인 해 봅시다.<br/>
아래 명령어를 통해서 현재 리눅스 pc에 연결된 디바이스를 확인할 수 있습니다.

```sh
lsblk
```

![Imgur](https://i.imgur.com/MtJyKFN.png)

<br/>
<br/>
 sdb에 라즈베리파이 sd카드의 용량인 16G만큼의 용량이 할당되어 있는 것을 확인할 수 있습니다. 
 <br/>
새롭게 추가된 sdb가 SD카드이고<br/> 
• sdb1 : 라즈베리파이 커널<br/>
• sdb2 : 라즈베리파이 파일 시스템<br/>

입니다.

우분투에 SD카드를 마운트 합니다. 

```sh
 sudo mkdir /mnt/kernel
 sudo mount /dev/sdb1 /mnt/kernel
 sudo mkdir /mnt/fs
 sudo mount /dev/sdb2 /mnt/fs
```

<br/>
<br/>
이제 라즈베리파이 SD카드에 좀 전에 만들었던 예제 프로그램을 복사 하면 됩니다. 

```sh
cp hello /mnt/fs/home/pi
```
 
 ![Imgur](https://i.imgur.com/txNC3Rh.png)
 <br/>
<br/>

이제 SD카드를 언마운트 하고 라즈베리파이에 연결해서 확인하면 끝입니다. 
```sh
sudo umount /mnt/fs
sudo umount /mnt/kernel
```
<br/>

아래는 라즈베리파이 에서의 실행 화면 입니다. 
<br/>
<br/>

![Imgur](https://i.imgur.com/AkSYGvU.png)

<br/>
<br/>
<br/>

 성공적으로 빌드가 잘 되는 모습을 확인하였습니다!<br/>
 다음에는 라즈베리파이에서의 시스템콜을 실습해보겠습니다. 
