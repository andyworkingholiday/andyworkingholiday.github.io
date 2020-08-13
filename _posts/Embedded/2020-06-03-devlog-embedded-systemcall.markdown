---
layout: post
title:  "[Raspberry Pi] System Call을 통해 라즈베리파이에의 커널 프로그래밍을 실습해보자"
subtitle:   "System Call"
categories: devlog
tags: embedded
---

## System Call

- 시스템 콜은 응용프로그램에서 운영체제에게 어떠한 기능을 수행해 달라고 하는 하나의 수단

- 운영 체제의 커널이 제공하는 서비스에 대해, 응용 프로그램의 요청에 따라 커널에 접근 하기 위한 인터페이스
<br/>
<br/>

## System Call의 기능 
- 사용자 모드에 있는 응용 프로그램이 커널의 기능을 사용할 수 있도록 한다.

- 시스템 호출을 하면 사용자 모드에서 커널 모드로 바뀐다,

- 커널에서 시스템 호출을 처리하면 커널 모드에서 사용자 모드로 돌아가 작업을 계속한다.

<br/>
<br/>

## Kernel Cross Compile 

아래 내용에서는 라즈베리파이 3와 Ubuntu 16.04환경에서 크로스컴파일을 통해 라즈베리파이3에서의 시스템 콜을 다룹니다. 다만 커널 프로그래밍 시 다음을 주의해야 할 필요가 있습니다. 

- Kernel은 하드웨어 접근에 대해 어떠한 제한도 없기 때문에 커널에서의 오류는 시스템
에 치명적인 결과를 발생시킴 (Kernel panic)

- 함수 호출 등의 작업시 모든 에러 코드를 검사하고 처리해야 함

- 커널이 사용하는 stack의 크기는 제한되어 있고, 인터럽트 핸들러도 동일한 스택을 사용
하므로 큰 배열을 사용하거나, recursion이 많이 일어나지 않도록 주의

또한 실수 연산이나 MMX 연산을 사용할 수 없습니다.

System Call에 처리 함수를 등록하고 System Call 처리 함수를 구현해보겠습니다. 
그리고 Kernel을 재 컴파일 한 후 라즈베리파이 system에 적용시켜봅시다!
<br/>
<br/>
<br/>
<br/>

먼저 커널 소스를 다운로드하고 커널 컴파일 환경을 하기위해 환경을 구축합시다. 

```sh
# Download kernel source
ubuntu@ubuntu:~ $ mkdir working; cd working
ubuntu@ubuntu:~/working $ git clone -b rpi-4.4.y --depth=1 https://github.com/raspberrypi/linux
ubuntu@ubuntu:~/working $ cd linux
# Kernel compile configuration
ubuntu@ubuntu:~/working/linux $ make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- \
bcm2709_defconfig
# Kernel compile
ubuntu@ubuntu:~/working/linux $ make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- zImage –j4
# Module compile
ubuntu@ubuntu:~/working/linux $ make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- modules \
–j4
# DTB(Device Tree Blob) compile
ubuntu@ubuntu:~/working/linux $ make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- dtbs –j4
# modules_install
ubuntu@ubuntu:~/working/linux $ sudo make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- \
INSTALL_MOD_PATH=../modules modules_install
ubuntu@ubuntu:~/working/linux $ cd ../modules
ubuntu@ubuntu:~/working/modules $ sudo rm lib/modules/4.4.50-v7+/build
ubuntu@ubuntu:~/working/modules $ sudo rm lib/modules/4.4.50-v7+/source
```
<br/>
<br/>

이후 라즈베리 파이 SD카드를 마운트 해봅시다. 마운트하는 자세한 내용은 전 포스트인 크로스 컴파일 구축에 포스팅되어 있습니다.

```sh
# Mount SD card
ubuntu@ubuntu:~/working/modules $ sudo mkdir /mnt/raspi
ubuntu@ubuntu:~/working/modules $ sudo mkdir /mnt/fs
ubuntu@ubuntu:~/working/modules $ sudo mount /dev/sdb1 /mnt/raspi
ubuntu@ubuntu:~/working/modules $ sudo mount /dev/sdb2 /mnt/fs
```
<br/>
<br/>

install한 라즈베리파이 linux kernel을 SD카드에 copy합니다.

```sh
# Copy kernel image(zImage)
ubuntu@ubuntu:~/working/modules $ cd ../linux
ubuntu@ubuntu:~/working/linux $ sudo cp arch/arm/boot/zImage /mnt/raspi/kernel7.img
# Copy modules
ubuntu@ubuntu:~/working/modules $ cd ../modules
ubuntu@ubuntu:~/working/modules $ sudo cp -r lib/modules/4.4.50-v7+/ /mnt/fs/lib/modules/
# Copy device tree blob
ubuntu@ubuntu:~/working/linux $ sudo cp arch/arm/boot/dts/*.dtb /mnt/raspi/
# Copy device tree blob overlays
ubuntu@ubuntu:~/working/linux $ sudo cp arch/arm/boot/dts/overlays/*.dtb* /mnt/raspi/overlays/
ubuntu@ubuntu:~/working/linux $ sudo cp arch/arm/boot/dts/overlays/README /mnt/raspi/overlays/
```

<br/>
<br/>
이후 SD카드를 언마운트 시키고 커널 이미지 외의 모듈 디렉토리 등 모든 정보들을 전송 시킵니다. <br/>
ssh 프로토콜을 이용하여 통신할 것이기 때문에 전송시킬 때 라즈베리파이의 ip주소를 입력해야 합니다. [192.168.0.26] 이렇게 말이죠 

```sh
# Transmit kernel image
ubuntu@ubuntu:~/working/modules $ cd ../linux
ubuntu@ubuntu:~/working/linux $ scp arch/arm/boot/zImage pi@[RPi ip address]:/home/pi
# Transmit module directory. (kernel version 4.4.50-v7+)
ubuntu@ubuntu:~/working/linux $ cd ../modules
ubuntu@ubuntu:~/working/modules $ scp -r lib/modules/4.4.50-v7+/ pi@[Rpi ip address]:/home/pi
# Transmit device tree blob
ubuntu@ubuntu:~/working/modules $ cd ../linux
ubuntu@ubuntu:~/working/modules $ scp arch/arm/boot/dts/*.dtb pi@[RPi ip address]:/home/pi
# Transmit dtb overlays
ubuntu@ubuntu:~/working/linux $ scp arch/arm/boot/dts/overlays/*.dtb* pi@[RPi ip address]:/home/pi
ubuntu@ubuntu:~/working/linux $ scp arch/arm/boot/dts/overlays/README pi@[RPi ip address]:/home/pi
```
<br/>
<br/>

갈 길이 멉니다. 라즈베리파이에 sd카드를 연결한 후 카피한 커널 이미지와 모듈 디렉토리를 라즈베리파이의 부트 시스템 안으로 옮겨줍시다! <br/>
이후 라즈베리파이를 재부팅 하면 copy한 커널이미지를 로드시킬 것 입니다.

```sh
# Copy kernel image & module directory
pi@raspberry:~ $ sudo mv zImage /boot/kernel7.img
pi@raspberry:~ $ sudo mv 4.4.50-v7+/ /lib/modules/
# Copy device tree blob
pi@raspberry:~ $ sudo mv *.dtb /boot
# Copy dtb overlays
pi@raspberry:~ $ sudo mv *.dtb* /boot/overlays/
pi@raspberry:~ $ sudo mv README /boot/overlays/
# Reboot
pi@raspberry:~ $ sudo reboot
```

```sh
# Check kernel version
pi@raspberry:~ $ uname –r
```
커널 업데이트가 4.4.50-v7+로 업데이트 되었다면 여기 까지는 성공입니다..!! 
<br/>
이제 본격적으로 시스템 콜을 추가하여 실습해 봅시다.
<br/>
unistd.h에 있는 시스템콜 호출 함수 목록에 NR_mysyscall 이라는 시스템 콜 함수를 추가시켜 보겠습니다.

<br/>
<br/>
<br/>
먼저 시스템 콜 넘버를 등록합니다. 

```sh
# Add System call number
ubuntu@ubuntu:~/working/linux $ vi arch/arm/include/uapi/asm/unistd.h
```
![Imgur](https://i.imgur.com/Sl3kd9a.png)
<br/>
<br/>
<br/>
<br/>
다음으로 해당 번호를 불러올 시스템을 설정합니다.
```sh
ubuntu@ubuntu:~/working/linux $ vi arch/arm/kernel/calls.S
```
![Imgur](https://i.imgur.com/fFezn8c.png)
<br/>
<br/>
<br/>
<br/>
이렇게 말이죠!
<br/>
<br/>
<br/>
후... 
<br/>
<br/>
그리고나서 시스템콜에서 불러올 함수를 syscall.h 파일 안에 선언해 줍니다. 
```sh
# Register System call function
ubuntu@ubuntu:~/working/linux $ vi include/linux/syscalls.h
```
![Imgur](https://i.imgur.com/n8JEKsx.png)
<br/>
<br/>
<br/>
<br/>


이후 호출될 시스템 콜에서 수행할 함수 내용을 적어봅시다. 저는 간단하게 parameter값을 연산하는 내용을 적어보았습니다.<br/>
printk는 커널안에서의 print함수라고 생각하시면 될 것 같습니다.
```sh
# System call code
ubuntu@ubuntu:~/working/linux $ vi kernel/mysyscall.c
```
![Imgur](https://i.imgur.com/3IBn3d4.png)

<br/>
<br/>
<br/>
이후 컴파일을 진행 할 Makefile코드를 작성해 봅시다. 

```sh
# Makefile code
ubuntu@ubuntu:~/working/linux $ vi kernel/Makefile
```
![Imgur](https://i.imgur.com/TAxx0vo.png)

<br/>
<br/>
<br/>

이후 크로스 컴파일 합니다.
```sh
# Kernel compile
ubuntu@ubuntu:~/working/linux $ make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- zImage –j4
```
<br/>
<br/>
<br/>

커널쪽에서의 코딩은 완료가 된 것 같습니다.<br/>이제 시스템 콜을 불러올 어플리케이션에서의 함수를 따로 작성 해 주어야겠죠?<br/>
시스템콜 넘버를 불러올 헤더파일을 추가 시키고 진행합니다. 
```sh
# System call library
ubuntu@ubuntu:~/working/linux $ mkdir ../syscall; cd ../syscall
ubuntu@ubuntu:~/working/syscall $ vi newsyscall.c
```

![Imgur](https://i.imgur.com/iI2I5bA.png)

<br/>
<br/>
<br/>
어플리케이션 단에서의 프로그램을 작성합시다.

```sh
# System call application
ubuntu@ubuntu:~/working/syscall $ vi syscall_app.c
```
![Imgur](https://i.imgur.com/ZnT1A1K.png)
<br/>
<br/>
<br/>
어플리케이션 단에서의 Makefile을 작성합니다.

```sh
# Makefile
ubuntu@ubuntu:~/working/syscall $ vi Makefile
```

![Imgur](https://i.imgur.com/N4jZS1z.png)
<br/>
<br/>
<br/>

이제 실행파일을 만들고 해당 파일을 라즈베리파이 SD카드로 옮깁시다!<br/>
다시 SD카드를 삽입시키고 마운트 시켜주세요.<br/>
점점 끝이 보이기 시작합니다 ㅎㅎ

```sh
# Make system call library & system call application
ubuntu@ubuntu:~/working/syscall $ make
# Check SD card
ubuntu@ubuntu:~/working/syscall $ cd ../linux
ubuntu@ubuntu:~/working/linux $ lsblk
# Mount SD card
ubuntu@ubuntu:~/working/linux $ sudo mount /dev/sdb1 /mnt/raspi
ubuntu@ubuntu:~/working/linux $ sudo mount /dev/sdb2 /mnt/fs
```
<br/>
<br/>
<br/>

아까 했던 짓을 한번더 반복합니다. <br/>
커널 이미지, 실행파일 등등을 카피하고 Transmit시켜 줍시다<br/>

```sh
# Copy kernel image(zImage)
ubuntu@ubuntu:~/working/linux $ sudo cp arch/arm/boot/zImage /mnt/raspi/kernel7.img
# Copy syscall_app
ubuntu@ubuntu:~/working/linux $ cd ../syscall
ubuntu@ubuntu:~/working/syscall $ cp syscall_app /mnt/fs/home/pi
# Unmount SD card
ubuntu@ubuntu:~/working/syscall $ sudo umount /mnt/raspi
ubuntu@ubuntu:~/working/syscall $ sudo umount /mnt/fs
```
```sh
# Transmit kernel image(zImage)
ubuntu@ubuntu:~/working/syscall $ cd ../linux
ubuntu@ubuntu:~/working/linux $ scp arch/arm/boot/zImage pi@[Rpi ip address]:/home/pi
# Transmit syscall_app
ubuntu@ubuntu:~/working/linux $ cd ../syscall
ubuntu@ubuntu:~/working/syscall $ scp syscall_app pi@[Rpi ip address]:/home/pi
```
<br/>
<br/>
<br/>
라즈베리파이를 켜고 커널 이미지를 reload 합니다.

```sh
# Change kernel image
pi@raspberry:~ $ sudo mv zImage /boot/kernel7.img
pi@raspberry:~ $ sudo reboot
```
실행파일을 실행시키면....
<br/>
<br/>
<br/><br/>
<br/>
<br/>
두근두근...
<br/>
<br/>
<br/><br/>
<br/>
<br/><br/>
<br/>
<br/>
![Imgur](https://i.imgur.com/YK3zefR.jpg)

<br/>
<br/>
<br/>
<br/>
<br/>

### Success!
<br/>
<br/>
<br/>
성공적으로 시스템콜을 이용한 크로스 컴파일을 완료 하였습니다.