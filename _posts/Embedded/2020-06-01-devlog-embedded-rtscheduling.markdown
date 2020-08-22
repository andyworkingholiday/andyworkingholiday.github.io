---
layout: post
title:  "[Embedded System] Real Time System에서의 스케쥴링 기법에 대해 알아보자"
subtitle:   "rtos_scheduling"
categories: devlog
tags: embedded
---

오늘은 Real Time System에서의 스케쥴링 기법에 대해 알아보겠습니다. Real Time System에서의 스케쥴링 방식에 대해 포스팅 할 것이므로 SJF, Round Robin 등의 스케쥴링 방식에 대해 다루지는 않습니다. 운영체제 수업 중간고사 공부를 하면서 공부한 내용을 최대한 쉽게 적어 보았습니다. 그럼 시작하겠습니다.

<br/>


## Scheduling in Real time System

<br/>

Real Time System에서Scheduling은 어떻게 적용될까? Real-time system은 task 처리에 걸리는 시간을 일관되게 유지할 수 있느냐가 중요한 성능의 척도입니다. 그러므로 Real-time system의 목표는 실시간 성능 보장에 있습니다. 첫번째로는 DeadLIne 내에 스케쥴링이 되어야 하고 두번째로는 예측성이 보장되어야 하는 것입니다. 제가 수강하고 있는 운영체제 수업에서는 주기적 태스크 스케쥴링 알고리즘과 비주기적 태스크가 포함될 때의 스케쥴링 기법에 대해 배웠습니다. 

<br/>

## Periodic Task Scheduling

<br/>

Periodic Task Scheduling 기법 중 하나로 RMS를 사용할 수 있습니다. **Rate-Monotonic scheduling algorithm**은 정적 우선순위의 선점형 알고리즘입니다. 우선순위는 주기에 반비례하여 할당되며, 짧은 주기의 작업이 높은 우선순위를 받게 됩니다. 예를 들어, 두 개의 process P1, P2가 있는데 주기가 각각 50, 100, CPU burst가 각각 20, 35, deadline은 주기와 같다고 가정해 봅시다. 만약 우선순위가 제대로 지켜지지 않아서 P2가 먼저 실행된다면, P1은 deadline 내에 작업을 끝내지 못할것 입니다. 

![Imgur](https://i.imgur.com/NCy3r51.png)

<br/>

반대로, P1이 높은 우선순위를 받으면, P1이 먼저 시작되고 P1이 작업을 마치면 P2가 시작됩니다. P2는 35 중 30만 작업을 완료하지만 P1의 주기가 돌아와 선점됩니. P1이 작업을 마치면 P2가 다시 시작되고, 나머지 작업 5를 마칩니다. 

<br/>

![Imgur](https://i.imgur.com/feXBcXA.png)

<br/>

위 그림처럼 두 Process는 75에 완료되고 반복하기 전까지 25는 Slack으로 유휴 상태입니다. 이 알고리즘으로 scheduling 할 수 없는 process 집합은 다른 priority scheduler로도 scheduling 할 수 없습니다. 따라서 정적 우선순위를 사용하는 알고리즘 중에서 Rate-Monotonic Scheduling이 가장 적합한 것으로 여겨집니다. 이처럼 주기적 Task Scheduling에서 마감시간 내에 작업이 안전하게 종료할 수 있는 지의 여부를 다음과 같은 Rate-Monotonic Algorithm공식을 통해 확인할 수 있습니다.

<br/>

![Imgur](https://i.imgur.com/tc6Stux.png)

U는 utilization을 의미하고 N은 주기적 태스크이 개수를 의미합니다. C는 태스크가 수행되는 Computation time을 의미하고 T는 주기입니다. 이 공식으로 주기적 task의 worst case에서 scheduling을 분석할 수 있습다. (n이 무한대로 발산하면 0.69의 값으로 수렴합니다.) 즉 태스크의 개수가 엄청나게 많을 때는 마감시간 내의 모든 주기적 태스크가 안전하게 보장될 때 효율성이 약 70퍼센트로 수렴하게 됩니다.

<br/>

## Aperiodic Task Scheduling

Periodic Task Scheduling 에서의 목적은 마감시간을 보장해주는 것 이였다면  Aperiodic Task Scheduling 반응시간을 빠르게 하는 것입니다. 이 스케줄링 목적을 달성하기 위해서 수행되는 스케줄링 알고리즘은 3가지가 있고 수업 과제를 통해 2가지를 실습해볼 수 있었습니다. 구체적인 코드는 제 [깃허브](https://github.com/andyworkingholiday/Operating_System/tree/master/Lab01) 에서 확인할 수 있습니다.


<br/>

### 1) Background Aperiodic Service

<br/>

![Imgur](https://i.imgur.com/7Eu3lSL.png)

<br/>

간단하게 말해서 Arrival Time에 일치하는 시간에 Aperiodic task가 와도 Periodic task를 수행할 수 있다면 먼저 수행하는 방식입니다. 그리고 남은 시간(slack)에 Aperiodic task를 수행하게 됩니다. 여기서 Aperiodic Task의 반응시간(평균대기시간)을 더 빠르게 하고싶은 방법이 대두되었습니다. 

<br/>

### 2) Pollng Service

<br/>

![Imgur](https://i.imgur.com/v13nmEG.png)

<br/>

주기적으로 Aperiodic task가 들어갈 수 있는 지를 체크하게 됩니다. 이 때 Polling server의 execution time이 나오게 되는데 이는 한번에 수행할 수 있는 수행 시간을 말합니다. 그리고 Polling Server의 주기는 Aperiodic Task의 유무를 검사하게 되는 주기입니다.

<br/>

### 3) Deferable Server

<br/>

서버가 항상 자기의 bandwith(capacity)를 유지하는 방식으로 스케줄링을 구현합니다. 쉽게 말하면 양동이에 일정량의 물을 채워놓고 period가 될 때마다 일정량 만큼의 물을 채우는 방식입니다. Polling Service는 Polling Period마다 Aperiodic Task를 체크하지만 Deferable Server의 경우 바로바로 서비스를 실행 할 수 있으므로 Polling Server보다 반응속도가 빠릅니다.

<br/>


![Imgur](https://i.imgur.com/M6V9DM6.png)

<br/>

Polling Service와 Deferable Server를 사용할 경우 주의할 점은 이 때의 주기는 주어진 주기적 task들의 가장 높은 우선순위를 가진 task의 주기보다 같거나 짧아야 한다는 것입니다. 즉, Polling 및 Deferable의 우선순위가 가장 높아야 합니다.

<br/>

다음에는 제가 구현한 각 알고리즘에 대한 코드 분석을 해보겠습니다.