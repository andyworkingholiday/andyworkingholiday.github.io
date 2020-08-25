---
layout: post
title:  "[Embedded System] FreeRTOS에 대해 알아보자"
subtitle:   "freertos"
categories: devlog
tags: embedded
---

오늘은 RTOS중에서 접근성이 좋은 FreeRTOS에 대해 알아보고 각 함수의 기능에 대해서 설명해 보겠습니다. 저번 포스팅에서 소개해 드린 것처럼 RTOS의 종류에는 **VxWorks**, **pSOS**, **QNX** 등이 있습니다. 저는 범용성이 좋고 상대적으로 사용하기 쉬운 FreeRTOS를 이용해 보려합니다. 그럼 시작하겠습니다!

<br/>


## FreeRTOS
<br/>

**FreeRTOS**는 오픈소스 RTOS 중 하나로, 단순함, 뛰어난 이식성, 간결함을 목표로 설계되었습니다. 실제로 운영체제 core 소스 코드가 4,000라인을 넘지 않으며, 불가피한 몇몇 코드를 제외한 거의 모든 code가 C 언어로 작성되어 이식성이 뛰어납니다. 현재 8051, AVR, PIC18, H8, MSP430, HCS12, PIC24, ARM Cortex-M3, ARM7, ARM9, AVR32, ColdFire, x86 등의 다양한 8bit, 16 bit, 32 bit 프로세서에 이식되어 있습니다.

FreeRTOS core는 아래와 같은 기능들을 제공합니다.

* Mutlitasking
  - 선점형, 비선점형 선택 가능
  - Task 수 제한 없음
  - 동적으로 task 생성 가능
  - Priority 수 제한 없음
  - 두개 이상의 task에 같은 priority 할당 가능 (이 경우 라운드 로빈 스케쥴링)

* Co-routine
* Message queue
* Semaphore
* Memory Management

<br/>


FreeRTOS는 변형된 GPL license 정책을 사용합니다. GPL license와 주된 차이점은 FreeRTOS와 link되어 실행되는 저작물을 공개할 의무가 없다는 것입니다. 이점은 보안이나 상업적인 목적으로 저작물에 대한 독점적 권리를 확보할 필요가 있을 때 이를 가능하게 합니다.

<br/>

## Multitasking

<br/>

**Multitasking**은 여러 개의 task를 동시에 실행시킬 수 있도록 해주는 기능을 말합니다. 실제로 여러 개의 task가 동시에 수행되는 것은 아니고, 여러 개의 task를 아주 짧은 시간 간격으로 번갈아 실행시켜 사용자가 그렇게 느끼도록 해줍니다.

<br/>


여러 개의 task 중 어느 task를 언제 얼마동안 실행 시킬 것인지를 결정하는 것을 scheduling이라고 합니다. Scheduling 방식은 크게 두가지로 분류됩니다. 하나는 현재 수행중인 task가 자발적으로 수행 권한은 양보할 때 다른 task가 수행 권한을 얻을 수 있는 nonpreemption 방식이고, 다른 하나는 scheduler가 필요에 따라 강제적으로 수행 권한을 다른 task에게 양도할 수 있는 preemption 방식입니다. 
<br/>

**Nonpreemption** 방식은 voluntary scheduling이라고도 합니다. 
**Preemption** 방식에는 여러 가지가 있는데 그들 중 가장 많이 사용하는 방식이 priority scheduling 과 round robin scheduling 방식을 혼합한 것입니다. 이것은 기본적으로 priority scheduling 방식을 따르되 같은 proirty를 가진 task들은 round robin 방식으로 scheduling을 하는 것입니다. 이 방식을 priotiry based round robin scheduling 이라고 합니다.

<br/>

FreeRTOS는 Nonpreemption 방식과 Preemption 방식을 모두 지원하며, preemption 방식으로는 prority based round robin scheduling 방식을 지원합니다. 또한 생성할 수 있는 task 수와 priority 수의 제한이 없으며, 실행중에 동적으로 task를 생성할 수도 있습니다.

FreeRTOS는 multitasking을 library 형태로 지원하며, 관련 API는 다음과 같습니다.
<br/>

* **xTaskCreate** : 새로운 task를 생성합니다.
* **vTaskDelete** : task를 삭제합니다.
* **vTaskDelay** : 주어진 clock tick 동안 task를 지연시킵니다.
* **vTaskDelayUntil** : 명시된 시점까지 task를 지연시킵니다.
* **uxTaskPriorityGet** :Task의 우선순위를 돌려줍니다.
* **vTaskPrioritySet** : Task의 우선순위를 지정합니다.
* **vTaskSuspend** : Task를 중지(suspend)시킵니다.
* **vTaskResume** : 중지(suspend)된 task를 다시 시작시킵니다.
* **vTaskResumeFromISR** :중지(suspend)된 task를 다시 시작시킵니다.이 함수는 ISR에서 사용 가능합니다.

<br/>

## Coroutine

<br/>

**Coroutine**이란 여러 개의 entry point를 가질 수 있고, 임의의 지점에서 suspending과 resuming이 가능한 subroutine을 말합니다. 이것은 원래 assembly-language로 프로그래밍을 할 때 사용하던 기술인데, Simula나 Modula-2 같은 몇몇 고급언어에서도 지원합니다. Coroutine의 장점은 작은 자원으로 multitasking 효과를 낼 수 있다는 것입니다. 이유는 task들과는 달리 coroutine간에 제어권 전환이 이루어질 때 context switching을 하지 않으며, coroutine들이 stack을 공유하기 때문입니다. 하지만 voluntary scheduling 만 지원 가능하고, local 변수 활용이 힘들다는 단점이 있습니다.

<br/>

FreeRTOS는 coroutine을 library 형태로 지원하며, 관련 API는 다음과 같습니다.



* **xCoRoutineCreate** : 새 coroutine을 생성합니다.
* **crDELAY** : 주어진 clock tick 동안 coroutine을 지연시킵니다.

<br/>

## Message queue

<br/>

Message queue는 task들이 작은 크기의 데이터인 message를 비동기(asynchronous)적으로 서로 주고 받을 수 있도록 해줍니다. 관련 API는 다음과 같습니다.

* **xQueueCreate** : 새 queue를 생성합니다.
* **xQueueDelete** : Queue를 삭제합니다.
* **xQueueSend** : Queue에 message를 넣습니다.
* **xQueueReceive** : Queue로부터 message를 가져옵니다.
* **uxQueueMessagesWaiting** : Queue에 저장된 message의 개수를 돌려줍니다.
* **xQueueSendFromISR** : Queue에 message를 넣습니다. 이 함수는 ISR에서 사용 가능합니다.
* **xQueueReceiveFromISR** : Queue로부터 message를 가져옵니다. 이 함수는 ISR에서 사용 가능합니다.
* **crQUEUE_SEND** : Message queue로 message를 보냅니다. 이 함수는 Coroueine에서 사용 가능합니다.
* **crQUEUE_RECEIVE** : Message queue로 message를 받습니다. 이 함수는 Coroueine에서 사용 가능합니다.
* **crQUEUE_SEND_FROM_ISR** : Message queue를 사용해 ISR에서 coroutine으로 message를 보낼 때 이 함수를 사용합니다.
* **crQUEUE_RECEIVE_FROM_ISR** : Message queue를 사용해 ISR에서 coroutine으로부터 message를 받을 때 이 함수를 사용합니다.

<br/>

## Semaphore

<br/>

**Semaphore**는 multitasking 환경에서 한 순간에 하나의 task만이 사용할 수 있도록 공유 자원을 보호해야 할 필요가 있는 경우 사용하는 대표적인 자료 구조입니다.

관련 API는 다음과 같습니다.



* **vSemaphoreCreateBinary** : Semaphore를 생성합니다.
* **xSemaphoreTake** : Semaphore를 획득합니다. Semaphore의 P operation 역할을 합니다.
* **xSemaphoreGive** : Semaphore를 풀어줍니다.. Semaphore의 V operation 역할을 합니다.
* **xSemaphoreGiveFromISR** : Semaphore를 풀어줍니다.. Semaphore의 V operation 역할을 합니다. 이 함수는 ISR에서 사용 가능합니다.

