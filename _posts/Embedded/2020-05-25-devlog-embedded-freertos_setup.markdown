---
layout: post
title:  "[Embedded System] FreeRTOS 개발환경을 구축해보자"
subtitle:   "freertos_setup"
categories: devlog
tags: embedded
---

오늘은 FreeRTOS 개발환경을 구축해 보겠습니다. 실습은 Arduino를 통해 진행할 예정입니다. 그럼 시작하겠습니다!

<br/>


## FreeRTOS 설치

<br/>

아두이노에서 라이브러리 형태로 쓸 수 있는 FreeRTOS는 [다음사이트](https://github.com/greiman/FreeRTOS-Arduino/tree/master/libraries/FreeRTOS_ARM/examples) 에서 다운받을 수 있습니다. git clone 명령어를 쓰셔서 다운받으시면 됩니다.

다운로드 받은 FreeRTOS-Arduino-master 폴더 내 'libraries' 내부로 접근하면 ‘FreeRTOS_ARM’, ‘FreeRTOS_AVR’, ‘SdFat’ 3개의 폴더가 있음을 확인할 수 있습니다.

<br/>

![Imgur](https://i.imgur.com/qcEDAwy.png)

<br/>

이후 ‘FreeRTOS_ARM’, ‘FreeRTOS_AVR’, ‘SdFat’ 3개의 폴더를 아두이노 라이브러리 폴더로 이동시키면 됩니다.
* **이동 위치**: Arduino IDE의 메뉴 표시줄의 ‘파일’–‘환경설정’–‘스케치북 위치’ 폴더 내 libraries

<br/>

![Imgur](https://i.imgur.com/kAUgD4f.png)

이제 아두이노에 들어가서 메뉴 표시줄의 ‘스케치’–‘라이브러리 포함하기’–‘라이브러리 관리’ 실행 후 종료시킵니다. 라이브러리 관리를 실행하여 추가된 FreeRTOS 라이브러리가 적용되도록 하기 위함입니다! 메뉴표시줄의 ‘스케치’–‘라이브러리 포함하기’ – 하단에 Contributed 라이브러리를 확인하면 추가한 **FreeRTOS**를 확인할 수 있습니다.

<br/>

![Imgur](https://i.imgur.com/bVJT6gD.png)

<br/>

## FreeRTOS Example

<br/>

설치가 완료 되었다면 예제를 통해 확인해 봐야겠죠? frBlink예제를 통해 Thread 2개를 이용하여 아두이노 LED ON/OFF를 수행해 봅시다. ‘파일’ – ‘예제’ – 하단의 FreeRTOS_AVR – frBlink 선택해 봅시다. 아 참고로 저는 AVR 계열의 Arduino UNO보드를 사용하여 실습하였습니다. 만약 ARM계열의 보드를 사용하신다면 아래의 '호환되지 않음'을 선택하여 **FreeRTOS_ARM** 을 사용할 수 있습니다.

<br/>

![Imgur](https://i.imgur.com/wZWvfW8.png)

```cpp 

/*
 * Example to demonstrate thread definition, semaphores, and thread sleep.
 */
#include <FreeRTOS_AVR.h>

// The LED is attached to pin 13 on Arduino.
const uint8_t LED_PIN = 13;

// Declare a semaphore handle.
SemaphoreHandle_t sem;
//------------------------------------------------------------------------------
/*
 * Thread 1, turn the LED off when signalled by thread 2.
 */
// Declare the thread function for thread 1.
static void Thread1(void* arg) {
  while (1) {

    // Wait for signal from thread 2.
    xSemaphoreTake(sem, portMAX_DELAY);

    // Turn LED off.
    digitalWrite(LED_PIN, LOW);
  }
}
//------------------------------------------------------------------------------
/*
 * Thread 2, turn the LED on and signal thread 1 to turn the LED off.
 */
// Declare the thread function for thread 2.
static void Thread2(void* arg) {

  pinMode(LED_PIN, OUTPUT);

  while (1) {
    // Turn LED on.
    digitalWrite(LED_PIN, HIGH);

    // Sleep for 200 milliseconds.
    vTaskDelay((200L * configTICK_RATE_HZ) / 1000L);

    // Signal thread 1 to turn LED off.
    xSemaphoreGive(sem);

    // Sleep for 200 milliseconds.
    vTaskDelay((200L * configTICK_RATE_HZ) / 1000L);
  }
}
//------------------------------------------------------------------------------
void setup() {
  portBASE_TYPE s1, s2;

  Serial.begin(9600);
  
  // initialize semaphore
  sem = xSemaphoreCreateCounting(1, 0);

  // create task at priority two
  s1 = xTaskCreate(Thread1, NULL, configMINIMAL_STACK_SIZE, NULL, 2, NULL);

  // create task at priority one
  s2 = xTaskCreate(Thread2, NULL, configMINIMAL_STACK_SIZE, NULL, 1, NULL);

  // check for creation errors
  if (sem== NULL || s1 != pdPASS || s2 != pdPASS ) {
    Serial.println(F("Creation problem"));
    while(1);
  }
  // start scheduler
  vTaskStartScheduler();
  Serial.println(F("Insufficient RAM"));
  while(1);
}
//------------------------------------------------------------------------------
// WARNING idle loop has a very small stack (configMINIMAL_STACK_SIZE)
// loop must never block
void loop() {
  // Not used.
}

```

<br/>

frblink를 예제를 보면 Thread1 과  Thread2 가 세마포어를 통해 두 개의 프로세스가 상호 동작하는 것을 확인할 수 있을 것 입니다. 