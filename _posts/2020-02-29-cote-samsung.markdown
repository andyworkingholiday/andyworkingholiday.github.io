---
layout: post
title:  "[삼성 SW 역량 테스트] 경사로 "
subtitle:   "삼성 기출문제"
categories: cote
tags: baekjoon
---

Na982님의 엄청난 풀이 이다. 

경사로가 높아지는 경우, 낮아지는 경우에 대해 

심도있게 생각해보면 무릎을 탁 치면서 이해가 될 것이다.

## Code
---

```cpp 
#include <iostream>
#include <algorithm>
#define MAX 100
using namespace std;

int N, L, answer = 0;
int map[MAX][MAX], reverses[MAX][MAX];

void check(int a[MAX][MAX]) {
	int count = 0;
	int i, j;
	for (i = 0; i < N; i++) {
		count = 1;
		for (j = 0; j < N-1; j++) {
			if (a[i][j] == a[i][j + 1]) count++;
			else if (a[i][j] + 1 == a[i][j + 1] && count >= L) count = 1;
			else if (a[i][j] - 1 == a[i][j + 1] && count >= 0) count = 1 - L;
			else break;
		}
		if (j == N - 1 && count >= 0) answer++;
	}
}

int main() {
	scanf("%d %d", &N, &L);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			scanf("%d", &map[i][j]);
			reverses[j][i] = map[i][j];
		}
	}
	check(map);
	check(reverses);
	printf("%d\n", answer);
	return 0;
}
```


