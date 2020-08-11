---
layout: post
title:  "[삼성 SW 역량 테스트] 주사위 윷놀이 "
subtitle:   "삼성 기출문제"
categories: cote
tags: baekjoon
---

나에겐 정말 어려운 문제였다. 

보자마자 어떻게 풀지 생각하다 결국 풀지못하고 풀이를 봐버렸다. 

직접 하드 코딩을 인내하면서 하는 것이 중요한 것 같다. 

관건은 백트래킹할 떄의 조건이다.

## Code
---

```cpp 
#include <iostream>
#include <algorithm>
using namespace std;

int arr[10], mal[4];
int map[33], blue[33], score[33];
bool check[33];
int ans;

void dfs(int cnt, int sum) {
	if (cnt == 10) {
		ans = max(ans, sum);
		return;
	}

	for (int i = 0; i < 4; i++) {
		int prev = mal[i];
		int now = prev;
		int move = arr[cnt];

		if (blue[now] > 0) {
			now = blue[now];
			move--;
		}

		while (move--) {
			now = map[now];
		}

		if (check[now] && now != 21) continue;

		check[now] = true;
		check[prev] = false;
		mal[i] = now;
		dfs(cnt + 1, sum + score[now]);
		mal[i] = prev;
		check[prev] = true;
		check[now] = false;		
	}

}


int main() {
	for (int i = 0; i < 10; i++) scanf("%d", arr + i);

	for (int i = 0; i < 21; i++) map[i] = i + 1;
	map[21] = 21;
	for (int i = 22; i < 27; i++) map[i] = i + 1;
	map[27] = 20, map[28] = 29, map[29] = 25;
	map[30] = 31, map[31] = 32, map[32] = 25;

	blue[5] = 22, blue[10] = 28, blue[15] = 30;

	for (int i = 0; i < 21; i++) score[i] = i * 2;
	score[22] = 13, score[23] = 16, score[24] = 19;
	score[25] = 25, score[26] = 30, score[27] = 35;
	score[28] = 22, score[29] = 24, score[30] = 28;
	score[31] = 27, score[32] = 26;

	dfs(0, 0);
	printf("%d\n", ans);
	return 0;
}
```


