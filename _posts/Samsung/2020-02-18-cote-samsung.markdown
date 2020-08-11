---
layout: post
title:  "[삼성 SW 역량 테스트] 스타트와 링크 "
subtitle:   "삼성 기출문제"
categories: cote
tags: samsung
---

조합이기 때문에 분명 끝까지 for문이 돌지 않고

중간에서 멈춰도 최소값을 구할 수 있다.

## Code
---

```cpp 
#include <iostream>
#include <vector>
#include <algorithm>
#define MAX 987654321
using namespace std;

int map[20][20];
bool people[20];
int N, sum_start, sum_link;
int result = MAX;
vector<int> v1, v2;
vector<int> start, link;

void calculate(int n) {
	if (start.size()==2) {
		sum_start += (map[start[0]][start[1]] + map[start[1]][start[0]]);
		sum_link += (map[link[0]][link[1]] + map[link[1]][link[0]]);
		return;
	}

	for (int i = n; i < v1.size(); i++) {
		start.push_back(v1[i]);
		link.push_back(v2[i]);
		calculate(i + 1);
		start.pop_back();
		link.pop_back();
	}

}

void pick(int n) {
	if (v1.size()==N/2) {
		sum_start = 0, sum_link = 0;
		for (int i = 0; i < N; i++) 
			if (!people[i]) v2.push_back(i);
		
		calculate(0);
		v2.clear();
		int ret = abs(sum_start - sum_link);
		if (ret == 0) {
			printf("%d\n", ret);
			exit(0);
		}
		result = min(result, ret);
		return;
	}

	for (int i = n; i < N; i++) {
		v1.push_back(i);
		people[i] = true;
		pick(i + 1);
		v1.pop_back();
		people[i] = false;
	}

}


int main() {
	scanf("%d", &N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			scanf("%d", &map[i][j]);
		}
	}
	
	pick(0);
	printf("%d\n", result);
	return 0;
}
```


