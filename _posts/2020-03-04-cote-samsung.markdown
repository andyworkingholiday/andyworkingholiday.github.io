---
layout: post
title:  "[삼성 SW 역량 테스트] 로봇청소기 "
subtitle:   "삼성 기출문제"
categories: cote
tags: samsung
---

기출 치고는 쉬웠던 문제.

시뮬레이션을 잘 하는 것이 중요하다.

## Code
---

```cpp 
#include <iostream>
#define MAX 51
using namespace std;

struct yxd {
	int y, x, d;
};

int N, M, answer = 0;
int map[MAX][MAX];
int start_y, start_x, dir;
const int dy[] = { -1,0,1,0 }, dx[] = { 0,1,0,-1 };

void dfs(yxd robot) {
	if (map[robot.y][robot.x]==0) {
		answer++;
		map[robot.y][robot.x] = 2;
	}
	int i = 0;
	for (i = 1; i <= 4; i++) {
		int nd = robot.d - i;
		if (nd < 0) nd += 4;
		int ny = robot.y + dy[nd], nx = robot.x + dx[nd];
		if (ny<0 || ny>N - 1 || nx<0 || nx>M - 1) continue;
		if (map[ny][nx] == 0) {
			dfs({ ny,nx,nd });
		}
	}

	int by = robot.y - dy[robot.d];
	int bx = robot.x - dx[robot.d];
	if (map[by][bx] == 1) {
		printf("%d\n", answer);
		exit(0);
	}
	dfs({ by,bx,robot.d });
}

int main() {
	scanf("%d %d", &N, &M);
	scanf("%d %d %d", &start_y, &start_x, &dir);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			scanf("%d", &map[i][j]);
		}
	}

	dfs({ start_y, start_x, dir});
	return 0;
}
```


