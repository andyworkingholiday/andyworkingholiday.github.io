---
layout: post
title:  "[Algorithm] 너비 우선 탐색(Breath First Search) BFS에 대해 알아보자"
subtitle: "BFS"
categories: devlog
tags: algorithm
---

이번 글에서는 **그래프(Graph)** 라는 자료구조를 순회하는 알고리즘 가운데 **너비우선탐색(Breath First Search)** 기법을 살펴보도록 하겠습니다. c++ 코드는 [이곳](https://hongku.tistory.com/156)을 참고하였습니다. 그럼 시작하겠습니다.


<br/>


## concepts

그래프 순회(traverse)란 그래프 내 모든 노드를 한번씩 모두 방문하는 걸 말합니다. 너비우선탐색이란 시작 노드를 방문한 후 시작 노드에 인접한 모든 노드를 우선 방문하는 그래프 순회 기법입니다. 더 이상 방문할 인접 노드가 없을 경우 그 다음 가까운 인접노드를 우선 방문합니다. 이 때 쓰는 자료구조가 바로 큐(queue)입니다. 너비우선탐색을 도식화한 그림은 다음과 같습니다.



<a href="https://imgur.com/SL1WRI6"><img src="https://i.imgur.com/SL1WRI6.png" width="250px" title="source: imgur.com" /></a>

<br/>
<br/>

## C++ 코드

너비우선탐색을 C++로 구현한 코드는 다음과 같습니다. 알고리즘은 다음과 같이 동작합니다.

```cpp
#include <iostream> 
#include <queue> 
#include <vector>
using namespace std; 

int number = 9; 
int visit[9];
vector<int> a[10]; 

void bfs(int start){ 
  queue<int> q; 
  q.push(start); 
  visit[start] = true; 
  while(!q.empty()){ 
    // 큐에 값이 있을경우 계속 반복 실행 
    // 큐에 값이 있다. => 아직 방문하지 않은 노드가 존재 한다. 
    int x = q.front(); 
    q.pop(); 
    printf("%d ", x); 
    for(int i=0; i< a[x].size(); i++){ 
      int y = a[x][i]; 
      if(!visit[y]){ // 방문하지 않았다면.. 
        q.push(y); 
        visit[y] = true; 
      } 
    } 
  } 
}
      
```


<br/>
<br/>


## 순회 예시1

하단 좌측 그림과 같은 그래프 *g*를 너비우선탐색 방식으로 순회해보겠습니다. 시작노드 *start*는 1이라고 두겠습니다. 우선 시작노드의 거리를 0으로 초기화한 뒤 시작노드를 큐에 집어 넣습니다. 하단 우측그림과 같습니다.

<br/>


<a href="https://imgur.com/XIZ75HD"><img src="https://i.imgur.com/XIZ75HD.png" width="500px" title="source: imgur.com" /></a>

<br/>


*dequque*를 합니다. 1이 나옵니다. 1(*currentVert*)의 이웃노드들(3, 6)을 차례대로 돌면서 그 색상을 *gray*로 색칠해 두고, 거리와 부모 정보를 업데이트 한 뒤 큐에 집어 넣습니다. 이렇게 이웃노드들을 모두 돌면 1의 색상을 *black*으로 색칠해 둡니다. 이렇게 한 결과가 하단 좌측 그림입니다.

<br/>

*dequque*를 합니다. 이번엔 3이 나옵니다. 3의 이웃노드들(1, 2, 4, 5)을 차례로 돌면서 그 색상을 *gray*로 색칠해 두고, 거리와 부모 정보를 업데이트 한 뒤 큐에 집어 넣습니다. 그런데 1은 *white*가 아니므로 이러한 처리에서 제외합니다. 어쨌든 이렇게 이웃노드들을 모두 돌면 3의 색상을 *black*으로 색칠해 둡니다. 이렇게 한 결과가 하단 우측 그림입니다.

한편 아래 그림에서 $d=1,2,2,2$라는 얘기는 시작노드와 노드6 사이의 거리가 1, 노드2, 노드4, 노드5와는 각각 2라는 뜻입니다.

<br/>

<a href="https://imgur.com/WJtlr3U"><img src="https://i.imgur.com/WJtlr3U.png" width="500px" title="source: imgur.com" /></a>

<br/>

*dequeue*를 합니다. 이번엔 6이 나옵니다. 6의 이웃노드들(1, 5)을 차례로 돌면서 같은 방식으로 처리하려고 했더니 1, 5 모두 *white*가 아닌 것을 알 수 있습니다. 처리할 것이 없으므로 6의 색상을 *black*으로 칠해둡니다. 이렇게 한 결과가 하단 우측 그림입니다.

그런데 여기에서 6과 5 사이의 엣지가 사라진 것이 의아할 수 있습니다. 조금 이따가 **너비우선트리(Breadth-first-tree)**를 설명하려고 이렇게 그림을 그린 것입니다. 그래프 구조가 바뀐 것은 아니니 신경 안쓰고 넘어가셔도 됩니다.

<br/>

<a href="https://imgur.com/JTdnggk"><img src="https://i.imgur.com/JTdnggk.png" width="500px" title="source: imgur.com" /></a>

<br/>

*dequeue*를 합니다. 이번엔 2가 나옵니다. 2의 이웃노드들(3)을 차례로 돌면서 같은 방식으로 처리하려고 했더니 3이 *white*가 아닌 것을 알 수 있습니다. 처리할 것이 없으므로 2의 색상을 *black*으로 칠해둡니다. 이렇게 한 결과가 하단 좌측 그림입니다. 마찬가지로 4와 5에 대해서도 수행하면 너비우선탐색이 종료됩니다.

<br/>

<a href="https://imgur.com/2Qx40Ng"><img src="https://i.imgur.com/2Qx40Ng.png" width="500px" title="source: imgur.com" /></a>

<br/>

시작노드 중심으로 너비우선탐색으로 부모-자식 관계를 만들어 나타낸 그래프를 `너비우선트리(Breath First Tree)`라고 합니다. 다음과 같습니다.

<br/>

<a href="https://imgur.com/1XbiAsN"><img src="https://i.imgur.com/1XbiAsN.png" width="350px" title="source: imgur.com" /></a>


<br/>


## 순회예시2

또다른 예시입니다. 다음과 같습니다.

<a href="https://imgur.com/F64H5hP"><img src="https://i.imgur.com/F64H5hP.png" title="source: imgur.com" /></a>


<br/>


## 계산복잡성

변수초기화 부분은 $O(1)$이므로 *whle* 반복문이 전체 계산량을 좌우합니다. 큐에 한번 넣었다가 큐에서 빼는 걸 모든 노드에 대해 수행하므로 $O($\|$V$\|$)$의 계산복잡성이 소요됩니다. 그런데 *currentVert*의 모든 인접노드에 대해 색상, 거리, 부모 정보를 업데이트하고, 이는 곧 그래프의 모든 엣지를 스캔한다는 의미이므로 *for* 반목문 안의 계산복잡성은 $O($\|$E$\|$)$가 됩니다. 따라서 전체적인 계산복잡성은 $O($\|$V$\|$+$\|$E$\|$)$가 됩니다. 

