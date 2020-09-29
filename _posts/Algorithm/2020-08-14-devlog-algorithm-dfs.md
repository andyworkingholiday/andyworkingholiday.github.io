---
layout: post
title:  "[Algorithm] 깊이 우선 탐색(Depth First Search) DFS에 대해 알아보자"
subtitle: "DFS"
categories: devlog
tags: algorithm
use_math: true
---

이번 글에서는 **그래프(Graph)** 라는 자료구조를 순회하는 알고리즘 가운데 **깊이우선탐색(Depth First Search)** 기법을 살펴보도록 하겠습니다. c++ 코드는 [이곳](https://hongku.tistory.com/157?category=804730)을 참고하였습니다.



<br/>

## concept

깊이우선탐색이란 더 이상 나아갈 엣지가 없을 때까지 깊이 탐색하는 그래프 순회(traverse) 기법입니다. 깊이우선탐색을 도식화한 그림은 다음과 같습니다.



<a href="https://imgur.com/YjfpWci"><img src="https://i.imgur.com/YjfpWci.png" width="250px" title="source: imgur.com" /></a>



<br/>
<br/>


## C++ 구현

깊이우선탐색을 C++로 구현한 코드는 다음과 같습니다. 알고리즘은 다음과 같이 동작합니다.
Stack을 이용하여 구현할 수 있으나 직관적인 재귀함수를 이용하였습니다. 


```cpp
#include <iostream> 
#include <vector> 
using namespace std; 

int number = 9; 
int visit[9]; 
vector<int> a[10]; 

void dfs(int start){ 
  if(visit[start]) return;
    // 방문한경우 바로 빠져나옴 
    visit[start] = true; 
    printf("%d ", start); 
    for(int i=0; i< a[start].size(); i++){ 
      // 인접한 노드를 방문 
      int x = a[start][i]; 
      dfs(x); 
    }
 }
      
```


<br/>
<br/>

## 순회 예시

(a)와 같은 그래프를 깊이우선탐색 방식으로 순회해보겠습니다. 시작노드 *startVertex*를 $u$로 두겠습니다. 그러면 우선 시작노드 발견시점을 1로 적어두고, *processing*하고 있다는 의미로 색상을 *gray*로 바꿉니다. 이후 *dfsvisit* 함수가 재귀적으로 세 번 호출되면서 $v$, $y$, $x$ 노드에 대해 같은 작업이 진행돼 전체적으로는 (b), (c), (d) 같은 모습을 띄게 됩니다.

<br/>

<a href="https://imgur.com/c1rJ8Hs"><img src="https://i.imgur.com/c1rJ8Hs.png" title="source: imgur.com" /></a>


<br/>


이번엔 (f)를 봅시다. (f)에서는 x를 시작노드로 하는 재귀함수 *dfsvisit*이 수행되고 있습니다. x의 이웃노드 가운데 *white*인 노드가 없어서 더 이상 깊이우선탐색이 불가능합니다. x의 *processing*이 완료되었다는 의미로 색상을 *black*으로 바꾸고 종료시점(5)을 기록하고 재귀함수 호출을 종료합니다.

(g)를 봅시다. (g)에서는 y를 시작노드로 하는 재귀함수 *dfsvisit*이 수행되고 있습니다. y의 이웃노드 가운데 *white*인 노드가 없어서 더 이상 깊이 우선 탐색이 불가능합니다. y의 *processing*이 완료되었다는 의미로 색상을 *black*으로 바꾸고 종료시점(6)을 기록하고 재귀함수 호출을 종료합니다.

(h)를 봅시다. (h)에서는 v를 시작노드로 하는 재귀함수 *dfsvisit*이 수행되고 있습니다. v의 이웃노드 가운데 *white*인 노드가 없어서 더 이상 깊이 우선 탐색이 불가능합니다. v의 *processing*이 완료되었다는 의미로 색상을 *black*으로 바꾸고 종료시점(7)을 기록하고 재귀함수 호출을 종료합니다.

<br/>

<a href="https://imgur.com/F6MtJzY"><img src="https://i.imgur.com/F6MtJzY.png" title="source: imgur.com" /></a>

<br/>

(m)을 봅시다. (m)에서는 u를 시작노드로 하는 재귀함수 *dfsvisit*이 수행되고 있습니다. u의 이웃노드 가운데 *white*인 노드가 없어서 더 이상 깊이 우선 탐색이 불가능합니다. u의 *processing*이 완료되었다는 의미로 색상을 *black*으로 바꾸고 종료시점(8)을 기록하고 재귀함수 호출을 종료합니다. 

이로써 *dfs*에서의 *dfsvisit* 재귀함수 호출이 종료됐습니다. 그런데 아직 아래의 반복문은 계속 돌고 있으므로, 나머지 노드 가운데 아직 방문하지 않은(*white*) 노드가 있다면 해당 노드들에 대해서도 *dfsvistit*을 수행해 줍니다.

<br/>

(m)을 다시 봅시다. $w$를 시작노드로 *dfvisit*을 수행합니다. $w$ 발견시점을 9로 하고 *processing*하고 있다는 의미로 색상을 *gray*로 바꿉니다. 이후 $w$의 이웃노드인 $z$를 시작노드로 *dfvisit*을 재귀적으로 수행합니다. $z$ 발견시점을 10으로 하고 *processing*하고 있다는 의미로 색상을 *gray*로 바꿉니다.

(o)를 봅시다. (o)에서는 $z$를 시작노드로 하는 재귀함수 *dfvisit*이 수행되고 있습니다. $z$의 이웃노드 가운데 *white*인 노드가 없어서 더 이상 깊이 우선 탐색이 불가능합니다. $z$의 *processing*이 완료되었다는 의미로 색상을 *black*으로 바꾸고 종료시점(11)을 기록하고 재귀함수 호출을 종료합니다. 

(p)를 봅시다. (p)에서는 $w$를 시작노드로 하는 재귀함수 *dfvisit*이 수행되고 있습니다. $w$의 이웃노드 가운데 *white*인 노드가 없어서 더 이상 깊이 우선 탐색이 불가능합니다. $w$의 *processing*이 완료되었다는 의미로 색상을 *black*으로 바꾸고 종료시점(12)을 기록하고 재귀함수 호출을 종료합니다. 

(p)까지 수행이 끝나면 그래프 전체에서 *white*인 노드가 없게 됩니다. 이로써 *dfs*의 반복문도 종료됩니다.


<br/>

<a href="https://imgur.com/Foq0m9r"><img src="https://i.imgur.com/Foq0m9r.png" title="source: imgur.com" /></a>


<br/>
<br/>

## 깊이우선탐색 그래프의 특성

위 예시에서 깊이우선탐색이 종료된 그래프의 모습은 다음과 같습니다.



<a href="https://imgur.com/CdjPenA"><img src="https://i.imgur.com/CdjPenA.png" width="300px" title="source: imgur.com" /></a>



깊이우선탐색으로 그래프를 순회한 결과로 부모-자식노드가 도출됩니다. 예컨대 위 그림에서는 $u$와 $v$, $w$와 $b$가 그러한 관계를 갖습니다. 부모에서 자식노드로 향하는 엣지를 `Tree edge/Discover edge`라고 합니다. 반대로 자식에서 부모노드로 향하는 엣지를 `Back edge`라고 합니다. 위 그림 기준으로는 음영표시된 엣지가 tree edge, **B**라고 표시된 엣지가 back edge입니다. 무방향그래프(undirected graph)에서는 Tree edge와 Back edge만 존재합니다.

<br/>

Tree edge가 아닌 엣지 가운데 자손노드로 향하는 노드를 `Forward edge`라고 합니다. `Cross edge`는 이 세 가지 경우를 제외한 모든 종류의 엣지를 나타냅니다. 위 그림 기준으로는 **F**가 forward edge, **B**가 back edge입니다. back edge가 존재한다는 말은 사이클(cycle)이 존재한다는 말과 같습니다.

<br/>

위 여섯개 노드를 발견시점과 탐색종료시점 순으로 나열하면 다음과 같습니다. 자식노드의 발견시점과 탐색종료시점은 부모노드의 부분집합이 됩니다. 이같은 구조를 *parenthesis structure*라고 합니다.

<br/>

<a href="https://imgur.com/8DqJqgR"><img src="https://i.imgur.com/8DqJqgR.png" width="500px" title="source: imgur.com" /></a>



뿐만 아니라 어떤 임의의 노드가 있고, 이 경로상에 존재하는 *white* 노드들은 깊이우선탐색 기준 해당 노드의 자식노드가 됩니다. 이를 *white path theorem*이라고 합니다.


<br/>
<br/>

## 깊이우선탐색의 계산복잡성

깊이우선탐색은 모든 노드와 엣지에 대해 한번씩은 탐색하게 되므로 전체적인 계산복잡성은 $O($\|$V$\|$+$\|$E$\|$)$가 됩니다.