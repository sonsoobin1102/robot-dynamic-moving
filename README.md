# 🤖 PC-Bang Autonomous Serving Robot Simulation
> **좁은 통로와 동적 장애물이 빈번한 PC방 환경에 특화된 자율주행 서빙 로봇 시뮬레이터**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-orange)
![NumPy](https://img.shields.io/badge/Library-NumPy-blue)

## 📝 프로젝트 개요 (Overview)
이 프로젝트는 **좁은 통로**와 **갑자기 튀어나오는 의자(Pop-out)** 등 PC방의 특수한 환경을 반영한 자율주행 시뮬레이션입니다. Python과 Matplotlib을 활용하여 직접 엔진을 구현하였으며, **A* 알고리즘(Global Path)**과 **DWA(Local Path)**를 결합하여 최적의 경로 생성 및 회피 주행을 시뮬레이션합니다.

*본 프로젝트는 자율주행 알고리즘에 대한 학습과 검증을 목적으로 제작된 개인 프로젝트입니다.*

---

## 🛠 핵심 기능 및 로직 (Key Features)

### 1. 하이브리드 주행 시스템 (Hybrid Path Planning)
* **전역 경로 (Global Planner):** **A* 알고리즘**을 사용하여 맵 전체의 정적 장애물을 고려한 최단 경로를 계산합니다.
* **지역 경로 (Local Planner):** **DWA (Dynamic Window Approach)**를 사용하여 로봇의 물리적 제약(속도, 가속도)을 고려해 실시간으로 동적 장애물을 회피합니다.

### 2. 스마트 회피 판단 (Smart Decision Making)
장애물 감지 시 무조건 회피하지 않고, 비용(Cost)을 계산하여 능동적으로 판단합니다.
* **거리 비용 계산:** `우회 경로 거리` vs `직선 최단 거리`
* **판단 로직:**
    * **우회 (Detour):** 우회 거리가 임계값($2.5m$) 이하이면 자연스럽게 돌아갑니다.
    * **상호작용 (Interaction):** 우회 거리가 너무 멀면 '요청 모드'로 전환하여 보행자에게 비켜달라는 신호를 보냅니다.

### 3. 사용자 상호작용 (HRI)
* **시각적 신호:** 주행 불가 상황 시 물결 파형(Wave)을 시각화하여 비켜달라는 의도를 표현합니다.
* **Timeout & Re-planning:** 신호 후 5초간 장애물이 사라지지 않으면, **강제로 경로를 재탐색**하여 교착 상태를 방지합니다.

### 4. 정밀 충돌 모델링 (Polygon Collision Check)
* 로봇을 단순한 점(Point)이 아닌 **직사각형($0.48m \times 0.52m$)**으로 모델링했습니다.
* 회전 시 튀어나오는 모서리 좌표를 실시간으로 계산하여 좁은 코너에서도 벽과 겹치지 않는 현실적인 주행을 구현했습니다.

---

## 💡 문제 해결 과정 (Troubleshooting)

프로젝트 개발 중 발생한 주요 문제와 이를 논리적으로 해결한 과정입니다.

### 1. 좁은 통로에서의 데드락 (Deadlock)
* **Problem:** 양옆이 벽인 좁은 통로에서 장애물을 만났을 때, DWA가 갈 곳을 찾지 못해 제자리에서 멈춤.
* **Cause:** DWA(지역 계획)는 눈앞의 상황만 판단하므로, 멀리 돌아가야 하는 경로를 인지하지 못함.
* **Solution:** 로봇이 일정 시간 정지하면 **A* 알고리즘(전역 계획)을 재호출**하여 처음부터 경로를 다시 탐색하도록 로직 개선.

### 2. 과도한 우회로 인한 비효율 (Inefficiency)
* **Problem:** 아주 작은 장애물임에도 안전 비용 때문에 맵 끝까지 멀리 돌아가 배달 시간이 2배 이상 소요됨.
* **Cause:** 장애물 회피 비용이 너무 낮게 설정되어 있어, "기다리는 것"보다 "돌아가는 것"이 낫다고 판단함.
* **Solution:** **우회 거리 임계값($2.5m$) 로직 도입.**
    * 우회 거리가 $2.5m$를 초과하면 무리하게 돌아가지 않고 "비켜달라고 요청"하는 모드로 전환하여 사람처럼 융통성 있게 주행.

### 3. 모서리 충돌 문제 (Corner Collision)
* **Problem:** 코너 회전 시 로봇의 모서리가 벽에 부딪혀 주행 불가.
* **Cause:** 초기에는 로봇을 원형(Circle)으로 모델링하여, 회전 시 튀어나오는 직사각형의 모서리 부분을 감지하지 못함.
* **Solution:** **직사각형(Polygon) 충돌 모델 적용.** 매 프레임 로봇의 회전각(Heading)을 반영한 4개의 꼭짓점 좌표를 갱신하여 엄격한 충돌 검사 수행.

---

## 🚀 실행 방법 (How to run)

### 1. 요구 사항 (Prerequisites)
Python 3.x 환경이 필요하며, 아래 라이브러리를 설치해야 합니다.

```bash

pip install matplotlib numpy
