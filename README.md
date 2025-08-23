## 🛒 Heterogeneous graph model for E-commerce Product Return Prediction

### 📌 프로젝트 개요
이 프로젝트는 **이커머스 환경에서의 상품 반품 예측**을 목표로 합니다.  
반품은 기업의 수익성, 재고 관리, 고객 만족도에 큰 영향을 미치며, 이를 사전에 예측하는 것은 비즈니스 효율을 높이는 핵심 과제입니다.

본 연구에서는 **Heterogeneous Graph Neural Network (HGNN)** 기반 접근법을 도입하여,  
- **Task 1: 주문 단위 반품 예측 (Order Return Prediction)**  
- **Task 2: 상품 단위 반품 예측 (Product Return Prediction)**  

두 가지 문제를 해결하고자 하였습니다.

### 🗂️ 데이터셋
실제 이커머스 거래 데이터를 사용하였으며, 주요 통계는 다음과 같습니다.

- 총 주문 수: **849,185건**
- 고객 수: **342,039명**
- 구매된 상품 수: **2,666,262개**
- 고유 상품 수: **58,415개**
- 상품 특징: `color (642종)`, `size (29종)`, `group (32종)`

데이터는 **train / validation / test**  70/15/15 비율로 분할

### 🔍 문제 정의
- **Task 1 (주문 단위 예측)**  (Order_return_prediction.ipynb)
  각 주문에 대해 **반품 없음 (0), 일부 반품 (1), 전체 반품 (2)** 세 가지 클래스 중 하나를 예측  

- **Task 2 (상품 단위 예측)**  (Product_return_prediction.ipynb)
  각 상품에 대해 **반품 아님 (0), 반품 (1)** 이진 분류 수행  

### 🧩 방법론
#### Heterogeneous Graph 설계
- 반품 여부는 단일 상품 특성뿐만 아니라 어떤 고객이, 어떤 주문 맥락에서, 어떤 상품을 샀는지에 따라 달라지기 때문에 **관계 기반 학습의 필요성** → 고객(Customer), 주문(Order), 상품(Product) 간의 관계 구조를 그래프 형태로 모델링
- 노드별 Feature에 단순한 raw ID embedding만 쓰지 않고, **구체적인 속성과 전역적 중요도(PageRank)** 를 반영 → 더 풍부한 표현력 확보
- **노드 타입**: 주문(Order), 상품(Product), 고객(Customer), 상품-주문(Product-Order)  
- **엣지 타입**: (주문–상품), (상품–상품), (고객–상품)  
- **노드 특징**:  
  - 주문 노드 → 포함된 상품 그룹, 상품 개수, 상품 PageRank 통계값  
  - 상품 노드 → color/size/group 기반 임베딩 + PageRank  
  - 고객 노드 → 주문한 상품 수  
  - 상품-주문 노드 → 상품 임베딩 + 해당 주문 내 다른 상품 수  

#### 모델: GraphSAGE
- **5-layer GraphSAGE + ReLU activation**  
- 이웃 샘플링과 feature aggregation을 통해 **보지 못한 노드에도 일반화 가능**  
- 최종 예측:  
  - Task 1 → 3-class classification (No return / Partial return / Complete return)  
  - Task 2 → Binary classification (Return / No return)

> HAN(Heterogeneous Graph Attention Network)도 시도했지만, 실험 결과 GraphSAGE가 더 높은 정확도를 보였기 때문에 **최종 모델은 GraphSAGE 기반 접근**으로 결정하였습니다.

![Image](https://github.com/user-attachments/assets/62a2c08a-26e8-4e1f-97e4-6f50fc7d6db0)
![Image](https://github.com/user-attachments/assets/2b31e62f-a86e-443f-9068-9298bf771894)

### 📊 성능 결과
✔️ Task 1 Validation Performance
| Method                             | Accuracy (%) |
|------------------------------------|-------------|
| Random                             | 33.29       |
| SVM                                | 37.89       |
| Random Forest                                | 38.74       |
| **Heterogeneous Graph (GraphSAGE)** | **61.63**   |

✔️ Task 2 Validation Performance

| Method                             | Accuracy (%) | AUROC  |
|------------------------------------|--------------|--------|
| Random                             | 49.94        | 49.94  |
| SVM                                | 49.42        | 50.28  |
| Random Forest                      | 53.54        | 52.31  |
| Heterogeneous graph (HAN)          | 58.55        | 61.61  |
| **Heterogeneous graph (GraphSAGE)**| **63.68**    | **68.57** |


### 🚀 결론 및 성과
- **이종 그래프(Heterogeneous Graph)** 구조를 활용하여 실제 전자상거래 데이터 모델링  
- **GraphSAGE** 기반 학습으로 주문 단위와 상품 단위 반품을 동시에 예측  
- 단순 ML 및 다른 GNN 대비 더 높은 성능 확보  
- 관계 정보를 고려하고, entity 의 속성을 고려하는 것이 해당 테스크에서 주요하게 작용



*2인 팀 프로젝트 (KAIST AI506: Data Mining and Search 수업)*




