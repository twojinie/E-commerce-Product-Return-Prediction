# Heterogeneous graph model for E-commerce Product Return Prediction

### 📌 프로젝트 개요
본 프로젝트는 Graph Mining 기법을 활용하여 급변하는 전자상거래 환경에서 상품 반품을 예측하고, 이를 통해 비즈니스 운영 효율과 고객 만족도를 극대화하는 것을 목표로 함

### 💡 핵심 목표
전자상거래 플랫폼에서 발생하는 상품 반품 패턴을 정확하게 예측하여 재고 관리, 가격 전략, 맞춤형 고객 서비스 개선에 활용. 따라서, 궁극적으로는 반품을 줄이고 더 효율적이며 수익성 높은 전자상거래 환경을 만드는 데 기여

#### 두 가지 주요 과제
1. 주문 단위 반품 예측 (Task 1): 특정 주문 전체가 반품될 확률을 예측
2. 개별 상품 단위 반품 예측 (Task 2): 주문 내에서 개별 상품이 반품될 확률을 예측

### 📊 데이터
실제 전자상거래 데이터를 활용하며, 각 주문과 상품에 대한 상세 정보(주문 ID, 상품 ID, 고객 ID, 상품 색상/사이즈 ID 등)를 포함한 약 85만 건의 주문과 260만 개 이상의 상품 데이터를 사용

### ✔️ 방법론
- 이종 그래프(Heterogeneous Graph) 활용: 반품 여부는 단일 상품 특성뿐만 아니라 어떤 고객이, 어떤 주문 맥락에서, 어떤 상품을 샀는지에 따라 달라지기 때문에 **관계 기반 학습의 필요성** -> 고객(Customer), 주문(Order), 상품(Product) 간의 관계 구조를 그래프 형태로 모델링
- 노드별 Feature 설계: 단순한 raw ID embedding만 쓰지 않고, **구체적인 속성과 전역적 중요도(PageRank)** 를 반영 → 더 풍부한 표현력 확보
- GraphSAGE 모델 선택: 이웃 정보를 샘플링 후 Aggregation → 대규모 그래프에서도 확장성 확보 , HAN(Heterogeneous Attention Network)보다도 더 나은 성능을 보임

![Image](https://github.com/user-attachments/assets/62a2c08a-26e8-4e1f-97e4-6f50fc7d6db0)
![Image](https://github.com/user-attachments/assets/2b31e62f-a86e-443f-9068-9298bf771894)
  
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


#### 🚀 결론 및 성과
- GraphSAGE 기반 heterogeneous graph 모델이 기존 방법 대비 우수한 성능을 보임
- 관계 정보를 고려하고, entity 의 속성을 고려하는 것이 해당 테스크에서 주요하게 작용함



*2인 팀 프로젝트 (KAIST AI506: Data Mining and Search 수업)*




