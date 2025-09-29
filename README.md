## 🛒 Heterogeneous graph model for E-commerce Product Return Prediction

This project tackles **product return prediction** in e-commerce, a key task affecting profitability, inventory, and customer satisfaction.  
We apply **Heterogeneous Graph Neural Networks (HGNNs)** to capture complex relations between **orders, products, and customers**.

### 📌 Tasks
- **Task 1: Order-level Return Prediction**  
  Predict whether an order is **No return (0) / Partial return (1) / Complete return (2)**  
  → [Task 1 README](./order_return_prediction/README.md)

- **Task 2: Product-level Return Prediction**  
  Predict whether a product is **Returned (1) or Not (0)**  
  → [Task 2 README](./product_return_prediction/README.md)

---

### - Dataset Overview
- **849,185 orders**, **342,039 customers**
- **2.66M purchased products**, covering **58,415 unique items**
- Product attributes: `color (642 types)`, `size (29 types)`, `group (32 types)`
- Split: **70% train / 15% validation / 15% test**

---

### - Methodology
- **Heterogeneous Graph Construction**  
  - Nodes: Orders, Products, Customers, Product-Order pairs  
  - Edges: (Order–Product), (Product–Product), (Customer–Product)  
  - Features:  
    - Orders → product groups, item count, PageRank stats  
    - Products → color/size/group embeddings + PageRank  
    - Customers → purchase history (unique product counts)  
    - Product-Orders → product embedding + order context

- **Models**  
  - **GraphSAGE (5-layer)**: best-performing GNN  
  - HAN also tested, but underperformed vs GraphSAGE  
  - Compared with traditional baselines (SVM, Random Forest)

![Image](https://github.com/user-attachments/assets/62a2c08a-26e8-4e1f-97e4-6f50fc7d6db0)
![Image](https://github.com/user-attachments/assets/2b31e62f-a86e-443f-9068-9298bf771894)

---

### - Results
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

---

### 🚀 Key Takeaways
- HGNNs effectively capture **structural relationships + node attributes** in e-commerce data  
- **GraphSAGE generalized better** than HAN and classical ML  
- Demonstrated the value of graph-based modeling for **real-world business prediction problems**


*2-person team project (KAIST AI506: Data Mining and Search)*




