# Product Return Prediction (Graph Neural Networks, PyTorch Geometric)

This project implements **product return prediction** using **heterogeneous graphs** of orders, products, and customers.  
Models include **GraphSAGE (SAGEConv/GATConv)** and **HAN** with PyTorch Geometric.

## Features
- End-to-end pipeline: raw txt → feature engineering → `HeteroData`
- Node types: product_order, product, customer
- Edge types: product_order–product, product_order–customer, product–product (same_order)
- Binary classification (return or not) on product_order nodes
- Multiple GNN architectures (GraphSAGE, HAN)
- Metrics: Accuracy & ROC-AUC

---

## 📂 Project Structure
```
product-return-prediction/
├── New_data/                     # supporting dict files
│   ├── product_info.csv
│   ├── order_product_dict.csv
│   ├── customer_return.csv
│   └── customer_info.csv
├── task2_data.txt
├── task2_train_label.txt
├── task2_valid_label.txt
├── task2_test_query.txt
├── preprocess.py                 # run preprocessing → processed_data.pt
├── main.py                       # training entrypoint
├── requirements.txt
├── README.md
└── src/
    ├── data_process.py           # data loading + feature eng + graph building
    ├── models.py                 # GNN (SAGEConv/GATConv), HAN
    ├── train.py                  # training loop (BCE, accuracy, ROC-AUC)
    └── __init__.py
```

---

## 🚀 Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
- Place the following raw files in the **project root**:
  - `task2_data.txt`
  - `task2_train_label.txt`
  - `task2_valid_label.txt`
  - `task2_test_query.txt`
- Place the supporting dictionary CSVs inside **New_data/**:
  - `product_info.csv`
  - `order_product_dict.csv`
  - `customer_return.csv`
  - `customer_info.csv`

### 3. Preprocess
```bash
python preprocess.py
```
This will generate `processed_data.pt`.

### 4. Train & Evaluate
```bash
# GraphSAGE
python main.py --model gnn --epochs 300

# HAN
python main.py --model han --epochs 300
```
