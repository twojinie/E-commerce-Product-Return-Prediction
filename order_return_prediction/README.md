# Order Return Prediction (Graph Neural Networks, PyTorch Geometric)

This project implements **order return prediction** using **heterogeneous graphs** built from e-commerce data.  
Models include **GraphSAGE, HAN, and GAT** with PyTorch Geometric.

## Features
- Data preprocessing (raw txt → features → HeteroData)
- Multiple GNN architectures (GraphSAGE, HAN, GAT)
- Train/validation/test split with masks
- Reproducible training & evaluation

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess data
Place raw files (`task1_data.txt`, `task1_train_label.txt`, `task1_valid_label.txt`, `task1_test_query.txt`) and supporting CSVs under `New_data/`.  
Then run:
```bash
python preprocess.py
```

This will generate `processed_data.pt`.

### 3. Train & Evaluate
```bash
# GraphSAGE
python main.py --model gnn --epochs 200

# HAN
python main.py --model han --epochs 200

# GAT
python main.py --model gat --epochs 200
```

