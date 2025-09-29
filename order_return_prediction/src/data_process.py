import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from collections import defaultdict
from itertools import combinations
import csv

def load_raw_data():
    task_data = pd.read_csv("task1_data.txt", sep=",")
    train_label = pd.read_csv("task1_train_label.txt", sep="\t", header=None, names=["order","label"])
    valid_label = pd.read_csv("task1_valid_label.txt", sep="\t", header=None, names=["order","label"])
    test_query = pd.read_csv("task1_test_query.txt", header=None, names=["order"])

    train_data = task_data.merge(train_label, on="order")
    valid_data = task_data.merge(valid_label, on="order")
    test_data  = task_data.merge(test_query, on="order")
    return train_data, valid_data, test_data

def load_dicts():
    def read_dict(path):
        d = defaultdict(list)
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                d[int(row[0])] = [int(x) for x in row[1:]]
        return d
    product_info  = read_dict("New_data/product_info.csv")
    group_info    = read_dict("New_data/group_product_dict.csv")
    size_info     = read_dict("New_data/size_product_dict.csv")
    customer_info = read_dict("New_data/customer_info.csv")
    return product_info, group_info, size_info, customer_info

def build_graph(train_data, valid_data, test_data, product_info, group_info, size_info, customer_info):
    product_idx = sorted(set(train_data["product"]) | set(valid_data["product"]) | set(test_data["product"]))
    product_x = []
    for idx in product_idx:
        onehot = [0]*32
        group_val = product_info[idx][2]
        onehot[group_val] += 1
        product_x.append(onehot)

    edge_source, edge_destination = [], []
    product_edge_source, product_edge_destination = [], []
    order_x, order_y = [], []

    # train data
    for order, group in train_data.groupby("order"):
        p_list = group["product"].tolist()
        group_feature = np.zeros(32)
        for p in p_list:
            edge_source.append(order)
            edge_destination.append(p)
            group_feature += np.array(product_x[p])
        for u,v in combinations(p_list,2):
            product_edge_source.append(u)
            product_edge_destination.append(v)
        group_feature = list(group_feature) + [len(p_list)]
        order_x.append(group_feature)
        order_y.append(np.eye(3)[group["label"].iloc[0]])

    train_num = len(order_y)

    # valid data
    for order, group in valid_data.groupby("order"):
        p_list = group["product"].tolist()
        group_feature = np.zeros(32)
        for p in p_list:
            edge_source.append(order)
            edge_destination.append(p)
            group_feature += np.array(product_x[p])
        for u,v in combinations(p_list,2):
            product_edge_source.append(u)
            product_edge_destination.append(v)
        group_feature = list(group_feature) + [len(p_list)]
        order_x.append(group_feature)
        order_y.append(np.eye(3)[group["label"].iloc[0]])

    val_num = len(order_y) - train_num

    # test data
    for order, group in test_data.groupby("order"):
        p_list = group["product"].tolist()
        group_feature = np.zeros(32)
        for p in p_list:
            edge_source.append(order)
            edge_destination.append(p)
            group_feature += np.array(product_x[p])
        for u,v in combinations(p_list,2):
            product_edge_source.append(u)
            product_edge_destination.append(v)
        group_feature = list(group_feature) + [len(p_list)]
        order_x.append(group_feature)
        order_y.append([0,0,0])

    test_num = len(order_y) - train_num - val_num

    data = HeteroData()
    data["order"].x = torch.tensor(order_x).float()
    data["product"].x = torch.tensor(product_x).float()
    data["order"].y = torch.tensor(order_y).float()
    data["order","contains","product"].edge_index = torch.tensor([edge_source,edge_destination])
    data["product","same_order","product"].edge_index = torch.tensor([product_edge_source,product_edge_destination])
    data = T.ToUndirected()(data)

    data["order"].train_mask = torch.zeros(data["order"].num_nodes, dtype=torch.bool)
    data["order"].val_mask   = torch.zeros_like(data["order"].train_mask)
    data["order"].test_mask  = torch.zeros_like(data["order"].train_mask)
    data["order"].train_mask[:train_num] = 1
    data["order"].val_mask[train_num:train_num+val_num] = 1
    data["order"].test_mask[train_num+val_num:] = 1
    return data
