import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from collections import defaultdict
from itertools import combinations
import csv
import networkx as nx
from networkx import pagerank

def load_raw_data():
    task_data = pd.read_csv("task2_data.txt", sep=",")
    train_label = pd.read_csv("task2_train_label.txt", sep="\t", header=None, names=["order","product","label"])
    valid_label = pd.read_csv("task2_valid_label.txt", sep="\t", header=None, names=["order","product","label"])
    test_query = pd.read_csv("task2_test_query.txt", sep="\t", header=None, names=["order","product"])

    train_data = task_data.merge(train_label, on=["order","product"])
    valid_data = task_data.merge(valid_label, on=["order","product"])
    test_data  = task_data.merge(test_query, on=["order","product"])
    return train_data, valid_data, test_data

def load_dicts():
    def read_dict(path):
        d = defaultdict(list)
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                d[int(row[0])] = [int(x) for x in row[1:]]
        return d
    product_info    = read_dict("New_data/product_info.csv")
    order_prod_dict = read_dict("New_data/order_product_dict.csv")
    customer_return = read_dict("New_data/customer_return.csv")
    customer_info   = read_dict("New_data/customer_info.csv")
    return product_info, order_prod_dict, customer_return, customer_info

def build_graph(train_data, valid_data, test_data, product_info, order_product_dict, customer_return, customer_info):
    # PageRank feature
    G = nx.Graph()
    for key, value in order_product_dict.items():
        p_list = value
        for u,v in combinations(p_list,2):
            if G.has_edge(u,v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u,v,weight=1)
    pr = pagerank(G, max_iter=100, weight="weight")
    pr_max, pr_min, pr_mean = max(pr.values()), min(pr.values()), np.mean(list(pr.values()))
    scaled_pr = {i: (pr[i]-pr_min)/(pr_max-pr_min) if i in pr else pr_mean for i in range(max(order_product_dict.keys())+1)}

    # product features
    product_idx = sorted(set(train_data["product"]) | set(valid_data["product"]) | set(test_data["product"]))
    product_x = []
    for idx in product_idx:
        onehot_g = [0]*32
        onehot_s = [0]*29
        onehot_c = [0]*65
        color, size, group = product_info[idx]
        onehot_g[group] += 1
        onehot_s[size] += 1
        onehot_c[color//10] += 1
        product_x.insert(idx, onehot_g+onehot_s+onehot_c+[scaled_pr[idx]])

    edge_source, edge_dest = [], []
    prod_edge_src, prod_edge_dst = [], []
    prod_cust_src, prod_cust_dst = [], []
    prod_order_x, prod_order_y = [], []

    idx_counter = 0
    for df, mode in [(train_data,"train"),(valid_data,"valid"),(test_data,"test")]:
        for _,row in df.iterrows():
            order, product, label, size, group, color, customer = row["order"], row["product"], row.get("label",0), row["size"], row["group"], row["color"], row["customer"]
            p_list = order_product_dict[order]

            onehot_g = [0]*32; onehot_s = [0]*29; onehot_c = [0]*65; onehot_p = [0]*30
            onehot_g[group]+=1; onehot_s[size]+=1; onehot_c[color//10]+=1
            onehot_p[min(len(p_list)-1,29)] += 1

            feat = np.concatenate((onehot_g, onehot_s, onehot_c, onehot_p, [scaled_pr[product]]))
            prod_order_x.append(feat)
            prod_order_y.append([label] if mode!="test" else [0])

            edge_source.append(idx_counter); edge_dest.append(product)
            prod_cust_src.append(idx_counter); prod_cust_dst.append(customer)
            for u,v in combinations(p_list,2):
                prod_edge_src.append(u); prod_edge_dst.append(v)
            idx_counter+=1

    train_num = len(train_data); val_num = len(valid_data)

    data = HeteroData()
    data["product_order"].x = torch.tensor(prod_order_x).float()
    data["product"].x = torch.tensor(product_x).float()
    # simple customer embedding (ID-based one-hot)
    num_customers = max(customer_info.keys())+1
    data["customer"].x = torch.eye(num_customers).float()

    data["product_order"].y = torch.tensor(prod_order_y).float()
    data["product_order","with","product"].edge_index = torch.tensor([edge_source,edge_dest])
    data["product_order","with","customer"].edge_index = torch.tensor([prod_cust_src,prod_cust_dst])
    data["product","same_order","product"].edge_index = torch.tensor([prod_edge_src,prod_edge_dst])
    data = T.ToUndirected()(data)
    data = T.NormalizeFeatures()(data)

    n_total = data["product_order"].num_nodes
    data["product_order"].train_mask = torch.zeros(n_total, dtype=torch.bool); data["product_order"].train_mask[:train_num]=1
    data["product_order"].val_mask = torch.zeros(n_total, dtype=torch.bool); data["product_order"].val_mask[train_num:train_num+val_num]=1
    data["product_order"].test_mask = torch.zeros(n_total, dtype=torch.bool); data["product_order"].test_mask[train_num+val_num:]=1
    return data
