import torch
from src.data_process import load_raw_data, load_dicts, build_graph

if __name__ == "__main__":
    train_data, valid_data, test_data = load_raw_data()
    product_info, order_product_dict, customer_return, customer_info = load_dicts()
    data = build_graph(train_data, valid_data, test_data,
                       product_info, order_product_dict, customer_return, customer_info)
    torch.save(data, "processed_data.pt")
    print("✅ processed_data.pt saved.")
