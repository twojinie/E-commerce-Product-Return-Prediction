import argparse
import torch
from src.models import GNN, HAN
from src.train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gnn", choices=["gnn","gat","han"])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    data = torch.load("processed_data.pt")
    if args.model=="gnn":
        model = GNN(out_channels=1, metadata=data.metadata(), conv_type="sage")
    elif args.model=="gat":
        model = GNN(out_channels=1, metadata=data.metadata(), conv_type="gat")
    elif args.model=="han":
        model = HAN(dim_in=-1, metadata=data.metadata(), out_channels=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, model = data.to(device), model.to(device)

    train_model(model, data, optimizer, num_epochs=args.epochs)

if __name__=="__main__":
    main()
