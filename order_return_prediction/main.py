import argparse
import torch
from src.models import GNN, HAN, GAT
from src.train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gnn", choices=["gnn", "han", "gat"])
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    data = torch.load("processed_data.pt")

    if args.model == "gnn":
        model = GNN(out_channels=3, metadata=data.metadata())
    elif args.model == "han":
        model = HAN(dim_in=-1, dim_out=3, metadata=data.metadata())
    elif args.model == "gat":
        model = GAT(dim_h=128, dim_out=3, metadata=data.metadata())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, model = data.to(device), model.to(device)

    train_model(model, data, optimizer, num_epochs=args.epochs)

if __name__ == "__main__":
    main()
