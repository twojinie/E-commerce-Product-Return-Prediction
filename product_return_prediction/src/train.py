import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)["product_order"]
        pred = (out[mask].cpu().numpy() >= 0.5).astype(int)
        labels = data["product_order"].y[mask].cpu().numpy()
        acc = accuracy_score(labels, pred)
        roc = roc_auc_score(labels, out[mask].cpu().numpy())
        return acc, roc

def train_model(model, data, optimizer, num_epochs=100):
    best_val_acc, best_epoch = 0, 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)["product_order"]
        mask = data["product_order"].train_mask
        loss = F.binary_cross_entropy(out[mask], data["product_order"].y[mask])
        loss.backward(); optimizer.step()

        train_acc, train_roc = test(model, data, data["product_order"].train_mask)
        val_acc, val_roc = test(model, data, data["product_order"].val_mask)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(),"best_model.pt")

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss={loss:.4f} TrainAcc={train_acc:.3f} ValAcc={val_acc:.3f} ROC={val_roc:.3f}")

    print(f"✅ Best Val Acc: {best_val_acc:.3f} at epoch {best_epoch}")
