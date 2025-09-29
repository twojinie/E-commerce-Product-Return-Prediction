import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)["order"]
        pred = out.argmax(dim=-1)[mask].cpu().tolist()
        answer = data["order"].y.argmax(dim=-1)[mask].cpu().tolist()
        return accuracy_score(answer, pred)

def train_model(model, data, optimizer, num_epochs=100):
    best_val_acc, best_epoch = 0, 0
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)["order"]
        mask = data["order"].train_mask
        loss = F.cross_entropy(out[mask], data["order"].y[mask])
        loss.backward()
        optimizer.step()

        train_acc = test(model, data, data["order"].train_mask)
        val_acc = test(model, data, data["order"].val_mask)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), "best_model.pt")

        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss={loss:.4f} TrainAcc={train_acc:.3f} ValAcc={val_acc:.3f}")

    print(f"✅ Best Val Acc: {best_val_acc:.3f} at epoch {best_epoch}")
