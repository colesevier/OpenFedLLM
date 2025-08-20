# fedllm_poc.py

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
import copy

# -------- Dummy dataset --------
class DummyDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# -------- Federated client --------
class FederatedClient:
    def __init__(self, name, dataset, model, lr=5e-5):
        self.name = name
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def train(self, epochs=1, batch_size=4):
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        losses = []
        correct, total = 0, 0
        for _ in range(epochs):
            for batch in loader:
                self.optimizer.zero_grad()
                outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"}, labels=batch["labels"])
                loss = outputs.loss
                logits = outputs.logits
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        avg_loss = np.mean(losses)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy, copy.deepcopy(self.model.state_dict())

# -------- Federated averaging --------
def federated_average(global_model, client_states):
    new_state = copy.deepcopy(global_model.state_dict())
    for key in new_state.keys():
        new_state[key] = torch.mean(torch.stack([cs[key].float() for cs in client_states]), dim=0)
    global_model.load_state_dict(new_state)
    return global_model

# -------- Main experiment --------
def main():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Fake client data
    texts_client1 = ["I love this!", "This is great!", "So happy", "Best day ever"]
    labels_client1 = [1, 1, 1, 1]

    texts_client2 = ["I hate this", "This is terrible", "Awful experience", "Worst ever"]
    labels_client2 = [0, 0, 0, 0]

    dataset1 = DummyDataset(tokenizer, texts_client1, labels_client1)
    dataset2 = DummyDataset(tokenizer, texts_client2, labels_client2)

    client1 = FederatedClient("client1", dataset1, base_model)
    client2 = FederatedClient("client2", dataset2, base_model)

    global_model = copy.deepcopy(base_model)

    # Tracking
    rounds = 3
    client1_history, client2_history, global_history = [], [], []

    for rnd in range(1, rounds + 1):
        print(f"\n[Round {rnd}]")

        # Train client 1
        print(f" Training on {client1.name}...")
        loss1, acc1, state1 = client1.train()
        client1_history.append((loss1, acc1))
        print(f"  -> loss={loss1:.4f}, acc={acc1:.4f}")

        # Train client 2
        print(f" Training on {client2.name}...")
        loss2, acc2, state2 = client2.train()
        client2_history.append((loss2, acc2))
        print(f"  -> loss={loss2:.4f}, acc={acc2:.4f}")

        # FedAvg
        print(" Aggregating global model...")
        global_model = federated_average(global_model, [state1, state2])

        # Evaluate global model
        global_loss, global_acc = evaluate_global(global_model, [dataset1, dataset2])
        global_history.append((global_loss, global_acc))
        print(f" Global model -> loss={global_loss:.4f}, acc={global_acc:.4f}")

    print("\n[Demo complete] Global model trained with FedAvg across 2 clients.")

    # Plot results
    plot_results(client1_history, client2_history, global_history)


# -------- Evaluation --------
def evaluate_global(model, datasets):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=4, shuffle=False)
            for batch in loader:
                outputs = model(**{k: v for k, v in batch.items() if k != "labels"}, labels=batch["labels"])
                losses.append(outputs.loss.item())
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
    return np.mean(losses), correct / total if total > 0 else 0


# -------- Visualization --------
def plot_results(client1_history, client2_history, global_history):
    rounds = range(1, len(global_history) + 1)

    # Per-client
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rounds, [l for l, _ in client1_history], label="Client1 Loss")
    plt.plot(rounds, [l for l, _ in client2_history], label="Client2 Loss")
    plt.plot(rounds, [l for l, _ in global_history], label="Global Loss", linestyle="--")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss per Round")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rounds, [a for _, a in client1_history], label="Client1 Acc")
    plt.plot(rounds, [a for _, a in client2_history], label="Client2 Acc")
    plt.plot(rounds, [a for _, a in global_history], label="Global Acc", linestyle="--")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Round")
    plt.legend()

    plt.tight_layout()
    plt.savefig("fedllm_training_results.png")
    plt.show()


if __name__ == "__main__":
    main()
