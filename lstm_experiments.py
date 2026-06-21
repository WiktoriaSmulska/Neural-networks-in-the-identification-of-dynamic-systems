import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_pipeline import prepare_data
from lstm_model import DynamicSystemLSTM
import matplotlib.pyplot as plt

import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Ueda", "Lorenz", "Rosler", "Halvorsen", "Rucklidge"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(matrix, title, filename):
    matrix = np.array(matrix)

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.title(title)
    plt.xlabel("Predykcja")
    plt.ylabel("Klasa rzeczywista")

    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")

    plt.colorbar(label="Liczba próbek")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    matrix = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)

    return accuracy, matrix, report


def train_one_lstm(
    hidden_size=64,
    num_layers=2,
    seq_len=200,
    stride=100,
    dropout=0.0,
    epochs=10,
    batch_size=64,
    lr=0.001,
    patience=3,
    min_delta=0.0001
):
    set_seed(SEED)
    print("\n" + "=" * 70)
    print("EKSPERYMENT LSTM")
    print("=" * 70)
    print(f"hidden_size={hidden_size}, num_layers={num_layers}, seq_len={seq_len}, stride={stride}")
    print(f"epochs={epochs}, batch_size={batch_size}, lr={lr}")

    train_loader, test_loader, info = prepare_data(
        seq_len=seq_len,
        stride=stride,
        train_ratio=0.8,
        batch_size=batch_size,
        normalize=True,
        verbose=False
    )

    model = DynamicSystemLSTM(
        input_size=info["num_features"],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=info["num_classes"],
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0.0
    best_matrix = None
    best_report = None
    epochs_without_improvement = 0
    stopped_epoch = epochs
    train_losses = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        accuracy, matrix, report = evaluate_model(model, test_loader)

        if accuracy > best_accuracy + min_delta:
            best_accuracy = accuracy
            best_matrix = matrix
            best_report = report
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Test accuracy: {accuracy * 100:.2f}%"
        )

        if epochs_without_improvement >= patience:
            stopped_epoch = epoch + 1
            print(f"Early stopping po epoce {stopped_epoch}")
            break

    training_time = time.time() - start_time

    print("\nWYNIKI KOŃCOWE")
    print(f"Best accuracy: {best_accuracy * 100:.2f}%")
    print(f"Czas uczenia: {training_time:.2f} s")

    print("\nClassification report:")
    print(best_report)

    print("\nConfusion matrix:")
    print(best_matrix)

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "seq_len": seq_len,
        "stride": stride,
        "epochs": epochs,
        "stopped_epoch": stopped_epoch,
        "accuracy": best_accuracy,
        "training_time": training_time,
        "confusion_matrix": best_matrix,
        "losses": train_losses,
        "dropout": dropout,
    }


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    results = []

    # 1. Wpływ hidden_size
    for hidden_size in [16, 32, 64, 128]:
        results.append(train_one_lstm(
            hidden_size=hidden_size,
            num_layers=2,
            seq_len=200,
            stride=100,
            epochs=10,
            batch_size=64,
            lr=0.001
        ))

    # 2. Wpływ liczby warstw
    for num_layers in [1, 2, 3]:
        results.append(train_one_lstm(
            hidden_size=64,
            num_layers=num_layers,
            seq_len=200,
            stride=100,
            epochs=10,
            batch_size=64,
            lr=0.001
        ))

    # 3. Wpływ długości segmentu
    for seq_len, stride in [(100, 50), (200, 100), (500, 250)]:
        results.append(train_one_lstm(
            hidden_size=64,
            num_layers=2,
            seq_len=seq_len,
            stride=stride,
            epochs=10,
            batch_size=64,
            lr=0.001
        ))
    
    # 4. Wpływ dropout
    for dropout in [0.0, 0.2, 0.5]:
        results.append(train_one_lstm(
            hidden_size=64,
            num_layers=2,
            seq_len=200,
            stride=100,
            dropout=dropout,
            epochs=10,
            batch_size=64,
            lr=0.001
        ))

    print("\n" + "=" * 70)
    print("PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW")
    print("=" * 70)

    for r in results:
        print(
            f"hidden_size={r['hidden_size']}, "
            f"num_layers={r['num_layers']}, "
            f"dropout={r['dropout']}, "
            f"seq_len={r['seq_len']}, "
            f"stride={r['stride']}, "
            f"stopped_epoch={r['stopped_epoch']}, "
            f"accuracy={r['accuracy'] * 100:.2f}%, "
            f"czas={r['training_time']:.2f}s"
        )
    
    selected_result = next(
        r for r in results
        if r["hidden_size"] == 32
        and r["num_layers"] == 2
        and r["dropout"] == 0.0
        and r["seq_len"] == 200
        and r["stride"] == 100
    )

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(selected_result["losses"]) + 1), selected_result["losses"], marker="o")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.title("LSTM - przebieg funkcji straty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lstm_loss.png", dpi=300)
    plt.show()

    plot_confusion_matrix(
        selected_result["confusion_matrix"],
        "LSTM - macierz pomyłek",
        "lstm_confusion_matrix.png"
    )