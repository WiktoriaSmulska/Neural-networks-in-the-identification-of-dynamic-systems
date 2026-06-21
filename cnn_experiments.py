import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data_pipeline import prepare_data
from cnn_model import DynamicSystemCNN
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np

CLASS_NAMES = ["Ueda", "Lorenz", "Rosler", "Halvorsen", "Rucklidge"]


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


def train_one_cnn(
    filters=32,
    kernel_size=5,
    dropout=0.2,
    seq_len=200,
    epochs=10,
    batch_size=64,
    lr=0.001,
    patience=3,
    min_delta=0.0001
):
    print("\n" + "=" * 70)
    print("EKSPERYMENT CNN")
    print("=" * 70)
    print(
        f"filters={filters}, kernel_size={kernel_size}, dropout={dropout}, "
        f"seq_len={seq_len}, stride={seq_len // 2}"
    )
    print(f"epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={patience}")

    train_loader, test_loader, info = prepare_data(
        seq_len=seq_len,
        stride=seq_len // 2,
        train_ratio=0.8,
        batch_size=batch_size,
        normalize=True,
        verbose=False
    )

    model = DynamicSystemCNN(
        input_size=info["num_features"],
        num_classes=info["num_classes"],
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        "filters": filters,
        "kernel_size": kernel_size,
        "dropout": dropout,
        "seq_len": seq_len,
        "stride": seq_len // 2,
        "epochs": epochs,
        "stopped_epoch": stopped_epoch,
        "accuracy": best_accuracy,
        "training_time": training_time,
        "confusion_matrix": best_matrix,
        "losses": train_losses,
    }


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    results = []

    # 1. Wpływ długości segmentu
    for seq_len in [100, 200, 500]:
        results.append(train_one_cnn(
            filters=32,
            kernel_size=5,
            dropout=0.2,
            seq_len=seq_len,
            epochs=10,
            batch_size=64,
            lr=0.001,
            patience=3
        ))

    # 2. Wpływ liczby filtrów
    for filters in [16, 32, 64]:
        results.append(train_one_cnn(
            filters=filters,
            kernel_size=5,
            dropout=0.2,
            seq_len=200,
            epochs=10,
            batch_size=64,
            lr=0.001,
            patience=3
        ))

    # 3. Wpływ rozmiaru kernela
    for kernel_size in [3, 5, 7]:
        results.append(train_one_cnn(
            filters=32,
            kernel_size=kernel_size,
            dropout=0.2,
            seq_len=200,
            epochs=10,
            batch_size=64,
            lr=0.001,
            patience=3
        ))

    # 4. Wpływ dropout
    for dropout in [0.0, 0.2, 0.5]:
        results.append(train_one_cnn(
            filters=32,
            kernel_size=5,
            dropout=dropout,
            seq_len=200,
            epochs=10,
            batch_size=64,
            lr=0.001,
            patience=3
        ))

    print("\n" + "=" * 70)
    print("PODSUMOWANIE WSZYSTKICH EKSPERYMENTÓW CNN")
    print("=" * 70)

    for r in results:
        print(
            f"filters={r['filters']}, "
            f"kernel_size={r['kernel_size']}, "
            f"dropout={r['dropout']}, "
            f"seq_len={r['seq_len']}, "
            f"stride={r['stride']}, "
            f"stopped_epoch={r['stopped_epoch']}, "
            f"accuracy={r['accuracy'] * 100:.2f}%, "
            f"czas={r['training_time']:.2f}s"
        )
    
    best_result = max(results, key=lambda r: (r["accuracy"], -r["training_time"]))

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(best_result["losses"]) + 1), best_result["losses"], marker="o")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.title("CNN - przebieg funkcji straty")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cnn_loss.png", dpi=300)
    plt.show()

    plot_confusion_matrix(
        best_result["confusion_matrix"],
        "CNN - macierz pomyłek",
        "cnn_confusion_matrix.png"
    )