import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def create_synthetic_data():
    X = np.random.rand(1000, 10).astype(np.float32)
    y = (X.sum(axis=1) > 5).astype(np.int64)
    return X, y


def build_model(trial):
    input_size = 10
    hidden_size = trial.suggest_int("hidden_size", 16, 128)
    output_size = 2

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    return model


def objective(trial):
    X, y = create_synthetic_data()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = build_model(trial)
    criterion = nn.CrossEntropyLoss()
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(10):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best trial:")
best_trial = study.best_trial
print(f"  Accuracy: {best_trial.value}")
print("  Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
