import optuna
import torch
from torch.utils.data import DataLoader
from training import train_model
from model import GenreRNN
from dataset import SparseDataset

def objective(trial, X_train, y_train, X_val, y_val, input_size, num_classes):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])

    model = GenreRNN(input_size, hidden_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(SparseDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SparseDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    try:
        best_val_loss = train_model(model, train_loader, val_loader, optimizer, torch.nn.CrossEntropyLoss(), num_epochs=2, patience=3)
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

    return best_val_loss
