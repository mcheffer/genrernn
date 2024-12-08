from data_preprocessing import load_and_preprocess_data
from dataset import SparseDataset
from model import GenreRNN
from training import train_model, evaluate_model,plot_confusion_matrix
from optimization import objective
import optuna
import torch
from torch.utils.data import DataLoader

# Load and preprocess data
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, vectorizer = load_and_preprocess_data('song_lyrics_med.csv')

# Define constants
input_size = X_train.shape[1]
num_classes = len(label_encoder.classes_)

# Hyperparameter optimization
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_size, num_classes), n_trials=20)
best_params = study.best_params
print(f"Best Parameters: {best_params}")

# Train best model
best_model = GenreRNN(input_size, best_params['hidden_size'], num_classes)
best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params['lr'])
train_loader = DataLoader(SparseDataset(X_train, y_train), batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(SparseDataset(X_val, y_val), batch_size=best_params['batch_size'], shuffle=False)
train_model(best_model, train_loader, val_loader, best_optimizer, torch.nn.CrossEntropyLoss(), num_epochs=10)

# Evaluate on test set
test_loader = DataLoader(SparseDataset(X_test, y_test), batch_size=best_params['batch_size'], shuffle=False)
test_loss, test_f1, all_targets, all_predictions = evaluate_model(best_model, test_loader, torch.nn.CrossEntropyLoss())

print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.2f}%")

# Plot the normalized confusion matrix
plot_confusion_matrix(all_targets, all_predictions, label_encoder.classes_)
