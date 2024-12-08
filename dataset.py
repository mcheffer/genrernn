import torch
from torch.utils.data import Dataset

class SparseDataset(Dataset):
    def __init__(self, sparse_matrix, targets):
        self.sparse_matrix = sparse_matrix
        self.targets = targets

    def __len__(self):
        return self.sparse_matrix.shape[0]

    def __getitem__(self, idx):
        dense_row = torch.tensor(self.sparse_matrix[idx].toarray(), dtype=torch.float32).squeeze()
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return dense_row, target