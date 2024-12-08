import torch.nn as nn

class GenreRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GenreRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence length dimension
        _, (hidden, _) = self.rnn(x)
        out = self.fc(hidden[-1])  # Use the last hidden state
        return out