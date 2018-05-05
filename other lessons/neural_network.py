from torch import nn
from torch.nn.functional import relu, log_softmax


class Network(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout=0.5):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        self.hidden_layers.extend(
            [nn.Linear(n_input, n_output) for n_input, n_output in zip(hidden_layers[:-1], hidden_layers[1:])])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = relu(x)
            x = self.dropout(x)

        x = self.output(x)

        return log_softmax(x, dim=1)
