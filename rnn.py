# SET UP BASICS
import torch.nn as nn
import torch.nn.functional as F

# CREATE NN
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size) # applies the linear equation (weights and biases)
        self.softmax = nn.LogSoftmax(dim = 1) # does log soft max (kinda like relu but different way)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor) # input to hidden layer
        output = self.hidden_to_output(hidden[0]) # hidden layer to output
        output = self.softmax(output) # apply soft max to output

        return output