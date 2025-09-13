import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class customFFN(nn.Module):

  def __init__(self, num_features, num_hidden_layers, width, output_size):
    super().__init__()
    # set the depth and width of the network as attributes
    self.num_hidden_layers = num_hidden_layers
    self.output_size = output_size
    # init the layers dict
    self.layers = nn.ModuleDict()
    # add input layer
    self.layers["input"] = nn.Linear(num_features, width)
    # add hidden layers
    for i in range(num_hidden_layers):
      self.layers[f"hidden_{i}"] = nn.Linear(width, width)
    # add output layer
    self.layers["output"] = nn.Linear(width, output_size)

  # the forward propapagation function
  def forward(self, x):
    x = self.layers["input"](x)
    # chose relu for activation in the hidden layers, also arbitrarily
    x = F.relu(x)
    for i in range(self.num_hidden_layers):
      x = self.layers[f"hidden_{i}"](x)
      x = F.relu(x)
    output = self.layers["output"](x)
    return output



class customLSTM(nn.Module):

  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super().__init__()
    # set hidden size in parameters
    self.hidden_size = hidden_size
    # init layers dict
    self.layers = nn.ModuleDict()
    # add the lstm layer
    self.layers["lstm"] = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    # add the linear (output) layer
    self.layers["linear"] = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    # pass through lstm
    lstm_out, _ = self.layers["lstm"](x)
    # select only last timestep, by default lstm returns all
    lstm_out = lstm_out[:, -1, :]
    # pass through output
    output = self.layers["linear"](lstm_out)
    return output



def reshape_for_LSTM(numpyarr_df, sequence_length):
  X = []
  target_column_index = numpyarr_df.shape[1] - 1
  for i in range(len(numpyarr_df) - sequence_length):
    input_sequence = numpyarr_df[i:i+sequence_length]
    X.append(input_sequence)
  return np.array(X)