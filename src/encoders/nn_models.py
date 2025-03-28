import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from prettytable import PrettyTable

def summary(self) -> None:
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in self.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f'Total Trainable Params: {total_params}')
    if next(self.parameters()).is_cuda:
        print('Model device: cuda')
    elif next(self.parameters()).is_mps:
        print('Model device: mps')
    else:
        print('Model device: cpu')

setattr(nn.Module, "summary", summary)


class MLP(nn.Module):
    def __init__(
                self, 
                input_dim:int, 
                hidden_dims:List[int], 
                out_dim:int, 
                dropout:float
            ) -> None:
        super(MLP, self).__init__()
        layers = []
        
        # Iterate over each hidden dimension to construct the layers
        in_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(in_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hdim  # update in_dim for the next layer
        
        # Add the final output layer
        layers.append(nn.Linear(in_dim, out_dim))
        
        # Wrap the layers in a Sequential module
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)