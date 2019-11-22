"""
bayesian_nn.py
classe BayesianNN
Esqueleto de uma Rede Neural Artificial apenas com o método foward
Ítalo Della Garza Silva
Adaptado de https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd
22 nov. 2019
"""
import torch.nn as nn
import torch.nn.functional as F

class BayesianNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        # Cria uma rede simples, com apenas uma camada escondida
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Faz o foward padrão de uma rede neural
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output
