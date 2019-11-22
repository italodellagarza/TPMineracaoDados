"""
nn_methods.py
classe NeuralNetworkMethods
Responsável pela inclusão dos métodos Bayesianos na rede neural
Ítalo Della Garza Silva
Adaptado de https://towardsdatascience.com/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd
22 nov. 2019
"""
import torch

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import torch.nn as nn

from bayesian_nn import BayesianNN

# Aqui, carregam-se algumas operações úteis da biblioteca torch.nn
log_softmax = nn.LogSoftmax(dim=1)
softplus = torch.nn.Softplus()

class NeuralNetworkMethods:

    def __init__(self, input_size, hidden_size, output_size, number):
        # Cria uma rede neural bayesiana a partir do esqueleto definido em bayesian_nn.py
        self.net = BayesianNN(input_size, hidden_size, output_size)
        self.number = number

    def model(self, x_data, y_data):
        # Responsável por definir o modelo no Pyro usado para a inferência. Define como a saída será gerada.
        fc1w_prior = Normal(loc=torch.zeros_like(self.net.fc1.weight), scale=torch.ones_like(self.net.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(self.net.fc1.bias), scale=torch.ones_like(self.net.fc1.bias))
        
        outw_prior = Normal(loc=torch.zeros_like(self.net.out.weight), scale=torch.ones_like(self.net.out.weight))
        outb_prior = Normal(loc=torch.zeros_like(self.net.out.bias), scale=torch.ones_like(self.net.out.bias))
        
        priors = {('fc1.weight' + str(self.number)): fc1w_prior, 
                  ('fc1.bias' + str(self.number)): fc1b_prior, 
                  ('out.weight' + str(self.number)): outw_prior,
                  ('out.bias' + str(self.number)): outb_prior}
        
        # parâmetros do módulo de elevação para variáveis aleatórias amostradas das prioris
        lifted_module = pyro.random_module(("module" + str(self.number)), self.net, priors)
        
        # amostra um regressor (que também amostra os pesos e o bias)
        lifted_red_model = lifted_module()
        
        lhat = log_softmax(lifted_red_model(x_data))
        
        pyro.sample(("obs" + str(self.number)), Categorical(logits=lhat), obs=y_data)


    def guide(self, x_data, y_data):
        # Responsável por inicializar a distribuição que será otimizada depois, 
        # através da inferência variacional.

        # Prioris da distribuição de pesos da primeira camada
        fc1w_mu = torch.randn_like(self.net.fc1.weight)
        fc1w_sigma = torch.randn_like(self.net.fc1.weight)
        fc1w_mu_param = pyro.param(("fc1w_mu" + str(self.number)), fc1w_mu)
        fc1w_sigma_param = softplus(pyro.param(("fc1w_sigma" + str(self.number)), fc1w_sigma))
        fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)
        
        # Prioris da distribuição de bias da primeira camada
        fc1b_mu = torch.randn_like(self.net.fc1.bias)
        fc1b_sigma = torch.randn_like(self.net.fc1.bias)
        fc1b_mu_param = pyro.param(("fc1b_mu" + str(self.number)), fc1b_mu)
        fc1b_sigma_param = softplus(pyro.param(("fc1b_sigma" + str(self.number)), fc1b_sigma))
        fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)
        
        # Prioris da distribuição de pesos da segunda camada
        outw_mu = torch.randn_like(self.net.out.weight)
        outw_sigma = torch.randn_like(self.net.out.weight)
        outw_mu_param = pyro.param(("outw_mu" + str(self.number)), outw_mu)
        outw_sigma_param = softplus(pyro.param(("outw_sigma" + str(self.number)), outw_sigma))
        outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param)
        
        # Prioris da distribuição de bias da segunda camada
        outb_mu = torch.randn_like(self.net.out.bias)
        outb_sigma = torch.randn_like(self.net.out.bias)
        outb_mu_param = pyro.param(("outb_mu" + str(self.number)), outb_mu)
        outb_sigma_param = softplus(pyro.param(("outb_sigma" + str(self.number)), outb_sigma))
        outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
        
        priors = {('fc1.weight' + str(self.number)): fc1w_prior, 
                  ('fc1.bias' + str(self.number)): fc1b_prior, 
                  ('out.weight' + str(self.number)): outw_prior,
                  ('out.bias' + str(self.number)): outb_prior}
        
        lifted_module = pyro.random_module(("module" + str(self.number)), self.net, priors)
        
        return lifted_module()

    def train(self, train_data, len_dataset, num_iterations):
        # Treina o algoritmo, usando inferência variacional com o
        # otimizador Adam, ambos implementados no Pyro
        optim = Adam({"lr": 0.01})
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        loss = 0
        losses = []
        for j in range(num_iterations):
            loss = 0
            for data in train_data:
                # calculate the loss and take a gradient step
                loss += svi.step(data[0].float(), data[1])
            total_epoch_loss_train = loss / len_dataset

            losses.append(total_epoch_loss_train)
            #print("Epoca: ", j, " Perda: ", total_epoch_loss_train)
        return losses
            


    def predict(self, x, num_samples):
        # Prediz a saída através de um conjunto de amostragens sucessivas e calculando
        # a média.
        sampled_models = [self.guide(None, None) for _ in range(num_samples)]
        yhats = [model(x).data for model in sampled_models]
        mean = torch.mean(torch.stack(yhats), 0)
        return torch.argmax(mean, dim=1)
