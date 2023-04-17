# https://github.com/autoliuweijie/K-BERT
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, layer, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = layer.weight
        self.beta = layer.bias

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
