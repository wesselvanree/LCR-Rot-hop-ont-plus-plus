# https://github.com/autoliuweijie/K-BERT
import torch.nn as nn
from transformers import BertLayer


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """

    def __init__(self, hidden_size, feedforward_size, layer: BertLayer):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = layer.intermediate.dense
        self.linear_2 = layer.output.dense

    def forward(self, x):
        inter = nn.functional.gelu(self.linear_1(x))
        output = self.linear_2(inter)
        return output
