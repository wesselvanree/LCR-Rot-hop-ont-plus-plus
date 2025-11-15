# https://github.com/autoliuweijie/K-BERT
import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertLayer


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout, layer: BertLayer):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([
            layer.attention.self.query, layer.attention.self.key, layer.attention.self.value
            # Uses the self attention linear layer from pre trained bert with weights and biases
        ])

        self.final_linear = layer.attention.output.dense

    def forward(self, key, value, query, vm: Optional[torch.Tensor] = None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            vm: [batch_size x 1 x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x). \
                                 view(batch_size, -1, heads_num, per_head_size). \
                                 transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                             ]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))

        if vm is not None:
            scores = scores + vm

        probs = nn.Softmax(dim=-1)(scores)
        output = unshape(torch.matmul(probs.float(), value.float()))
        output = self.final_linear(output)

        return output
