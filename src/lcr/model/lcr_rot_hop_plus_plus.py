from typing import Optional

import torch
from torch import nn


class BilinearAttention(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.bilinear = nn.Bilinear(
            in1_features=input_size,
            in2_features=input_size,
            out_features=1,
            bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0)

    def forward(self, hidden_states: torch.Tensor, representation: torch.Tensor):
        """
        :param hidden_states: [n x input_size] where n is the number of tokens
        :param representation: [input_size]
        :return: [input_size] the new representation
        """
        n_hidden_states, _ = hidden_states.size()

        # [n_hidden_states x 1]
        att_scores = self.tanh(self.bilinear(hidden_states, representation.repeat(n_hidden_states, 1)))
        att_scores = self.softmax(att_scores)

        return torch.einsum('ij,ik->k', att_scores, hidden_states)


class HierarchicalAttention(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=1, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(0)

    def forward(self, representation1: torch.Tensor, representation2: torch.Tensor):
        """
        :param representation1: [input_size]
        :param representation2: [input_size]
        :return: representation1, representation2: the representations scaled by their corresponding attention score
        """
        representations = torch.cat((
            self.tanh(self.linear(representation1)),
            self.tanh(self.linear(representation2))
        ))

        attention_scores = self.softmax(representations)
        representation1 = attention_scores[0] * representation1
        representation2 = attention_scores[1] * representation2

        return representation1, representation2


class LCRRotHopPlusPlus(nn.Module):
    def __init__(self, dropout_prob=0.7, output_size=3, input_size=768, hidden_size=300, hops=3,
                 gamma: Optional[int] = None):
        super().__init__()
        self.hops = hops
        self.gamma = gamma

        if hops < 1:
            raise ValueError("Invalid number of hops")

        self.dropout = nn.Dropout(p=dropout_prob)

        self.lstm_left = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.lstm_target = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.lstm_right = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True)
        self.representation_size = 2 * hidden_size

        self.bilinear_left = BilinearAttention(self.representation_size)
        self.bilinear_target_left = BilinearAttention(self.representation_size)
        self.bilinear_target_right = BilinearAttention(self.representation_size)
        self.bilinear_right = BilinearAttention(self.representation_size)

        self.hierarchical_context = HierarchicalAttention(self.representation_size)
        self.hierarchical_target = HierarchicalAttention(self.representation_size)

        self.output_linear = nn.Linear(in_features=4 * self.representation_size, out_features=output_size)
        self.softmax = nn.Softmax(0)

    def forward(self, left: torch.Tensor, target: torch.Tensor, right: torch.Tensor,
                hops: Optional[torch.Tensor] = None):
        """
        :param left: [n_left x input_size] left-context embeddings, where n_left is the number of tokens in the left
                     context
        :param target: [n_target x input_size] target embeddings, where n_target is the number of tokens in the target
        :param right: [n_right x input_size] right-context embeddings, where n_right is the number of tokens in the
                      right context
        :param hops: [n_left + n_target + n_right] vector indicating the number of hops for each token, where a negative
                     number indicates that a word is in the original sentence
        :return: [1 x output_size] output probabilities for each class
        """
        n_left, _ = left.size()
        n_target, _ = target.size()
        n_right, _ = right.size()

        # determine weights and scale embeddings
        if self.gamma is not None and hops is not None:
            weights: torch.Tensor = hops
            for i, n_hops in enumerate(weights):
                if n_hops < 0:
                    weights[i] = 1
                else:
                    weights[i] = 1 / (self.gamma + n_hops)
            weights_left: torch.Tensor = weights[:n_left]
            weights_target: torch.Tensor = weights[n_left:(n_left + n_target)]
            weights_right: torch.Tensor = weights[(n_left + n_target):]

            left = torch.einsum('i,ij->ij', weights_left, left)
            target = torch.einsum('i,ij->ij', weights_target, target)
            right = torch.einsum('i,ij->ij', weights_right, right)

        # calculate hidden states
        left_hidden_states: Optional[torch.Tensor]
        if n_left != 0:
            left_hidden_states, _ = self.lstm_left(self.dropout(left))
        else:
            left_hidden_states = None

        target_hidden_states, _ = self.lstm_target(self.dropout(target))

        right_hidden_states: Optional[torch.Tensor]
        if n_right != 0:
            right_hidden_states, _ = self.lstm_right(self.dropout(right))
        else:
            right_hidden_states = None

        # initial representations using pooling
        representation_target_left = torch.mean(target_hidden_states, dim=0)
        representation_target_right = representation_target_left

        representation_left: torch.Tensor | None = None
        representation_right: torch.Tensor | None = None

        # rotatory attention
        for i in range(self.hops):
            # target-to-context
            if left_hidden_states is not None:
                representation_left = self.bilinear_left(left_hidden_states, representation_target_left)
            if right_hidden_states is not None:
                representation_right = self.bilinear_right(right_hidden_states, representation_target_right)
            if left_hidden_states is not None and right_hidden_states is not None:
                representation_left, representation_right = self.hierarchical_context(
                    representation_left,
                    representation_right
                )

            # context-to-target
            if representation_left is not None and representation_right is not None:
                representation_target_left, representation_target_right = self.hierarchical_target(
                    self.bilinear_target_left(target_hidden_states, representation_left),
                    self.bilinear_target_right(target_hidden_states, representation_right)
                )
            elif representation_left is not None:
                representation_target_left = self.bilinear_target_left(target_hidden_states, representation_left)
            elif representation_right is not None:
                representation_target_right = self.bilinear_target_right(target_hidden_states, representation_right)

        if representation_left is None:
            representation_left = torch.zeros(self.representation_size)
        if representation_right is None:
            representation_right = torch.zeros(self.representation_size)

        # determine output probabilities
        output = torch.concat([
            representation_left,
            representation_target_left,
            representation_target_right,
            representation_right,
        ])
        output = self.dropout(output)
        output = self.output_linear(output)

        # CrossEntropyLoss requires raw logits
        if not self.training:
            output = self.softmax(output)

        return output
