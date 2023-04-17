# https://github.com/autoliuweijie/K-BERT
import torch.nn as nn
import torch
from .transformer import TransformerLayer


class BertEncoderArgs:
    def __init__(self, param={}):
        self.emb_size = param.get("emb_size", 768)
        self.hidden_size = param.get("hidden_size", 768)
        self.kernel_size = param.get("kernel_size", 3)
        self.block_size = param.get("block_size", 2)
        self.feedforward_size = param.get("feedforward_size", 3072)
        self.heads_num = param.get("heads_num", 12)
        self.layers_num = param.get("layers_num", 12)
        self.dropout = param.get("dropout", 0.1)


class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, model, args=BertEncoderArgs()):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([
            TransformerLayer(args, model.base_model.encoder.layer._modules.get(key)) for key in
            model.base_model.encoder.layer._modules
        ])

    def forward(self, emb, vm: torch.Tensor = None):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            vm: [seq_length x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """

        # Generate mask according to segment indicators.
        # mask: [batch_size x 1 x seq_length x seq_length]
        hidden_layers = []
        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, vm)
            hidden_layers.append(hidden)

        hidden = (hidden_layers[11] + hidden_layers[10] + hidden_layers[9] + hidden_layers[8]) / 4
        n_digits = 8
        hidden = torch.round(hidden * 10 ** n_digits) / (10 ** n_digits)
        return hidden
