from typing import Optional

import torch
from rdflib import Graph
from transformers import BertTokenizer, BertModel

from .bert_encoder import BertEncoder
from .sentence_tree import SentenceTree

tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model: BertModel = BertModel.from_pretrained("bert-base-uncased")
# print empty line after bert-base-uncased warnings
print()


class EmbeddingsLayer:
    def __init__(self, hops: Optional[int] = None, ontology: Optional[Graph] = None, use_vm=True, use_soft_pos=True,
                 device=torch.device('cpu')):
        super().__init__()

        self.ont_hops = hops
        self.ontology = ontology
        self.use_vm = use_vm
        self.use_soft_pos = use_soft_pos

        self.device = device
        self.tokenizer: BertTokenizer = tokenizer
        self.model: BertModel = model.to(device)
        self.model.eval()
        self.encoder = BertEncoder(self.model)

    def forward(self, sentence: str, target_start: int, target_end: int) -> tuple[
        torch.Tensor, tuple[int, int], Optional[torch.Tensor]
    ]:
        sentence = f"[CLS] {sentence} [SEP]"
        target_start += 6
        target_end += 6

        # insert knowledge
        if self.ont_hops is not None and self.ont_hops >= 0 and self.ontology is not None:
            tree = SentenceTree(sentence, target_start, target_end, self.ontology, self.tokenizer, self.device,
                                self.ont_hops)

            # generate embeddings for the BERT model
            tree_embeddings = tree.build_embedding()
            target_index_start = tree_embeddings.target_start - 1
            target_index_end = tree_embeddings.target_end - 1

            # generate embeddings using pre-trained BERT model
            initial_embeddings = self.model.embeddings.forward(
                input_ids=tree_embeddings.input_ids,
                token_type_ids=tree_embeddings.token_type_ids,
                position_ids=tree_embeddings.position_ids if self.use_soft_pos else None
            )
            embeddings: torch.Tensor = self.encoder(initial_embeddings, vm=tree_embeddings.vm if self.use_vm else None)
            embeddings = embeddings[0][1:-1]
            hops = tree_embeddings.hops[1:-1]

            return embeddings, (target_index_start, target_index_end), hops

        # do not insert knowledge
        left_str = self.tokenizer.tokenize(sentence[0:target_start])
        target_str = self.tokenizer.tokenize(sentence[target_start:target_end])
        target_index_start = len(left_str) - 1
        target_index_end = target_index_start + len(target_str)

        tokens = self.tokenizer.tokenize(sentence)
        ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)], device=self.device)
        token_type_ids = torch.tensor([[0] * len(tokens)], device=self.device)

        initial_embeddings = self.model.embeddings.forward(input_ids=ids, token_type_ids=token_type_ids)
        embeddings: torch.Tensor = self.encoder(initial_embeddings)
        embeddings = embeddings[0][1:-1]

        return embeddings, (target_index_start, target_index_end), None
