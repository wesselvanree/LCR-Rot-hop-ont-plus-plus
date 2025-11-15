import glob
from typing import Optional

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class EmbeddingsDataset(Dataset):
    def __init__(self, year: int, phase='Train', ont_hops: Optional[int] = None, device=torch.device('cpu'),
                 empty_ok=False, enable_cache=True, use_vm=True, use_soft_pos=True):
        self.dir = f'data/embeddings/{year}-{phase}'
        if ont_hops is not None:
            self.dir += f"_hops-{ont_hops}"
        if not use_vm:
            self.dir += f"_no-vm"
        if not use_soft_pos:
            self.dir += f"_no-sp"

        self.device = device
        self.length = len(glob.glob(f'{self.dir}/*.pt'))
        self.cache: dict[int, tuple] = {}
        self.enable_cache = enable_cache

        if not empty_ok and self.length == 0:
            raise ValueError(f"Could not find embeddings at {self.dir}")

    def __getitem__(self, item: int):
        if item in self.cache:
            return self.cache[item]

        data: dict = torch.load(f"{self.dir}/{item}.pt", map_location=self.device)
        label: torch.Tensor = torch.tensor(data['label'], requires_grad=False, device=self.device)
        embeddings: torch.Tensor = data['embeddings']
        target_pos: tuple[int, int] = data['target_pos']
        hops: Optional[torch.Tensor] = data['hops']
        target_index_start, target_index_end = target_pos

        left: torch.Tensor = embeddings[0:target_index_start]
        target: torch.Tensor = embeddings[target_index_start:target_index_end]
        right: torch.Tensor = embeddings[target_index_end:]

        result = (
            (left.to(self.device), target.to(self.device), right.to(self.device)),
            label,
            hops
        )

        if self.enable_cache:
            self.cache[item] = result

        return result

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"EmbeddingsDataset({self.dir})"


def train_validation_split(dataset: EmbeddingsDataset, validation_size=0.2, seed: Optional[float] = None):
    # create list of all labels
    loader = DataLoader(dataset, collate_fn=lambda batch: batch)
    labels: list[int] = [data[0][1].item() for data in loader]

    # create stratified train-validation split
    train_idx, validation_idx = train_test_split(
        range(len(dataset)), test_size=validation_size, shuffle=True, stratify=labels, random_state=seed)

    return train_idx, validation_idx
