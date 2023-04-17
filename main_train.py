import argparse
import os
from typing import Optional

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split


def stringify_float(value: float):
    return str(value).replace('.', '-')


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--hops", default=3, type=int,
                        help="The number of hops to use in the rotatory attention mechanism")
    parser.add_argument("--ont-hops", default=None, type=int, required=False,
                        help="The number of hops in the ontology to use")
    parser.add_argument("--val-ont-hops", default=None, type=int, required=False,
                        help="The number of hops to use in the validation phase, this option overrides the --ont-hops option.")
    args = parser.parse_args()

    year: int = args.year

    lcr_hops: int = args.hops
    ont_hops: Optional[int] = args.ont_hops
    val_ont_hops: Optional[int] = args.val_ont_hops
    dropout_rate = 0.5

    learning_rate = 0.005
    momentum = 0.99
    weight_decay = 0.00001
    n_epochs = 100
    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    # create training anf validation DataLoader
    train_dataset = EmbeddingsDataset(year=year, device=device, phase="Train", ont_hops=ont_hops)
    print(f"Using {train_dataset} with {len(train_dataset)} obs for training")
    train_idx, validation_idx = train_validation_split(train_dataset)

    training_subset = Subset(train_dataset, train_idx)

    if val_ont_hops is not None:
        train_val_dataset = EmbeddingsDataset(year=year, device=device, phase="Train", ont_hops=val_ont_hops)
        validation_subset = Subset(train_val_dataset, validation_idx)
        print(f"Using {train_val_dataset} with {len(validation_subset)} obs for validation")
    else:
        validation_subset = Subset(train_dataset, validation_idx)
        print(f"Using {train_dataset} with {len(validation_subset)} obs for validation")

    training_loader = DataLoader(training_subset, batch_size=batch_size, collate_fn=lambda batch: batch)
    validation_loader = DataLoader(validation_subset, collate_fn=lambda batch: batch)

    # Train model
    model = LCRRotHopPlusPlus(hops=lcr_hops, dropout_prob=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    best_accuracy: Optional[float] = None
    best_state_dict: Optional[dict] = None
    epochs_progress = tqdm(range(n_epochs), unit='epoch')

    try:
        for epoch in epochs_progress:
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):
                torch.set_default_device(device)

                batch_outputs = torch.stack(
                    [model(left, target, right, hops) for (left, target, right), _, hops in batch], dim=0)
                batch_labels = torch.tensor([label.item() for _, label, _ in batch])

                loss: torch.Tensor = criterion(batch_outputs, batch_labels)

                train_loss += loss.item()
                train_steps += 1
                train_n_correct += (batch_outputs.argmax(1) == batch_labels).type(torch.int).sum().item()
                train_n += len(batch)

                epoch_progress.set_description(
                    f"Train Loss: {train_loss / train_steps:.3f}, Train Acc.: {train_n_correct / train_n:.3f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                torch.set_default_device('cpu')

            # Validation loss
            epoch_progress = tqdm(validation_loader, unit='obs', leave=False)
            model.eval()

            val_loss = 0.0
            val_steps = 0
            val_n = 0
            val_n_correct = 0
            for i, data in enumerate(epoch_progress):
                torch.set_default_device(device)

                with torch.no_grad():
                    (left, target, right), label, hops = data[0]

                    output: torch.Tensor = model(left, target, right, hops)
                    val_n_correct += (output.argmax(0) == label).type(torch.int).item()
                    val_n += 1

                    loss = criterion(output, label)
                    val_loss += loss.item()
                    val_steps += 1

                    epoch_progress.set_description(
                        f"Test Loss: {val_loss / val_steps:.3f}, Test Acc.: {val_n_correct / val_n:.3f}")

                torch.set_default_device('cpu')

            validation_accuracy = val_n_correct / val_n

            if best_accuracy is None or validation_accuracy > best_accuracy:
                epochs_progress.set_description(f"Best Test Acc.: {validation_accuracy:.3f}")
                best_accuracy = validation_accuracy
                best_state_dict = model.state_dict()
    except KeyboardInterrupt:
        print("Interrupted training procedure, saving best model...")

    if best_state_dict is not None:
        models_dir = os.path.join("data", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir,
                                  f"{year}_LCR_hops{lcr_hops}_dropout{stringify_float(dropout_rate)}_acc{stringify_float(best_accuracy)}.pt")
        with open(model_path, "wb") as f:
            torch.save(best_state_dict, f)
            print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
