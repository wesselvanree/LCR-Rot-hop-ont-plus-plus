import argparse
import json
import os
from typing import Optional

import optuna
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from ulid import ulid

from lcr.model import LCRRotHopPlusPlus
from lcr.utils import EmbeddingsDataset, train_validation_split
from lcr.utils.paths import Paths
from lcr.utils.torch import get_torch_device


class Objective:
    def __init__(
        self,
        year: int,
        val_ont_hops: Optional[int],
        n_epochs: int,
        device: torch.device,
    ):
        self.year = year
        self.device = device
        self.val_ont_hops = val_ont_hops
        self.n_epochs = n_epochs

    def objective(self, trial: optuna.Trial) -> float:
        tqdm.write("Running objective...")

        learning_rate: float = trial.suggest_categorical(
            "learning_rate", [0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]
        )
        dropout_rate: float = trial.suggest_float(
            "dropout_rate", low=0.25, high=0.75, step=0.1
        )
        momentum: float = trial.suggest_categorical("momentum", [0.85, 0.9, 0.95, 0.99])
        weight_decay: float = trial.suggest_categorical(
            "weight_decay", [0.00001, 0.0001, 0.001, 0.01, 0.1]
        )
        lcr_hops: int = trial.suggest_categorical("lcr_hops", [2, 3, 4, 8])

        # create training and validation DataLoader
        train_dataset = EmbeddingsDataset(
            year=self.year, device=self.device, phase="Train"
        )
        tqdm.write(f"Using {train_dataset} with {len(train_dataset)} obs for training")
        train_idx, validation_idx = train_validation_split(train_dataset)

        training_subset = Subset(train_dataset, train_idx)

        validation_subset: Subset
        if self.val_ont_hops is not None:
            train_val_dataset = EmbeddingsDataset(
                year=self.year,
                device=self.device,
                phase="Train",
                ont_hops=self.val_ont_hops,
            )
            validation_subset = Subset(train_val_dataset, validation_idx)
            tqdm.write(
                f"Using {train_val_dataset} with {len(validation_subset)} obs for validation"
            )
        else:
            validation_subset = Subset(train_dataset, validation_idx)
            tqdm.write(
                f"Using {train_dataset} with {len(validation_subset)} obs for validation"
            )
        training_loader = DataLoader(
            training_subset, batch_size=32, collate_fn=lambda batch: batch
        )
        validation_loader = DataLoader(
            validation_subset, collate_fn=lambda batch: batch
        )

        # Train model
        model = LCRRotHopPlusPlus(hops=lcr_hops, dropout_prob=dropout_rate).to(
            self.device
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        best_accuracy: Optional[float] = None
        best_state_dict: Optional[tuple[dict, dict]] = None
        epochs_progress = tqdm(range(self.n_epochs), unit="epoch")

        for epoch in epochs_progress:
            epoch_progress = tqdm(training_loader, unit="batch", leave=False)
            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):
                torch.set_default_device(self.device)

                batch_outputs = torch.stack(
                    [
                        model(left, target, right, hops)
                        for (left, target, right), _, hops in batch
                    ],
                    dim=0,
                )
                batch_labels = torch.tensor([label.item() for _, label, _ in batch])

                loss: torch.Tensor = criterion(batch_outputs, batch_labels)

                train_loss += loss.item()
                train_steps += 1
                train_n_correct += (
                    (batch_outputs.argmax(1) == batch_labels)
                    .type(torch.int)
                    .sum()
                    .item()
                )
                train_n += len(batch)

                epoch_progress.set_description(
                    f"Train Loss: {train_loss / train_steps:.3f}, Train Acc.: {train_n_correct / train_n:.3f}"
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                torch.set_default_device("cpu")

            # Validation loss
            epoch_progress = tqdm(validation_loader, unit="obs", leave=False)
            model.eval()

            val_loss = 0.0
            val_steps = 0
            val_n = 0
            val_n_correct = 0
            for i, data in enumerate(epoch_progress):
                torch.set_default_device(self.device)

                with torch.no_grad():
                    (left, target, right), label, hops = data[0]

                    output: torch.Tensor = model(left, target, right, hops)
                    val_n_correct += (output.argmax(0) == label).type(torch.int).item()
                    val_n += 1

                    loss = criterion(output, label)
                    val_loss += loss.item()
                    val_steps += 1

                    epoch_progress.set_description(
                        f"Val Loss: {val_loss / val_steps:.3f}, Val Acc.: {val_n_correct / val_n:.3f}"
                    )

                torch.set_default_device("cpu")

            validation_accuracy = val_n_correct / val_n
            tqdm.write(f"Validation accuracy: {validation_accuracy:.3f}")

            if best_accuracy is None or validation_accuracy > best_accuracy:
                epochs_progress.set_description(
                    f"Best Test Acc.: {validation_accuracy:.3f}"
                )
                best_accuracy = validation_accuracy
                best_state_dict = (model.state_dict(), optimizer.state_dict())

        if best_accuracy is None:
            best_accuracy = 0.0

        return best_accuracy


def main(year: int, n_epochs: int, n_trials: int, val_ont_hops: Optional[int]):
    device = get_torch_device()

    study = optuna.create_study(
        study_name=f"params{year}_t{n_trials}_e{n_epochs}_{ulid()}",  # NOTE: ULIDs are sortable
        direction="maximize",  # We maximize the accuracy
    )
    study.optimize(
        Objective(
            year=year, val_ont_hops=val_ont_hops, n_epochs=n_epochs, device=device
        ).objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    hyperparams_dir = Paths.repo_root / "results" / study.study_name
    os.makedirs(hyperparams_dir, exist_ok=True)

    with open(hyperparams_dir / "best_value.txt", "w") as f:
        f.write(str(study.best_value))
    with open(hyperparams_dir / "best_params.json", "w") as f:
        json.dump(study.best_params, f)

    print(
        f"Saved best trial with objective {study.best_value} and hyperparameters {study.best_params} to {hyperparams_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)"
    )
    parser.add_argument(
        "--val-ont-hops",
        default=None,
        type=int,
        required=False,
        help="The number of hops to use in the validation phase",
    )
    parser.add_argument("--epochs", default=10, type=int, help="The number of epochs")
    parser.add_argument("--trials", default=40, type=int, help="The number of trials")

    args = parser.parse_args()

    main(
        year=args.year,
        val_ont_hops=args.val_ont_hops,
        n_trials=args.trials,
        n_epochs=args.epochs,
    )
