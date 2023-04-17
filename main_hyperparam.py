import argparse
import json
import os
import pickle
from typing import Optional

import torch
from hyperopt import hp, tpe, fmin, Trials, STATUS_OK
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import LCRRotHopPlusPlus
from utils import EmbeddingsDataset, train_validation_split


class HyperOptManager:
    """A class that performs hyperparameter optimization and stores the best states as checkpoints."""

    def __init__(self, year: int, val_ont_hops: Optional[int]):
        self.year = year
        self.n_epochs = 20
        self.val_ont_hops = val_ont_hops

        self.eval_num = 0
        self.best_loss = None
        self.best_hyperparams = None
        self.best_state_dict = None
        self.trials = Trials()

        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else 'cpu')

        # read checkpoint if exists
        self.__checkpoint_dir = f"data/checkpoints/{year}_epochs{self.n_epochs}"

        if os.path.isdir(self.__checkpoint_dir):
            try:
                self.best_state_dict = torch.load(f"{self.__checkpoint_dir}/state_dict.pt")
                with open(f"{self.__checkpoint_dir}/hyperparams.json", "r") as f:
                    self.best_hyperparams = json.load(f)
                with open(f"{self.__checkpoint_dir}/trials.pkl", "rb") as f:
                    self.trials = pickle.load(f)
                    self.eval_num = len(self.trials)
                with open(f"{self.__checkpoint_dir}/loss.txt", "r") as f:
                    self.best_loss = float(f.read())

                print(f"Resuming from previous checkpoint {self.__checkpoint_dir} with best loss {self.best_loss}")
            except IOError:
                raise ValueError(f"Checkpoint {self.__checkpoint_dir} is incomplete, please remove this directory")
        else:
            print("Starting from scratch")

    def run(self):
        # TODO: convert to dict for better readability in json file?
        space = [
            hp.choice('learning_rate', [0.02, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.1]),
            hp.quniform('dropout_rate', 0.25, 0.75, 0.1),
            hp.choice('momentum', [0.85, 0.9, 0.95, 0.99]),
            hp.choice('weight_decay', [0.00001, 0.0001, 0.001, 0.01, 0.1]),
            hp.choice('lcr_hops', [2, 3, 4, 8])
        ]

        best = fmin(self.objective, space=space, algo=tpe.suggest, trials=self.trials, show_progressbar=False)

    def objective(self, hyperparams):
        self.eval_num += 1
        learning_rate, dropout_rate, momentum, weight_decay, lcr_hops = hyperparams
        print(f"\n\nEval {self.eval_num} with hyperparams {hyperparams}")

        # create training and validation DataLoader
        train_dataset = EmbeddingsDataset(year=self.year, device=self.device, phase="Train")
        print(f"Using {train_dataset} with {len(train_dataset)} obs for training")
        train_idx, validation_idx = train_validation_split(train_dataset)

        training_subset = Subset(train_dataset, train_idx)

        validation_subset: Subset
        if self.val_ont_hops is not None:
            train_val_dataset = EmbeddingsDataset(year=self.year, device=self.device, phase="Train",
                                                  ont_hops=self.val_ont_hops)
            validation_subset = Subset(train_val_dataset, validation_idx)
            print(f"Using {train_val_dataset} with {len(validation_subset)} obs for validation")
        else:
            validation_subset = Subset(train_dataset, validation_idx)
            print(f"Using {train_dataset} with {len(validation_subset)} obs for validation")
        training_loader = DataLoader(training_subset, batch_size=32, collate_fn=lambda batch: batch)
        validation_loader = DataLoader(validation_subset, collate_fn=lambda batch: batch)

        # Train model
        model = LCRRotHopPlusPlus(hops=lcr_hops, dropout_prob=dropout_rate).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        best_accuracy: Optional[float] = None
        best_state_dict: Optional[tuple[dict, dict]] = None
        epochs_progress = tqdm(range(self.n_epochs), unit='epoch')

        for epoch in epochs_progress:
            epoch_progress = tqdm(training_loader, unit='batch', leave=False)
            model.train()

            train_loss = 0.0
            train_n_correct = 0
            train_steps = 0
            train_n = 0

            for i, batch in enumerate(epoch_progress):
                torch.set_default_device(self.device)

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
                        f"Test Loss: {val_loss / val_steps:.3f}, Test Acc.: {val_n_correct / val_n:.3f}")

                torch.set_default_device('cpu')

            validation_accuracy = val_n_correct / val_n

            if best_accuracy is None or validation_accuracy > best_accuracy:
                epochs_progress.set_description(f"Best Test Acc.: {validation_accuracy:.3f}")
                best_accuracy = validation_accuracy
                best_state_dict = (model.state_dict(), optimizer.state_dict())

        # we want to maximize accuracy, which is equivalent to minimizing -accuracy
        objective_loss = -best_accuracy
        self.check_best_loss(objective_loss, hyperparams, best_state_dict)

        return {
            'loss': loss,
            'status': STATUS_OK,
            'space': hyperparams,
        }

    def check_best_loss(self, loss: float, hyperparams, state_dict: tuple[dict, dict]):
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_hyperparams = hyperparams
            self.best_state_dict = state_dict

            os.makedirs(self.__checkpoint_dir, exist_ok=True)

            torch.save(state_dict, f"{self.__checkpoint_dir}/state_dict.pt")
            with open(f"{self.__checkpoint_dir}/hyperparams.json", "w") as f:
                json.dump(hyperparams, f)
            with open(f"{self.__checkpoint_dir}/loss.txt", "w") as f:
                f.write(str(self.best_loss))
            print(
                f"Best checkpoint with loss {self.best_loss} and hyperparameters {self.best_hyperparams} saved to {self.__checkpoint_dir}")

        with open(f"{self.__checkpoint_dir}/trials.pkl", "wb") as f:
            pickle.dump(self.trials, f)


def main():
    # parse CLI args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--year", default=2015, type=int, help="The year of the dataset (2015 or 2016)")
    parser.add_argument("--val-ont-hops", default=None, type=int, required=False,
                        help="The number of hops to use in the validation phase")
    args = parser.parse_args()
    val_ont_hops: Optional[int] = args.val_ont_hops
    year: int = args.year

    opt = HyperOptManager(year=year, val_ont_hops=val_ont_hops)
    opt.run()


if __name__ == "__main__":
    main()
