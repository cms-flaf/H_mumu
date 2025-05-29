import os
import pickle as pkl
import tomllib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        training_data,
        validation_data,
        hyperparams,
        batch_size,
        epochs,
        early_stop=False,
        patience=None,
    ):

        # Training data as a tuple of NumPy arrays (x_data, y_data)
        self.training_data = training_data
        # Convert the data to a Torch DataLoader, for optimal training
        self.train_dataloader = self._make_dataloader(training_data, batch_size)
        self.valid_dataloader = self._make_dataloader(validation_data, batch_size)

        self.weight = self._get_pos_weight()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.weight)

        self.hyperparams = hyperparams

        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        if patience:
            self.patience = patience

        # Lists to store average epoch losses
        self.training_loss = []
        self.validation_loss = []

    ### Init helpers ###

    def _make_dataloader(self, data, batch_size):
        """
        Converts the data (tuple of Numpy arrays) into a DataLoader object
        """
        x_data, y_data = self.training_data
        dataset = TensorDataset(torch.Tensor(x_data), torch.Tensor(y_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def _get_pos_weight(self):
        """
        Calculates the loss weight of the signal samples (label = 1)
        vs the background samples (label = 0)
        """
        # https://discuss.pytorch.org/t/use-class-weight-with-binary-cross-entropy-loss/125265/2
        _, y_data = self.training_data
        w = (len(y_data) - sum(y_data)) / sum(y_data)
        return torch.Tensor(w)

    ### Utility functions ###

    def plot_losses(self, valid=True, show=False):
        """
        Creates a scatter plot showing the average epoch loss
        for both training and validation. Saves to file, unless
        show = True
        """
        plt.clf()
        if valid:
            plt.plot(self.validation_loss, color="tab:red", marker="o", label="validation")
            filename = "training_loss_plot_with_valid.png"
        else:
            filename = "training_loss_plot.png"
        plt.plot(self.training_loss, color="tab:green", marker="o", label="training")
        plt.legend()
        plt.xlabel("Training epoch")
        plt.ylabel("Avg. loss")
        if show:
            plt.show()
        else:
            plt.savefig(filename)

    ### Main training functions ###

    def train_single_epoch(self, model):
        """
        The main training loop. Runs a single epoch
        and runs validation
        """
        epoch_loss = 0
        model.train()
        for sample in tqdm(self.train_dataloader):
            # Every data instance is an input + label pair
            inputs, labels = sample
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
        # Add epoch training loss data to validator
        avg_loss = epoch_loss / len(self.train_dataloader)
        self.training_loss.append(avg_loss)
        # Run validation
        valid_loss = self.validate(model)
        self.validation_loss.append(valid_loss)
        return model

    def validate(self, model):
        """
        Runs the model on non-training data to evaluate progress
        """
        print("Validating...")
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for sample in self.valid_dataloader:
                x, y = sample
                guess = model(x)
                loss = self.loss_fn(guess, y)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.valid_dataloader)
        return avg_loss

    ### Training loop definitions ###

    def train_fixed(self, model):
        """
        Runs the training loop for a fixed number of epochs
        """
        for i in range(self.epochs):
            print("On epoch:", i)
            model = self.train_single_epoch(model)
        return model

    def train_early_stop(self, model):
        """
        Runs the training loop until the validation data performs worse
        """
        best_model = None
        bad_trains = 0
        i = 0
        while True:
            print("On epoch", i)
            model = self.train_single_epoch(model)
            i += 1
            if self.validation_loss[-1] > min(self.validation_loss):
                bad_trains += 1
            else:
                best_model = model
                bad_trains = 0
            if bad_trains > self.patience:
                break
        return best_model

    ### Main (call this function) ###

    def train(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), **self.hyperparams)
        if self.early_stop and self.patience is not None:
            model = self.train_early_stop(model)
        else:
            model = self.train_fixed(model)
        print("Training finished!")
        return model
