"""
Created on Fri June 27 4:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code used to train neural network to perform image segmentation.

"""

from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from aind_exaspim_neuron_segmentation.machine_learning.unet3d import UNet
from aind_exaspim_neuron_segmentation.utils import util


class Trainer:
    """
    Trainer class for managing the full training pipeline of a
    segmentation model.
    """

    def __init__(
        self,
        output_dir,
        batch_size=16,
        lr=1e-3,
        max_epochs=1000,
    ):
        """
        Instantiates Trainer object.

        Parameters
        ----------
        output_dir : str
            Directory where logs, model checkpoints, and TensorBoard is saved.
        batch_size : int, optional
            Number of samples per batch for both training and validation.
            Default is 16.
        lr : float, optional
            Initial learning rate for the optimizer. Default is 1e-3.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 1000.

        Returns
        -------
        None
        """
        # Initializations
        exp_name = "session-" + datetime.today().strftime("%Y%m%d_%H%M")
        log_dir = os.path.join(output_dir, exp_name)
        util.mkdir(log_dir)

        # Instance attributes
        self.batch_size = batch_size
        self.best_f1 = 0
        self.max_epochs = max_epochs
        self.log_dir = log_dir

        self.criterion = nn.BCEWithLogitsLoss()
        self.model = UNet().to("cuda")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)
        self.writer = SummaryWriter(log_dir=log_dir)

    # --- Core Routines ---
    def run(self, train_dataset, val_dataset):
        """
        Run the full training and validation loop.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Dataset used for training.
        val_dataset : torch.utils.data.Dataset
            Dataset used for validation.

        Returns
        -------
        None
        """
        # Initializations
        print("Experiment:", os.path.basename(os.path.normpath(self.log_dir)))
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Main
        for epoch in range(self.max_epochs):
            # Train-Validate
            train_stats = self.train_step(train_dataloader, epoch)
            val_stats, new_best = self.validate_step(val_dataloader, epoch)

            # Report reuslts
            print(f"\nEpoch {epoch}: " + "New Best!" if new_best else " ")
            self.report_stats(train_stats, is_train=True)
            self.report_stats(val_stats, is_train=False)

            # Step scheduler
            self.scheduler.step()

    def train_step(self, train_dataloader, epoch):
        """
        Perform a single training epoch over the provided DataLoader.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        dict
            Dictionary of aggregated training metrics.
        """
        stats = {"f1": [], "precision": [], "recall": [], "loss": []}
        self.model.train()
        for x, y in train_dataloader:
            # Forward pass
            hat_y, loss = self.forward_pass(x, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store stats for tensorboard
            stats["loss"].append(float(toCPU(loss)))
            for key, value in self.compute_stats(y, hat_y).items():
                stats[key].extend(value)

        # Write stats to tensorboard
        self.update_tensorboard(stats, epoch, "train_")
        return stats

    def validate_step(self, val_dataloader, epoch):
        """
        Perform a full validation loop over the given dataloader.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        tuple
            stats : dict
                Dictionary of aggregated validation metrics.
            is_best : bool
                True if the current F1 score is the best so far.
        """
        stats = {"f1": [], "precision": [], "recall": [], "loss": []}
        with torch.no_grad():
            self.model.eval()
            for x, y in val_dataloader:
                # Run model
                hat_y, loss = self.forward_pass(x, y)

                # Store stats for tensorboard
                stats["loss"].append(float(toCPU(loss)))
                for key, value in self.compute_stats(y, hat_y).items():
                    stats[key].extend(value)

        # Write stats to tensorboard
        self.update_tensorboard(stats, epoch, "val_")

        # Check for new best
        if stats["f1"] > self.best_f1:
            self.save_model(epoch)
            self.best_f1 = stats["f1"]
            return stats, True
        else:
            return stats, False

    def forward_pass(self, x, y):
        """
        Perform a forward pass through the model and compute loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, D, H, W).
        y : torch.Tensor
            Ground truth labels with shape (B, C, D, H, W).

        Returns
        -------
        tuple
            hat_y : torch.Tensor
                Model predictions.
            loss : torch.Tensor
                Computed loss value.
        """
        x = x.to("cuda", dtype=torch.float)
        y = y.to("cuda", dtype=torch.float)
        hat_y = self.model(x)
        loss = self.criterion(hat_y, y)
        return hat_y, loss

    # --- Helpers
    def compute_stats(self, y, hat_y):
        """
        Compute F1 score, precision, and recall for each sample in a batch.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth labels of shape (B, 1, D, H, W) or (B, 1, H, W).
        hat_y : torch.Tensor
            Model predictions of the same shape as ground truth.

        Returns
        -------
        dict
            Dictionary containing lists of per-sample metrics.
        """
        y, hat_y = toCPU(y, True), toCPU(hat_y, True)
        stats = {"f1": list(), "precision": list(), "recall": list()}
        for i in range(y.shape[0]):
            # Ensure binary format
            gt = (y[i, 0, ...] > 0).astype(np.uint8).flatten()
            pred = (hat_y[i, 0, ...] > 0).astype(np.uint8).flatten()

            # Compute metrics
            stats["f1"].append(f1_score(gt, pred, zero_division=np.nan))
            stats["precision"].append(precision_score(gt, pred, zero_division=np.nan))
            stats["recall"].append(recall_score(gt, pred, zero_division=np.nan))
        return stats

    def report_stats(self, stats, is_train=True):
        """
        Print a summary of training or validation statistics.

        Parameters
        ----------
        stats : dict
            Dictionary of metric names to values.
        is_train : bool, optional
            Indication of whether "stats" were computed during training step.
            Default is True.

        Returns
        -------
        None
        """
        summary = "   Train: " if is_train else "   Val: "
        for key, value in stats.items():
            summary += f"{key}={value:.4f}, "
        print(summary)

    def save_model(self, epoch):
        """
        Save the current model state to a file.

        Parameters
        ----------
        epoch : int
            Current training epoch.

        Returns
        -------
        None
        """
        date = datetime.today().strftime("%Y%m%d")
        filename = f"UNet3d-{date}-{epoch}-{self.best_f1:.4f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), path)

    def update_tensorboard(self, stats, epoch, prefix):
        """
        Log scalar statistics to TensorBoard.

        Parameters
        ----------
        stats : dict
            Dictionary of metric names (str) to lists of values.
        epoch : int
            Current training epoch.
        prefix : str
            Prefix to prepend to each metric name when logging.

        Returns
        -------
        None
        """
        for key, value in stats.items():
            stats[key] = np.nanmean(value)
            self.writer.add_scalar(prefix + key, stats[key], epoch)


# --- Helpers ---
def toCPU(tensor, to_numpy=False):
    """
    Move PyTorch tensor to the CPU and optionally convert it to a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to move to CPU.
    to_numpy : bool, optional
        If True, converts the tensor to a NumPy array. Default is False.

    Returns
    -------
    torch.Tensor or np.ndarray
        Tensor or array on CPU.
    """
    if to_numpy:
        return np.array(tensor.detach().cpu())
    else:
        return tensor.detach().cpu()
