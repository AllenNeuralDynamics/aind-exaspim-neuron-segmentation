"""
Created on Fri June 27 4:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code used to train neural network to perform image segmentation.

"""

from contextlib import nullcontext
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
from torch import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from aind_exaspim_neuron_segmentation.machine_learning.unet3d import UNet3D
from aind_exaspim_neuron_segmentation.utils import util


class Trainer:
    """
    Trainer class for managing the full training pipeline of a segmentation
    model.
    """

    def __init__(
        self,
        output_dir,
        affinity_mode=True,
        batch_size=16,
        lr=1e-3,
        max_epochs=1000,
        use_amp=True,
    ):
        """
        Instantiates a Trainer object.

        Parameters
        ----------
        output_dir : str
            Directory where logs, model checkpoints, and TensorBoard is saved.
        affinity_mode : bool, optional
            Indication of whether the task is instance segmentation. In this
            case, the model learns affinity channels. Default is True.
        batch_size : int, optional
            Number of samples per batch for both training and validation.
            Default is 16.
        lr : float, optional
            Initial learning rate for the optimizer. Default is 1e-3.
        max_epochs : int, optional
            Maximum number of training epochs. Default is 1000.
        use_amp : bool, optional
            Indication of whether to use mixed precision. Default is True.
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

        output_channels = 3 if affinity_mode else 1
        self.criterion = nn.BCEWithLogitsLoss()
        self.model = UNet3D(output_channels=output_channels).to("cuda")
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=25)
        self.writer = SummaryWriter(log_dir=log_dir)

        if use_amp:
            self.autocast = autocast(device_type="cuda", dtype=torch.float16)
        else:
            self.autocast = nullcontext()


    # --- Core Routines ---
    def run(self, train_dataset, val_dataset):
        """
        Runs the full training and validation for the given maximum number of
        epochs.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Dataset used for training.
        val_dataset : torch.utils.data.Dataset
            Dataset used for validation.
        """
        # Initializations
        print("\nExperiment:", os.path.basename(os.path.normpath(self.log_dir)))
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
            print(f"\nEpoch {epoch}: " + ("New Best!" if new_best else " "))
            self.report_stats(train_stats, is_train=True)
            self.report_stats(val_stats, is_train=False)

            # Step scheduler
            self.scheduler.step()

    def train_step(self, train_dataloader, epoch):
        """
        Performs a single training epoch over the provided DataLoader.

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        stats : Dict[str, List[float]]
            Dictionary of aggregated training metrics.
        """
        stats = {"f1": None, "precision": [], "recall": [], "loss": []}
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
        Performs a full validation loop over the given dataloader.

        Parameters
        ----------
        val_dataloader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        epoch : int
            Current training epoch.

        Returns
        -------
        stats : Dict[str, List[float]]
            Dictionary of aggregated validation metrics.
        is_best : bool
            True if the current F1 score is the best so far.
        """
        stats = {"f1": None, "precision": [], "recall": [], "loss": []}
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
        Performs a forward pass through the model and computes loss.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, C, D, H, W).
        y : torch.Tensor
            Ground truth labels with shape (B, C, D, H, W).

        Returns
        -------
        hat_y : torch.Tensor
            Model predictions.
        loss : torch.Tensor
            Computed loss value.
        """
        with self.autocast:
            x = x.to("cuda", dtype=torch.float)
            y = y.to("cuda", dtype=torch.float)
            hat_y = self.model(x)
            loss = self.criterion(hat_y, y)
        return hat_y, loss

    # --- Helpers
    def compute_stats(self, y, hat_y):
        """
        Computes F1 score, precision, and recall for each example in a batch.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth labels of shape (B, 1, D, H, W) or (B, 1, H, W).
        hat_y : torch.Tensor
            Model predictions of the same shape as ground truth.

        Returns
        -------
        stats : Dict[str, List[float]]
            Dictionary containing lists of per-example metrics.
        """
        y, hat_y = toCPU(y, True), toCPU(hat_y, True)
        stats = {"precision": list(), "recall": list()}
        for i in range(y.shape[0]):
            # Ensure binary format
            gt = (y[i, 0, ...] > 0).astype(np.uint8).flatten()
            pred = (hat_y[i, 0, ...] > 0).astype(np.uint8).flatten()

            # Compute metrics
            stats["precision"].append(precision_score(gt, pred, zero_division=np.nan))
            stats["recall"].append(recall_score(gt, pred, zero_division=np.nan))
        return stats

    def report_stats(self, stats, is_train=True):
        """
        Prints a summary of training or validation statistics.

        Parameters
        ----------
        stats : dict
            Dictionary containing lists of per-example metrics.
        is_train : bool, optional
            Indication of whether "stats" were computed during training step.
            Default is True.
        """
        summary = "   Train: " if is_train else "   Val: "
        for key, value in stats.items():
            summary += f"{key}={value:.4f}, "
        print(summary)

    def save_model(self, epoch):
        """
        Saves the current model state to a file.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        """
        date = datetime.today().strftime("%Y%m%d")
        filename = f"UNet3d-{date}-{epoch}-{self.best_f1:.4f}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), path)

    def update_tensorboard(self, stats, epoch, prefix):
        """
        Logs scalar statistics to TensorBoard.

        Parameters
        ----------
        stats : dict
            Dictionary of metric names to lists of values.
        epoch : int
            Current training epoch.
        prefix : str
            Prefix to prepend to each metric name when logging.
        """
        # Compute avg f1 score
        avg_prec = np.nanmean(stats["precision"])
        avg_recall = np.nanmean(stats["recall"])
        stats["f1"] = [2 * avg_prec * avg_recall / (avg_prec + avg_recall)]

        # Write to tensorboard
        for key, value in stats.items():
            stats[key] = np.nanmean(value)
            self.writer.add_scalar(prefix + key, stats[key], epoch)


# --- Helpers ---
def toCPU(tensor, to_numpy=False):
    """
    Moves PyTorch tensor to the CPU and optionally converts it to a NumPy
    array.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to move to CPU.
    to_numpy : bool, optional
        If True, converts the tensor to a NumPy array. Default is False.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        Tensor or array on CPU.
    """
    if to_numpy:
        return np.array(tensor.detach().cpu())
    else:
        return tensor.detach().cpu()
