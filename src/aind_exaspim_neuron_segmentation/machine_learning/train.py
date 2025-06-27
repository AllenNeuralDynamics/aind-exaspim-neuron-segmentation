"""
Created on Fri Jan 3 12:30:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code used to train neural network to classify somas proposals.

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
    def __init__(
        self,
        output_dir,
        batch_size=16,
        lr=1e-3,
        max_epochs=1000,
    ):
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
            print(f"Epoch {epoch}: " + "New Best!" if new_best else "")
            print("  Train Results...")
            self.report_stats(train_stats)

            print("  Validate Results...")
            self.report_stats(val_stats)

            # Step scheduler
            self.scheduler.step()

    def train_step(self, train_dataloader, epoch):
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
        x = x.to("cuda", dtype=torch.float)
        y = y.to("cuda", dtype=torch.float)
        hat_y = self.model(x)
        loss = self.criterion(hat_y, y)
        return hat_y, loss

    # --- Helpers
    def compute_stats(self, y, hat_y):
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

    def report_stats(self, stats):
        summary = "    "
        for key, value in stats.items():
            summary += f"{key}={value:.4f}, "
        print(summary)

    def save_model(self, epoch):
        date = datetime.today().strftime("%Y%m%d")
        filename = f"UNet3d-{date}-{epoch}-{round(self.best_f1, 4)}.pth"
        path = os.path.join(self.log_dir, filename)
        torch.save(self.model.state_dict(), path)

    def update_tensorboard(self, stats, epoch, prefix):
        for key, value in stats.items():
            stats[key] = np.nanmean(value)
            self.writer.add_scalar(prefix + key, stats[key], epoch)


# --- Helpers ---
def toCPU(tensor, to_numpy=False):
    if to_numpy:
        return np.array(tensor.detach().cpu())
    else:
        return tensor.detach().cpu()
