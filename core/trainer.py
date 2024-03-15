import os
import sys
import math
import logging
from typing import Tuple, Literal
from abc import ABC, abstractmethod

import numpy as np
import torch
from progress.bar import Bar
from sklearn.metrics import roc_auc_score
from torch.nn.modules.loss import _Loss
from torch.nn.modules.module import Module
from torch.optim.optimizer import Optimizer as Optimizer

from .metrics import AverageMeter
from .loss import SimCLRLoss


class BaseTrainer(ABC):
    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.logger = logging.Logger("Trainer") if logger is None else logger

    @abstractmethod
    def make_bar_sentence(self):
        pass

    @abstractmethod
    def run_epoch(self):
        pass

    def get_accuracy(
        self, logit: torch.Tensor, labels: torch.Tensor, threshold: int = 0.5
    ) -> float:
        confidence = torch.sigmoid(logit).flatten()
        pred_labels = (confidence > threshold).float().flatten()
        return (pred_labels == labels).sum().item() / len(labels)

    def get_auroc(self, logit: torch.Tensor, labels: torch.Tensor) -> float:
        confidence = torch.sigmoid(logit).flatten()
        return roc_auc_score(labels.flatten(), confidence)


class SimCLRTrainer(BaseTrainer):
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer = None,
        logger: logging.Logger = None,
        device: str = "cuda",
    ):
        super().__init__(
            model, loss=SimCLRLoss(device=device), optimizer=optimizer, logger=logger
        )

    def make_bar_sentence(
        self,
        phase: str,
        epoch: int,
        total_step: int,
        step: int,
        eta: str,
        total_loss: float,
    ) -> str:
        """ProgressBar의 stdout의 string을 생성하여 반환

        Args:
            phase (str): Epoch의 phase
            epoch (int): epoch
            total_step (int): total steps for one epoch
            step (int): Step (in a epoch)
            eta (str): Estimated Time of Arrival
            loss (float): loss
            accuracy (float): accuracy
            auroc (float): auroc
            prauc (float): prauc

        Returns:
            str: progressbar senetence

        """
        total_loss = round(total_loss, 5)

        return (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | total_loss: {total_loss}"
        )

    def run_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        phase: Literal["train", "val", "Test"],
        epoch: int,
    ) -> AverageMeter:

        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        loss_meter = AverageMeter("loss")
        for i, batch in enumerate(dataloader):
            xi, xj = batch

            if phase == "train":
                vector_i = self.model(xi)
                vector_j = self.model(xj)

            else:
                with torch.no_grad():
                    vector_i = self.model(xi)
                    vector_j = self.model(xj)

            loss = self.loss(vector_i, vector_j)

            if phase == "train":
                loss.backward()
                self.optimizer.step()

            loss_meter.update(loss.item(), len(xj))
            bar.suffix = self.make_bar_sentence(
                phase,
                epoch,
                total_step,
                i + 1,
                bar.eta,
                loss_meter.avg,
            )
            bar.next()

        bar.finish()

        return loss_meter
