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

from .metrics import AverageMeter, Metrics


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
        loss: _Loss,
        optimizer: Optimizer = None,
        logger: logging.Logger = None,
    ):
        super().__init__(model, loss, optimizer, logger)

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
        threshold: float = 0.5,
    ) -> Tuple[float, float]:

        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        for i, batch in enumerate(dataloader):
            xi, xj = batch

            if phase == "train":
                vector_i = self.model(xi)
                vector_j = self.model(xj)

                return vector_i, vector_j

            # inference
            else:
                raise NotImplementedError

            loss = self.loss(logit_i, logit_j)

            if phase == "train":
                loss.backward()
                self.optimizer.step()

            total_loss = loss.item()
            accuracy = self.get_accuracy(logit_i, xj, threshold)
            auroc = self.get_auroc(logit_i, xj)

            bar_sentence = self.make_bar_sentence(
                phase,
                epoch,
                len(dataloader),
                i + 1,
                "N/A",
                total_loss,
                accuracy,
            )


class BinaryClassifierTrainer(ABC):
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
        self.logger = (
            logging.Logger("BinaryClassifierTrainer") if logger is None else logger
        )

    def make_bar_sentence(
        self,
        phase: str,
        epoch: int,
        total_step: int,
        step: int,
        eta: str,
        total_loss: float,
        accuracy: float,
        auroc: float,
        prauc: float,
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
        accuracy = round(accuracy, 5)
        auroc = round(auroc, 5)
        prauc = round(prauc, 5)

        return (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | total_loss: {total_loss} | "
            f"accuracy: {accuracy} | auroc: {auroc} | prauc: {prauc}"
        )

    def run_epoch(
        self,
        phase: str,
        epoch: int,
        dataloader: torch.utils.data.DataLoader,
        threshold: float = 0.5,
    ) -> Tuple[float, float]:
        """1회 Epoch을 각 페이즈(train, validation)에 따라서 학습하거나 손실값을
        반환함.

        Note:
            - 1 epoch = Dataset의 전체를 학습한경우
            - 1 step = epoch을 하기위해 더 작은 단위(batch)로 학습할 떄의 단위

        Args:
            phase (str): training or validation
            epoch (int): epoch
            dataloader (torch.utils.data.DataLoader): dataset (train or validation)

        Returns:
            Tuple: loss, accuracy, top_k_recall
        """
        total_step = len(dataloader)
        bar = Bar(max=total_step, check_tty=False)

        loss_meter = AverageMeter("loss")
        metrics_meter = MetricsMeter(name=phase, accuracy_threshold=threshold)
        for step, batch in enumerate(dataloader):
            xs, ys = batch

            if phase == "train":
                self.model.train()
                logits = self.model(xs)
            else:
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(xs)

            loss = self.loss(logits, ys)

            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # metric
            loss_meter.update(loss.item(), len(ys))
            flatten_ys = ys.flatten()
            model_confidence = torch.sigmoid(logits).flatten()
            metrics_meter.update(
                model_confidence.detach().cpu().numpy().tolist(),
                flatten_ys.cpu().numpy().tolist(),
            )
            self.logger.debug(
                f"Step({step}): \n"
                f"logits({str(logits)}) \n"
                f"labels({str(flatten_ys)})"
            )
            self.logger.debug(
                f"Step ({step}): Accuracy({metrics_meter.accuracy}), "
                f"AUROC({metrics_meter.auroc}), PRAUC({metrics_meter.prauc})"
            )

            bar.suffix = self.make_bar_sentence(
                phase=phase,
                epoch=epoch,
                step=step,
                total_step=total_step,
                eta=bar.eta,
                total_loss=loss_meter.avg,
                accuracy=metrics_meter.accuracy,
                auroc=metrics_meter.auroc,
                prauc=metrics_meter.prauc,
            )
            bar.next()

        bar.finish()

        return (loss_meter, metrics_meter)
