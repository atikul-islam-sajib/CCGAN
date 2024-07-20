import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./")

from helper import helpers
from generator import Generator
from discriminator import Discriminator
from loss import AdversarialLoss


class Trainer:
    def __init__(
        self,
        epochs: int = 100,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.9999,
        momentum: float = 0.9,
        weight_delacy: float = 0.001,
        step_size: int = 20,
        gamma: float = 0.85,
        device: str = "cpu",
        adam: bool = True,
        SGD: bool = False,
        l1_regularization: bool = False,
        l2_regularization: bool = False,
        elasticnet_regularization: bool = False,
        lr_scheduler: bool = False,
        verbose: bool = False,
        mlflow: bool = False,
    ):
        self.epochs = epochs
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.weight_delacy = weight_delacy
        self.step_size = step_size
        self.gamma = gamma
        self.device = device
        self.adam = adam
        self.SGD = SGD
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.elasticnet_regularization = elasticnet_regularization
        self.lr_scheduler = lr_scheduler
        self.verbose = verbose
        self.mlflow = mlflow

        self.init = helpers(
            lr=self.lr,
            adam=self.adam,
            SGD=self.SGD,
            momentum=self.momentum,
            beta1=self.beta1,
            beta2=self.beta2,
        )

        assert (
            self.init["train_dataloader"].__class__ == torch.utils.data.DataLoader
        ), "Train dataloader must be a PyTorch DataLoader".capitalize()
        assert (
            self.init["valid_dataloader"].__class__ == torch.utils.data.DataLoader
        ), "Valid dataloader must be a PyTorch DataLoader".capitalize()

        assert (
            self.init["netG"].__class__ == Generator
        ), "netG must be a Generator".capitalize()
        assert (
            self.init["netD"].__class__ == Discriminator
        ), "netD must be a Discriminator".capitalize()

        assert (
            self.init["optimizerG"].__class__ == torch.optim.Adam
        ), "optimizerG must be an Adam optimizer".capitalize()
        assert (
            self.init["optimizerD"].__class__ == torch.optim.Adam
        ), "optimizerD must be an Adam optimizer".capitalize()

        assert (
            self.init["loss"].__class__ == AdversarialLoss
        ), "loss must be an AdversarialLoss".capitalize()

        self.train_dataloader = self.init["train_dataloader"]
        self.valid_dataloader = self.init["valid_dataloader"]

        self.netG = self.init["netG"]
        self.netD = self.init["netD"]

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD = self.init["optimizerD"]

        self.adversarial_loss = self.init["loss"]
