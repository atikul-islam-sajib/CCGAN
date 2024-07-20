import sys
import torch
import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

sys.path.append("./")

from utils import config, weight_init, device_init, CustomException
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
        weight_initialization: bool = True,
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
        self.weight_initialization = weight_initialization
        self.adam = adam
        self.SGD = SGD
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.elasticnet_regularization = elasticnet_regularization
        self.lr_scheduler = lr_scheduler
        self.verbose = verbose
        self.mlflow = mlflow

        self.device = device_init(device=self.device)

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

        self.netG = self.init["netG"].to(self.device)
        self.netD = self.init["netD"].to(self.device)

        if self.weight_initialization:
            self.netG.apply(weight_init)
            self.netD.apply(weight_init)

        self.optimizerG = self.init["optimizerG"]
        self.optimizerD = self.init["optimizerD"]

        self.adversarial_loss = self.init["loss"]

        if self.lr_scheduler:
            self.schedulerG = StepLR(
                optimizer=self.optimizerG, step_size=self.step_size, gamma=self.gamma
            )
            self.schedulerD = StepLR(
                optimizer=self.optimizerD, step_size=self.step_size, gamma=self.gamma
            )

    def l1(self, model: Discriminator):
        if isinstance(model, Discriminator):
            self.loss = sum(torch.norm(params, 1) for params in model.parameters())
            return self.weight_delacy * self.loss

        else:
            raise TypeError("model must be an instance of Discriminator".capitalize())

    def l2(self, model: Discriminator):
        if isinstance(model, Discriminator):
            self.loss = sum(torch.norm(params, 2) for params in model.parameters())
            return self.weight_delacy * self.loss

        else:
            raise TypeError("model must be an instance of Discriminator".capitalize())

    def elasticNet(self, model: Discriminator):
        if isinstance(model, Discriminator):
            self.loss1 = self.l1(model=model)
            self.loss2 = self.l2(model=model)

            return self.weight_delacy * (self.loss1 + self.loss2)

        else:
            raise TypeError("model must be an instance of Discriminator".capitalize())

    def saved_checkpoints(self, **kwargs: dict):
        pass

    def saved_images(self, **kwargs: dict):
        pass

    def display_progress(self, **kwargs: dict):
        pass

    def update_train_generator(self, **kwargs: dict):
        pass

    def update_train_discriminator(self, **kwargs: dict):
        pass

    def train(self):
        pass

    @staticmethod
    def display_history():
        pass


if __name__ == "__main__":
    trainer = Trainer(
        epochs=1,
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        momentum=0.90,
        weight_delacy=0.001,
        step_size=20,
        gamma=0.85,
        device="mps",
        adam=True,
        SGD=False,
        l1_regularization=False,
        l2_regularization=False,
        elasticnet_regularization=False,
        lr_scheduler=False,
        verbose=True,
        mlflow=False,
    )
