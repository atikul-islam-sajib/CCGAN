import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

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

        self.history = {"netG_loss": [], "netD_loss": []}

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
        epoch = kwargs["epoch"]

        X, y, lr_image = next(iter(self.train_dataloader))

        X = X.to(self.device)
        y = y.to(self.device)
        lr_image = lr_image.to(self.device)

        generated_images = self.netG(X, lr_image)

        save_image(
            generated_images,
            os.path.join(
                config()["path"]["TRAIN_OUTPUT_IMAGES"], "train_image{}.png"
            ).format(epoch),
        )

    def display_progress(self, **kwargs: dict):
        epoch = kwargs["epoch"]
        netG_loss = kwargs["netG_loss"]
        netD_loss = kwargs["netD_loss"]

        if self.verbose:
            print(
                "Epochs: [{}/{}] - netG_loss: [{:.4f}] - netD_loss: [{:.4f}]".format(
                    epoch, self.epochs, netG_loss, netD_loss
                )
            )
        else:
            print(
                "Epochs: [{}/{}] is completed".capitalize().format(epoch, self.epochs)
            )

    def update_train_generator(self, **kwargs: dict):
        X = kwargs["X"]
        y = kwargs["y"]
        lr_image = kwargs["lr_image"]

        self.optimizerG.zero_grad()

        generated_image = self.netG(X, lr_image)
        generated_predict = self.netD(generated_image)

        generated_loss = self.adversarial_loss(
            generated_predict, torch.ones_like(generated_predict)
        )

        generated_loss.backward()
        self.optimizerG.step()

        return generated_loss.item()

    def update_train_discriminator(self, **kwargs: dict):
        X = kwargs["X"]
        y = kwargs["y"]
        lr_image = kwargs["lr_image"]

        self.optimizerD.zero_grad()

        real_predict = self.netD(y)
        real_image_loss = self.adversarial_loss(
            real_predict, torch.ones_like(real_predict)
        )

        generated_image = self.netG(X, lr_image)
        generated_predict = self.netD(generated_image)

        generated_image_loss = self.adversarial_loss(
            generated_predict, torch.zeros_like(generated_predict)
        )

        total_loss = 0.5 * (real_image_loss + generated_image_loss)

        total_loss.backward()
        self.optimizerD.step()

        return total_loss.item()

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            self.netG_loss = []
            self.netD_loss = []

            for index, (X, y, lr_image) in enumerate(self.train_dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                lr_image = lr_image.to(self.device)

                self.netD_loss.append(
                    self.update_train_discriminator(X=X, y=y, lr_image=lr_image)
                )

                self.netG_loss.append(
                    self.update_train_generator(X=X, y=y, lr_image=lr_image)
                )

            if self.lr_scheduler:
                self.schedulerG.step()
                self.schedulerD.step()

            self.display_progress(
                epoch=epoch + 1,
                netG_loss=np.mean(self.netG_loss),
                netD_loss=np.mean(self.netD_loss),
            )

            if self.epochs % self.step_size:
                self.saved_images(epoch=epoch + 1)

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

    trainer.train()
