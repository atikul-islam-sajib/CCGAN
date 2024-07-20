import os
import sys
import torch
import dagshub
import mlflow
import warnings
import traceback
import pandas as pd
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image

warnings.filterwarnings("ignore")

sys.path.append("./")

from utils import dump, load, config, weight_init, device_init, CustomException
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
        weight_decay: float = 0.001,
        step_size: int = 20,
        gamma: float = 0.85,
        threshold: int = 50,
        device: str = "cpu",
        weight_initialization: bool = True,
        adam: bool = True,
        SGD: bool = False,
        pixelLoss: bool = False,
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
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.threshold = threshold
        self.device = device
        self.weight_initialization = weight_initialization
        self.adam = adam
        self.SGD = SGD
        self.pixelLoss = pixelLoss
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

        if self.mlflow:
            try:
                dagshub.init(
                    repo_owner=config()["MLFlow"]["MLFLOW_TRACKING_USERNAME"],
                    repo_name=config()["MLFlow"]["REPO_NAME"],
                    mlflow=self.mlflow,
                )
            except Exception as e:
                print("An error occured while initializing dagshub: ", e)
                traceback.print_exc()

        self.loss = float("inf")

        self.total_netG_loss = []
        self.total_netD_loss = []

        self.history = {"netG_loss": [], "netD_loss": []}

    def l1(self, model: Discriminator):
        if isinstance(model, Discriminator):
            self.loss = sum(torch.norm(params, 1) for params in model.parameters())
            return self.weight_decay * self.loss

        else:
            raise TypeError("model must be an instance of Discriminator".capitalize())

    def l2(self, model: Discriminator):
        if isinstance(model, Discriminator):
            self.loss = sum(torch.norm(params, 2) for params in model.parameters())
            return self.weight_decay * self.loss

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
        epoch = kwargs["epoch"]
        netG_loss = kwargs["netG_loss"]
        netD_loss = kwargs["netD_loss"]

        if epoch > self.threshold:
            if self.loss > netG_loss:
                self.loss = netG_loss

                torch.save(
                    {
                        "netG": self.netG.state_dict(),
                        "epoch": epoch,
                        "netG_loss": netG_loss,
                        "netD_loss": netD_loss,
                    },
                    os.path.join(config()["path"]["BEST_MODEL"], "bestmodel.pth"),
                )

        torch.save(
            self.netG.state_dict(),
            os.path.join(config()["path"]["TRAIN_MODELS"], f"netG{epoch}.pth"),
        )

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

        if self.pixelLoss:
            pixel_loss = torch.abs(generated_image - y).mean()
            generated_loss += pixel_loss

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
        mlflow.set_experiment("CCGAN".title())
        with mlflow.start_run(
            description="Context-Conditional Generative Adversarial Networks (CC-GANs) are conditional GANs where The Generator 𝐺 is trained to fill in a missing image"
        ) as run:
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

                if epoch % self.step_size == 0:
                    self.saved_images(epoch=epoch + 1)

                self.saved_checkpoints(
                    epoch=epoch + 1,
                    netG_loss=np.mean(self.netG_loss),
                    netD_loss=np.mean(self.netD_loss),
                )

                try:
                    self.history["netG_loss"].append(np.mean(self.netG_loss))
                    self.history["netD_loss"].append(np.mean(self.netD_loss))

                    self.total_netG_loss.append(np.mean(self.netG_loss))
                    self.total_netD_loss.append(np.mean(self.netD_loss))

                except CustomException as e:
                    print("An error occurred: ", e)
                    traceback.print_exc()
                except Exception as e:
                    print("An error occurred: ", e)
                    traceback.print_exc()

                mlflow.log_params(
                    {
                        "epoch": self.epochs,
                        "lr": self.lr,
                        "beta1": self.beta1,
                        "beta2": self.beta2,
                        "momentum": self.momentum,
                        "weight_decay": self.weight_decay,
                        "step_size": self.step_size,
                        "gamma": self.gamma,
                        "device": self.device,
                        "adam": self.adam,
                        "SGD": self.SGD,
                        "pixelLoss": self.pixelLoss,
                        "l1_regularization": self.l1_regularization,
                        "l2_regularization": self.l2_regularization,
                        "lr_scheduler": self.lr_scheduler,
                        "verbose": self.verbose,
                        "MLFlow": self.mlflow,
                    }
                )

                mlflow.log_metric("netG_loss", np.mean(self.netG_loss), step=epoch + 1)
                mlflow.log_metric("netD_loss", np.mean(self.netD_loss), step=epoch + 1)

            mlflow.pytorch.log_model(self.netG, "netG")
            mlflow.pytorch.log_model(self.netD, "netD")

            try:
                dump(
                    value=self.history,
                    filename=os.path.join(
                        config()["path"]["METRICS_PATH"], "history.pkl"
                    ),
                )
                print(
                    "Model history saved in the folder {}".format(
                        config()["path"]["METRICS_PATH"]
                    )
                )
            except Exception as e:
                print("An error occurred while logging the history: ", e)

            else:
                pd.DataFrame(
                    {
                        "netG_loss": self.total_netG_loss,
                        "netD_loss": self.total_netD_loss,
                    }
                ).to_csv(os.path.join(config()["path"]["FILES_PATH"], "model_loss.csv"))

                print(
                    "Model loss saved in the format of csv in the directory {}".format(
                        config()["path"]["FILES_PATH"]
                    )
                )

    @staticmethod
    def display_history():
        metrics_path = os.path.join(config()["path"]["METRICS_PATH"], "history.pkl")
        history = load(filename=metrics_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("netG vs netD loss".title(), fontsize=16)

        axes[0].plot(history["netG_loss"], label="netG_loss")
        axes[0].legend()
        axes[0].grid(which="both", linestyle="--")
        axes[0].set_title("netG loss".title())
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Loss")

        axes[1].plot(history["netD_loss"], label="netD_loss")
        axes[1].legend()
        axes[1].grid(which="both", linestyle="--")
        axes[1].set_title("netD loss".title())
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the model for CCGAN".capitalize()
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config()["trainer"]["epochs"],
        help="Number of epochs to train the model".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["trainer"]["lr"],
        help="Learning rate for the model".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["trainer"]["beta1"],
        help="Beta1 for Adam".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["trainer"]["beta2"],
        help="Beta2 for Adam".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config()["trainer"]["momentum"],
        help="Momentum for SGD".capitalize(),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config()["trainer"]["weight_decay"],
        help="Weight decay for the model".capitalize(),
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=config()["trainer"]["step_size"],
        help="Step size for the scheduler".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config()["trainer"]["gamma"],
        help="Gamma for the scheduler".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Device to train the model on".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["trainer"]["adam"],
        help="Use Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["trainer"]["SGD"],
        help="Use SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--pixelLoss",
        type=bool,
        default=config()["trainer"]["pixelLoss"],
        help="Use pixel loss".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=config()["trainer"]["l1_regularization"],
        help="Use L1 loss".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        type=bool,
        default=config()["trainer"]["l2_regularization"],
        help="Use L2 loss".capitalize(),
    )
    parser.add_argument(
        "--elasticnet_regularization",
        type=bool,
        default=config()["trainer"]["elasticnet_regularization"],
        help="Use elastic loss".capitalize(),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=config()["trainer"]["lr_scheduler"],
        help="Use lr scheduler".capitalize(),
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config()["trainer"]["verbose"],
        help="Verbose".capitalize(),
    )
    parser.add_argument(
        "--mlflow",
        type=bool,
        default=config()["trainer"]["mlflow"],
        help="Use mlflow".capitalize(),
    )

    args = parser.parse_args()

    trainer = Trainer(
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        device=args.device,
        adam=args.adam,
        SGD=args.SGD,
        pixelLoss=args.pixelLoss,
        l1_regularization=args.l1_regularization,
        l2_regularization=args.l2_regularization,
        elasticnet_regularization=args.elasticnet_regularization,
        lr_scheduler=args.lr_scheduler,
        verbose=args.verbose,
        mlflow=args.mlflow,
    )

    trainer.train()
