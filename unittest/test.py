import os
import sys
import torch
import unittest

sys.path.append("./src")

from helper import helpers
from generator import Generator
from loss import AdversarialLoss
from discriminator import Discriminator
from utils import config, load, connect_database


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.train_dataloader = os.path.join(
            config()["path"]["PROCESSED_IMAGE_DATA_PATH"], "train_dataloader.pkl"
        )
        self.valid_dataloader = os.path.join(
            config()["path"]["PROCESSED_IMAGE_DATA_PATH"], "valid_dataloader.pkl"
        )

        self.train_dataloader = load(filename=self.train_dataloader)
        self.valid_dataloader = load(filename=self.valid_dataloader)

        self.batch_size = config()["dataloader"]["batch_size"]
        self.channels = config()["dataloader"]["channels"]
        self.image_size = config()["dataloader"]["image_size"]

        self.netG = Generator(
            image_size=(
                self.batch_size,
                self.channels,
                self.image_size,
                self.image_size,
            )
        )

        self.netD = Discriminator(
            image_size=(
                self.batch_size,
                self.channels,
                self.image_size,
                self.image_size,
            )
        )

        self.init = helpers(
            lr=2e-4, adam=True, SGD=True, momentum=0.9, beta1=0.5, beta2=0.999
        )

        self.netG_params = 29260422
        self.netD_params = 1554625

    def test_type_dataloader(self):
        self.assertIsInstance(self.train_dataloader, torch.utils.data.DataLoader)
        self.assertIsInstance(self.valid_dataloader, torch.utils.data.DataLoader)

    def test_total_data(self):
        self.directory = os.path.join(
            config()["path"]["RAW_IMAGE_DATA_PATH"], "dataset"
        )
        self.X = os.path.join(self.directory, "X")

        self.total_data1 = len(os.listdir(self.X))

        self.total_data2 = sum(X.size(0) for X, y, _ in self.train_dataloader) + sum(
            X.size(0) for X, y, _ in self.valid_dataloader
        )

        self.assertEqual(self.total_data1, self.total_data2)

    def test_lr_image_data(self):
        self.image_size = config()["dataloader"]["image_size"]
        self.channels = config()["dataloader"]["channels"]
        self.batch_size = config()["dataloader"]["batch_size"]

        _, _, lr_image = next(iter(self.train_dataloader))

        self.assertEqual(
            lr_image.size(),
            torch.Size(
                [
                    self.batch_size,
                    self.channels,
                    self.image_size // 4,
                    self.image_size // 4,
                ]
            ),
        )

    def test_generator(self):
        self.assertEqual(
            self.netG(
                torch.randn(
                    self.batch_size, self.channels, self.image_size, self.image_size
                ),
                torch.randn(
                    self.batch_size,
                    self.channels,
                    self.image_size // 4,
                    self.image_size // 4,
                ),
            ).size(),
            torch.Size(
                [self.batch_size, self.channels, self.image_size, self.image_size]
            ),
        )

        # Check for other dimension for the image

        self.assertEqual(
            self.netG(
                torch.randn(
                    self.batch_size,
                    self.channels,
                    self.image_size * 2,
                    self.image_size * 2,
                ),
                torch.randn(
                    self.batch_size,
                    self.channels,
                    self.image_size * 2 // 4,
                    self.image_size * 2 // 4,
                ),
            ).size(),
            torch.Size(
                [
                    self.batch_size,
                    self.channels,
                    self.image_size * 2,
                    self.image_size * 2,
                ]
            ),
        )

        # Check for other dimension for the image

        self.assertEqual(
            self.netG(
                torch.randn(
                    self.batch_size,
                    self.channels,
                    self.image_size * 2,
                    self.image_size * 2,
                ),
                torch.randn(
                    self.batch_size,
                    self.channels,
                    self.image_size * 2 // 4,
                    self.image_size * 2 // 4,
                ),
            ).size(),
            torch.Size(
                [
                    self.batch_size,
                    self.channels,
                    self.image_size * 2,
                    self.image_size * 2,
                ]
            ),
        )

    def test_discriminator(self):
        self.assertEqual(
            self.netD(
                torch.randn(
                    self.batch_size, self.channels, self.image_size, self.image_size
                )
            ).size(),
            torch.Size(
                [
                    self.batch_size,
                    self.channels // self.channels,
                    self.image_size // 16,
                    self.image_size // 16,
                ]
            ),
        )

    def test_netG_params(self):
        self.assertEqual(
            sum(params.numel() for params in self.netG.parameters()), self.netG_params
        )

    def test_netD_params(self):
        self.assertEqual(
            sum(params.numel() for params in self.netD.parameters()), self.netD_params
        )

    def test_helper(self):
        assert self.init["train_dataloader"].__class__ == torch.utils.data.DataLoader
        assert self.init["valid_dataloader"].__class__ == torch.utils.data.DataLoader

        assert self.init["netG"].__class__ == Generator
        assert self.init["netD"].__class__ == Discriminator

        assert self.init["optimizerG"].__class__ == torch.optim.Adam
        assert self.init["optimizerD"].__class__ == torch.optim.Adam

        assert self.init["loss"].__class__ == AdversarialLoss

    def test_mongoDB(self):
        is_connect, _ = connect_database()
        self.assertEqual(
            is_connect, True
        )  # Must be change the config.yml file "database" where "USERNAME": "zyz" and "PASSWORD": "123456"


if __name__ == "__main__":
    unittest.main()
