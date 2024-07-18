import sys
import torch
import string
import argparse
import torch.nn as nn
from torchsummary import summary

sys.path.append("./")

from utils import config
from discriminator_block import DiscriminatorBlock


class Discriminator(nn.Module):
    def __init__(self, image_size: tuple = (1, 3, 128, 128)):
        super(Discriminator, self).__init__()

        self.batch_size, self.channels, self.image_height, self.image_width = image_size
        self.out_channels = 64
        self.kernel_size = 3
        self.stride_size = 2
        self.padding_size = 1

        self.layers = []

        for _ in range(4):
            self.layers.append(
                DiscriminatorBlock(
                    in_channels=self.channels, out_channels=self.out_channels
                )
            )
            self.channels = self.out_channels
            self.out_channels *= 2

        self.layers.append(
            nn.Conv2d(
                in_channels=self.channels,
                out_channels=self.channels // self.channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size // self.stride_size,
                padding=1,
            )
        )

        self.model = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.model(x)
        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


def parse_tuple(string):
    return tuple(map(int, string.strip("()").split(",")))


if __name__ == "__main__":
    batch_size = config()["dataloader"]["batch_size"]
    channels = config()["dataloader"]["channels"]
    image_height = config()["dataloader"]["image_size"]
    image_width = config()["dataloader"]["image_size"]

    image_size = (batch_size, channels, image_height, image_width)

    parser = argparse.ArgumentParser(
        description="Discriminator Block for CCGAN".title()
    )
    parser.add_argument(
        "--image_size",
        type=parse_tuple,
        default=image_size,
        help="Image size (e.g., '(1,3,128,128)')".capitalize(),
    )

    args = parser.parse_args()

    netD = Discriminator(image_size=args.image_size)

    assert netD(torch.randn(args.image_size)).size() == torch.Size(
        [
            args.image_size[0],
            args.image_size[1] // args.image_size[1],
            args.image_size[2] // 16,
            args.image_size[3] // 16,
        ]
    )

    summary(model=netD, input_size=args.image_size[1:])
