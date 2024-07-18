import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./")

from utils import config


class DiscriminatorBlock(nn.Module):
    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, instance_norm: bool = True
    ):
        super(DiscriminatorBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.instance_norm = instance_norm

        self.kernel_size = 3
        self.stride_size = 2
        self.padding_size = 1
        self.negative_slope = 0.2

        self.discriminator_block = self.block()

    def block(self):
        self.layers = []

        self.layers.append(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride_size,
                padding=self.padding_size,
                bias=False,
            )
        )

        if self.instance_norm:
            self.layers.append(nn.InstanceNorm2d(num_features=self.out_channels))

        self.layers.append(
            nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)
        )

        return nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.discriminator_block(x)

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    image_size = config()["dataloader"]["image_size"]

    parser = argparse.ArgumentParser(
        description="Discriminator Block for CCGAN".title()
    )
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Input Channels".capitalize()
    )
    parser.add_argument(
        "--out_channels", type=int, default=64, help="Output Channels".capitalize()
    )
    parser.add_argument(
        "--instance_norm",
        type=bool,
        default=True,
        help="Instance Normalization".capitalize(),
    )

    args = parser.parse_args()

    netD = DiscriminatorBlock(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        instance_norm=args.instance_norm,
    )

    assert netD(
        torch.randn(1, args.in_channels, image_size, image_size)
    ).shape == torch.Size([1, args.out_channels, image_size // 2, image_size // 2])
