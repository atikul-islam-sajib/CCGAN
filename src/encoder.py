import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./")


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        batchnorm: bool = True,
        leakyrelu: bool = False,
    ):
        super(EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.leakyrelu = leakyrelu

        self.kernel_size = 3
        self.stride_size = 2
        self.padding_size = 1
        self.momentum = 0.8
        self.negative_slope = 0.2

        self.encoder_block = self.block()

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
        if self.batchnorm:
            self.layers.append(
                nn.BatchNorm2d(num_features=self.out_channels, momentum=self.momentum)
            )
        if self.leakyrelu:
            self.layers.append(
                nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)
            )

        return nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.encoder_block(x)
        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoder Block for Generator".title())
    parser.add_argument(
        "--in_channels", type=int, default=3, help="Input Channels".title()
    )
    parser.add_argument(
        "--out_channels", type=int, default=64, help="Output Channels".title()
    )
    parser.add_argument(
        "--batchnorm", type=bool, default=True, help="Batch Normalization".title()
    )
    parser.add_argument(
        "--leakyrelu", type=bool, default=False, help="Leaky ReLU".title()
    )

    args = parser.parse_args()

    encoder = EncoderBlock(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        batchnorm=args.batchnorm,
        leakyrelu=args.leakyrelu,
    )

    assert encoder(torch.randn(1, 3, 128, 128)).size() == torch.Size(
        [1, 64, 64, 64]
    ), "Encoder Block is not working properly".capitalize()
