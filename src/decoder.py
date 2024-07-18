import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        batchnorm: bool = True,
        leakyrelu: bool = True,
    ):
        super(DecoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batchnorm = batchnorm
        self.leakyrelu = leakyrelu

        self.kernel_size = 4
        self.stride_size = 2
        self.padding_size = 1
        self.momentum = 0.8

        self.decoder_block = self.block()

    def block(self):
        self.layers = []

        self.layers.append(
            nn.ConvTranspose2d(
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
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            self.layers.append(nn.Tanh())

        return nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.decoder_block(x)

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder Block for Generator".title())
    parser.add_argument(
        "--in_channels",
        type=int,
        default=512,
        help="Number of input channels".capitalize(),
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=512,
        help="Number of output channels".capitalize(),
    )
    parser.add_argument(
        "--batchnorm", type=bool, default=True, help="Batch Normalization".capitalize()
    )
    parser.add_argument(
        "--leakyrelu", type=bool, default=True, help="Leaky ReLU".capitalize()
    )

    args = parser.parse_args()

    decoder = DecoderBlock(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        batchnorm=args.batchnorm,
        leakyrelu=args.leakyrelu,
    )

    assert decoder(torch.randn(1, 512, 2, 2)).size() == torch.Size(
        [1, 512, 4, 4]
    ), "Decoder Block is not working properly".capitalize()
