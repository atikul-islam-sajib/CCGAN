import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./")


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
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

    def block(self):
        pass
