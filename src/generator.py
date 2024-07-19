import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./")

from utils import config, parse_tuple
from encoder import EncoderBlock
from decoder import DecoderBlock


class Generator(nn.Module):
    def __init__(self, image_size: tuple = (1, 3, 128, 128)):
        super(Generator, self).__init__()
        self.batch_size, self.channels, self.image_height, self.image_width = image_size

        self.encoder1 = EncoderBlock(
            in_channels=self.channels,
            out_channels=self.image_height // 2,
            batchnorm=False,
        )
        self.encoder2 = EncoderBlock(
            in_channels=self.image_height // 2,
            out_channels=self.image_height,
            batchnorm=True,
        )
        self.encoder3 = EncoderBlock(
            in_channels=self.image_height + self.channels,
            out_channels=self.image_height * 2,
            batchnorm=True,
        )
        self.encoder4 = EncoderBlock(
            in_channels=self.image_height * 2,
            out_channels=self.image_height * 4,
            batchnorm=True,
        )
        self.encoder5 = EncoderBlock(
            in_channels=self.image_height * 4,
            out_channels=self.image_height * 4,
            batchnorm=True,
        )
        self.encoder6 = EncoderBlock(
            in_channels=self.image_height * 4,
            out_channels=self.image_height * 4,
            batchnorm=True,
        )

        self.decoder1 = DecoderBlock(
            in_channels=self.image_height * 4, out_channels=self.image_height * 4
        )
        self.decoder2 = DecoderBlock(
            in_channels=self.image_height * 8, out_channels=self.image_height * 4
        )
        self.decoder3 = DecoderBlock(
            in_channels=self.image_height * 8, out_channels=self.image_height * 2
        )
        self.decoder4 = DecoderBlock(
            in_channels=self.image_height * 4, out_channels=self.image_height
        )
        self.decoder5 = DecoderBlock(
            in_channels=self.image_height * 2 + self.channels,
            out_channels=self.image_height // 2,
        )
        self.decoder6 = DecoderBlock(
            in_channels=self.image_height, out_channels=self.channels
        )

    def forward(self, x: torch.Tensor, lr_image: torch.Tensor):
        if (isinstance(x, torch.Tensor)) and (isinstance(lr_image, torch.Tensor)):
            encoder1 = self.encoder1(x)

            encoder2 = self.encoder2(encoder1)

            _encoder2 = torch.cat((encoder2, lr_image), dim=1)

            encoder3 = self.encoder3(_encoder2)
            encoder3 = torch.dropout(input=encoder3, p=0.5, train=self.training)

            encoder4 = self.encoder4(encoder3)
            encoder4 = torch.dropout(input=encoder4, p=0.5, train=self.training)

            encoder5 = self.encoder5(encoder4)
            encoder5 = torch.dropout(input=encoder5, p=0.5, train=self.training)

            encoder6 = self.encoder6(encoder5)
            encoder6 = torch.dropout(input=encoder6, p=0.5, train=self.training)

            decoder1 = torch.cat((self.decoder1(encoder6), encoder5), dim=1)
            decoder1 = torch.dropout(input=decoder1, p=0.5, train=self.training)

            decoder2 = torch.cat((self.decoder2(decoder1), encoder4), dim=1)
            decoder2 = torch.dropout(input=decoder2, p=0.5, train=self.training)

            decoder3 = torch.cat((self.decoder3(decoder2), encoder3), dim=1)
            decoder3 = torch.dropout(input=decoder3, p=0.5, train=self.training)

            decoder4 = torch.cat((self.decoder4(decoder3), _encoder2), dim=1)

            decoder5 = torch.cat((self.decoder5(decoder4), encoder1), dim=1)

            output = self.decoder6(decoder5)

            return output

    @staticmethod
    def total_params(model=None):
        if isinstance(model, Generator):
            return sum(params.numel() for params in model.parameters())

        else:
            raise TypeError("Model must be of type Generator".capitalize())


if __name__ == "__main__":
    batch_size = config()["dataloader"]["batch_size"]
    channels = config()["dataloader"]["channels"]
    image_size = config()["dataloader"]["image_size"]

    image = (batch_size, channels, image_size, image_size)

    parser = argparse.ArgumentParser(description="Generator for CCGAN".title())
    parser.add_argument(
        "--image_size",
        type=parse_tuple,
        default=image,
        help="Image size (e.g., '(1,3,128,128)')".capitalize(),
    )

    args = parser.parse_args()

    netG = Generator(image_size=args.image_size)

    assert (
        netG(
            torch.randn(args.image_size),
            torch.randn(
                args.image_size[0],
                args.image_size[1],
                args.image_size[2] // 4,
                args.image_size[3] // 4,
            ),
        ).size()
    ) == (args.image_size), "Image size is incorrect in Generator".capitalize()

    print("Total params of the netG = {}".format(Generator.total_params(netG)))
