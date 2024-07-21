import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./")


class PixelLoss(nn.Module):
    def __init__(self, name: str = "pixelloss".title(), reduction: str = "mean"):
        super(PixelLoss, self).__init__()

        self.name = name
        self.reduction = reduction

        self.loss = nn.L1Loss(reduction=self.reduction)

    def forward(self, pred: torch.Tensor, actual: torch.Tensor):
        if isinstance(pred, torch.Tensor) and isinstance(actual, torch.Tensor):
            return self.loss(pred, actual)

        else:
            raise TypeError("Both inputs must be of type torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Loss".title())
    parser.add_argument(
        "--name",
        type=str,
        default="pixelloss".title(),
        help="Name of the loss function".capitalize(),
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        help="Reduction method for the loss function".capitalize(),
    )

    args = parser.parse_args()

    predicted = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    actual = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    loss = PixelLoss(name=args.name, reduction=args.reduction)

    assert loss(predicted, actual) == torch.abs(predicted - actual).mean()
