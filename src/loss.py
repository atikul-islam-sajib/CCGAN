import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./")


class AdversarialLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super(AdversarialLoss, self).__init__()

        self.reduction = reduction
        self.loss = nn.MSELoss(reduction=self.reduction)

    def forward(self, actual: torch.Tensor, pred: torch.Tensor):
        if (isinstance(actual, torch.Tensor)) and (isinstance(pred, torch.Tensor)):
            return self.loss(actual, pred)

        else:
            raise TypeError("Both actual and pred must be torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the Adeversarial Loss for CCGAN".capitalize()
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="mean",
        help="Define the reduction method".capitalize(),
    )

    args = parser.parse_args()

    loss = AdversarialLoss(reduction=args.reduction)

    actual = torch.tensor([1.0, 0.0, 0.0, 0.0])
    predicted = torch.tensor([1.0, 0.0, 0.0, 1.0])

    ((actual - predicted) ** 2).mean()

    assert loss(actual, predicted) == ((actual - predicted) ** 2).mean()
