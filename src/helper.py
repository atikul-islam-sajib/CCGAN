import os
import sys
import torch

sys.path.append("./")

from utils import config, validate_path, load, CustomException


def load_dataloader():
    if validate_path(path=config()["path"]["PROCESSED_IMAGE_DATA_PATH"]):
        dataloader_path = config()["path"]["PROCESSED_IMAGE_DATA_PATH"]

        train_dataloader = os.path.join(dataloader_path, "train_dataloader.pkl")
        valid_dataloader = os.path.join(dataloader_path, "valid_dataloader.pkl")

        return {
            "train_dataloader": load(filename=train_dataloader),
            "valid_dataloader": load(filename=valid_dataloader),
        }

    else:
        raise CustomException(
            "dataloader cannot be retrieved from the helper function".capitalize()
        )


def helpers(**kwargs):
    lr = kwargs["lr"]
    adam = kwargs["adam"]
    SGD = kwargs["SGD"]
    beta1 = kwargs["beta1"]
    beta2 = kwargs["beta2"]
    momentum = kwargs["momentum"]


if __name__ == "__main__":
    dataloader = load_dataloader()

    assert (
        dataloader["train_dataloader"].__class__
        == torch.utils.data.dataloader.DataLoader
    )
