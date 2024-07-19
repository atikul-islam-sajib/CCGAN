import os
import sys
import torch
import traceback
import torch.optim as optim

sys.path.append("./")

from generator import Generator
from loss import AdversarialLoss
from discriminator import Discriminator
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

    batch_size = config()["dataloader"]["batch_size"]
    channels = config()["dataloader"]["channels"]
    image_height = config()["dataloader"]["image_size"]
    image_width = config()["dataloader"]["image_size"]

    netG = Generator(image_size=(batch_size, channels, image_height, image_width))
    netD = Discriminator(image_size=(batch_size, channels, image_height, image_width))

    if adam:
        optimizerG = optim.Adam(params=netG.parameters(), lr=lr, betas=(beta1, beta2))
        optimizerD = optim.Adam(params=netD.parameters(), lr=lr, betas=(beta1, beta2))

    elif SGD:
        optimizerG = optim.SGD(params=netG.parameters(), lr=lr, momentum=momentum)
        optimizerD = optim.SGD(params=netD.parameters(), lr=lr, momentum=momentum)

    try:
        dataloader = load_dataloader()

    except CustomException as e:
        print("An error occurred: ", e)
        traceback.print_exc()
    except Exception as e:
        print("An error occurred: ", e)
        traceback.print_exc()

    try:
        adversarial_loss = AdversarialLoss(reduction="mean")
    except TypeError as e:
        print("An error occurred: ", e)
        traceback.print_exc()
    except Exception as e:
        print("An error occurred: ", e)
        traceback.print_exc()

    return {
        "train_dataloader": dataloader["train_dataloader"],
        "valid_dataloader": dataloader["valid_dataloader"],
        "netG": netG,
        "netD": netD,
        "optimizerG": optimizerG,
        "optimizerD": optimizerD,
        "loss": adversarial_loss,
    }


if __name__ == "__main__":
    init = helpers(lr=2e-4, adam=True, SGD=True, momentum=0.9, beta1=0.5, beta2=0.999)

    assert init["train_dataloader"].__class__ == torch.utils.data.DataLoader
    assert init["valid_dataloader"].__class__ == torch.utils.data.DataLoader

    assert init["netG"].__class__ == Generator
    assert init["netD"].__class__ == Discriminator

    assert init["optimizerG"].__class__ == torch.optim.Adam
    assert init["optimizerD"].__class__ == torch.optim.Adam

    assert init["loss"].__class__ == AdversarialLoss

    print(init["netD"](torch.randn(1, 3, 128, 128)).size())
    image = torch.randn(1, 3, 128, 128)
    lr = torch.randn(1, 3, 32, 32)
    print(init["netG"](image, lr).size())
