import os
import sys
import torch
import traceback

sys.path.append("./")

from utils import config, device_init, validate_path, load
from generator import Generator


class Tester:
    def __init__(
        self, model: str = "best", dataloader: str = "valid", device: str = "cpu"
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.device = device_init(device=self.device)

    def initialize_netG(self):
        batch_size = config()["dataloader"]["batch_size"]
        channels = config()["dataloader"]["channels"]
        image_size = config()["dataloader"]["image_size"]

        self.netG = Generator(image_size=(batch_size, channels, image_size, image_size))

        self.netG.to(self.device)

        return self.netG

    def select_model(self):
        if self.model == "best":
            bestmodel = config()["path"]["BEST_MODEL"]

            if validate_path(path=bestmodel):
                bestmodel = os.path.join(bestmodel, "bestmodel.pth")
                bestmodel = torch.load(bestmodel)

                return bestmodel["netG"]
            else:
                raise FileNotFoundError("Best model not found".capitalize())
        else:
            model_name = self.model.split("/")[-1]
            if model_name in os.listdir(config()["path"]["TRAIN_MODELS"]):
                return torch.load(
                    os.path.join(config()["path"]["TRAIN_MODELS"], model_name)
                )
            else:
                raise FileNotFoundError(
                    "Model not found, that is defined by user".capitalize()
                )

    def test(self):
        netG = self.initialize_netG()
        try:
            netG.load_state_dict(self.select_model())
            print(netG)
        except FileNotFoundError as e:
            print("An error occurred: ", e)
            traceback.print_exc()
        except Exception as e:
            print("An error occurred: ", e)
            traceback.print_exc()


if __name__ == "__main__":
    tester = Tester(model="./checkpoints/train_models/netG2.pth", device="mps")
    tester.test()
