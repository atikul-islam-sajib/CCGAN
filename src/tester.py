import os
import sys
import torch
import argparse
import traceback
import matplotlib.pyplot as plt

sys.path.append("./")

from generator import Generator
from utils import (
    config,
    device_init,
    validate_path,
    load,
)


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

    def load_dataloader(self):
        if self.dataloader == "valid":
            processed_path = config()["path"]["PROCESSED_IMAGE_DATA_PATH"]

            if validate_path(path=processed_path):
                return load(
                    filename=os.path.join(processed_path, "valid_dataloader.pkl")
                )
            else:
                raise FileNotFoundError("Valid dataloader not found".capitalize())

        else:
            if validate_path(path=processed_path):
                return load(
                    filename=os.path.join(processed_path, "valid_dataloader.pkl")
                )
            else:
                raise FileNotFoundError("Train dataloader not found".capitalize())

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

    def plot_images(self, model: Generator):
        try:
            dataloader = self.load_dataloader()
        except FileNotFoundError as e:
            print(
                "An error occured while loading the dataloader {}".capitalize().format(
                    e
                )
            )
        except Exception as e:
            print(
                "An error occured while loading the dataloader {}".capitalize().format(
                    e
                )
            )
        else:
            X, y, lr_image = next(iter(dataloader))

            X = X.to(self.device)
            y = y.to(self.device)
            lr_image = lr_image.to(self.device)

            generated_image = model(X, lr_image)

            number_of_rows = X.size(0) // 2
            number_of_coumns = X.size(0) // number_of_rows

        plt.figure(figsize=(10, 10))

        for index, image in enumerate(X):
            realimage = image.permute(2, 1, 0).detach().cpu().numpy()
            generated = generated_image[index].permute(2, 1, 0).cpu().detach().numpy()

            realimage = (realimage - realimage.min()) / (
                realimage.max() - realimage.min()
            )
            generated = (generated - generated.min()) / (
                generated.max() - generated.min()
            )

            plt.subplot(2 * number_of_rows, 2 * number_of_coumns, 2 * index + 1)
            plt.imshow(realimage)
            plt.title("Real".capitalize())
            plt.axis("off")

            plt.subplot(2 * number_of_rows, 2 * number_of_coumns, 2 * index + 2)
            plt.imshow(generated)
            plt.title("Generated".capitalize())
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(config()["path"]["TEST_OUTPUT_IMAGES"]))
        plt.show()

        print(
            "Images saved to {}".format(
                os.path.join(config()["path"]["TEST_OUTPUT_IMAGES"])
            )
        )

    def test(self):
        netG = self.initialize_netG()
        try:
            netG.load_state_dict(self.select_model())

            self.plot_images(model=netG)

        except FileNotFoundError as e:
            print("An error occurred: ", e)
            traceback.print_exc()
        except Exception as e:
            print("An error occurred: ", e)
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tester code for CCGAN".capitalize())
    parser.add_argument(
        "--model",
        type=str,
        default=config()["tester"]["model"],
        help="Model to be tested".capitalize(),
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default=config()["tester"]["dataloader"],
        help="Data loader to be used".capitalize(),
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device to be used".capitalize()
    )

    args = parser.parse_args()

    tester = Tester(
        model=args.model,
        dataloader=args.dataloader,
        device=args.device,
    )

    tester.test()
