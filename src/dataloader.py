import os
import cv2
import sys
import zipfile
import argparse
import traceback
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append("./")

from utils import (
    config,
    dump,
    load,
    validate_path,
    CustomException,
)


class Loader:
    def __init__(
        self,
        image_path=None,
        image_size: int = 128,
        channels: int = 3,
        batch_size: int = 8,
        split_size: float = 0.20,
    ):
        self.image_path = image_path
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.split_size = split_size

        self.CONFIG = config()

        self.independent: list = []
        self.dependent: list = []
        self.lr_independent: list = []
        self.lr_dependent: list = []

    def unzip_folder(self):
        if validate_path(path=self.CONFIG["path"]["RAW_IMAGE_DATA_PATH"]):
            with zipfile.ZipFile(file=self.image_path, mode="r") as zip_file:
                zip_file.extractall(path=self.CONFIG["path"]["RAW_IMAGE_DATA_PATH"])

        else:
            raise CustomException("Raw data path cannot be found".capitalize())

    def transforms(self, type="lr"):
        if type == "hr":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.CenterCrop((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        elif type == "lr":
            return transforms.Compose(
                [
                    transforms.Resize((self.image_size // 4, self.image_size // 4)),
                    transforms.CenterCrop((self.image_size // 4, self.image_size // 4)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def split_dataset(self, X: list, y: list):
        if isinstance(X, list) and isinstance(y, list):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.split_size
            )

            return {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
        else:
            raise CustomException("X and y should be list".capitalize())

    def feature_extractor(self):
        self.directory = os.path.join(
            self.CONFIG["path"]["RAW_IMAGE_DATA_PATH"], "dataset"
        )
        assert (
            self.directory.split("/")[-1] == "dataset"
        ), "Directory name should be dataset"

        self.X = os.path.join(self.directory, "X")
        self.y = os.path.join(self.directory, "y")

        print(self.X, self.y)

        assert (
            self.X.split("/")[-1] == "X" and self.y.split("/")[-1] == "y"
        ), "Directory name should be X and y"

        for _, image in tqdm(enumerate(os.listdir(self.X))):
            if image in os.listdir(self.y):
                self.imageX = os.path.join(self.X, image)
                self.imageY = os.path.join(self.y, image)

                self.imageX = cv2.imread(self.imageX)
                self.imageY = cv2.imread(self.imageY)

                self.imageX = cv2.cvtColor(self.imageX, cv2.COLOR_BGR2RGB)
                self.imageY = cv2.cvtColor(self.imageY, cv2.COLOR_BGR2RGB)

                self.imageX = Image.fromarray(self.imageX)
                self.imageY = Image.fromarray(self.imageY)

                self._imageX = self.transforms(type="hr")(self.imageX)
                self._imageY = self.transforms(type="hr")(self.imageY)

                self.lr_imageX = self.transforms(type="lr")(self.imageX)
                self.lr_imageY = self.transforms(type="lr")(self.imageY)

                self.independent.append(self._imageX)
                self.dependent.append(self._imageY)

                self.lr_independent.append(self.lr_imageX)
                self.lr_dependent.append(self.lr_imageY)

        assert (
            len(self.independent)
            == len(self.dependent)
            == len(self.lr_dependent)
            == len(self.lr_dependent)
        ), "Length of independent and dependent should be equal"

        try:
            dataset = self.split_dataset(X=self.independent, y=self.dependent)
            lr_dataset = self.split_dataset(X=self.lr_independent, y=self.lr_dependent)
        except CustomException as e:
            print("An error occurred: ", e)
            traceback.print_exc()
        except Exception as e:
            print("An error occurred: ", e)
            traceback.print_exc()
        else:
            return dataset, lr_dataset

    def create_dataloader(self):
        dataset, lr_dataset = self.feature_extractor()

        train_dataloader = DataLoader(
            dataset=list(
                zip(dataset["X_train"], dataset["y_train"], lr_dataset["X_train"])
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            dataset=list(
                zip(dataset["X_test"], dataset["y_test"], lr_dataset["X_test"])
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        if validate_path(path=self.CONFIG["path"]["PROCESSED_IMAGE_DATA_PATH"]):
            for value, filename in [
                (train_dataloader, "train_dataloader.pkl"),
                (valid_dataloader, "valid_dataloader.pkl"),
            ]:
                dump(
                    value=value,
                    filename=os.path.join(
                        self.CONFIG["path"]["PROCESSED_IMAGE_DATA_PATH"], filename
                    ),
                )

            print(
                "Train and valid dataloader is saved in the folder {}".format(
                    self.CONFIG["path"]["PROCESSED_IMAGE_DATA_PATH"]
                )
            )
        else:
            raise CustomException(
                "Cannot be created the dataloader and processed path is not found".capitalize()
            )

    @staticmethod
    def plot_images():
        processed_data_path = config()["path"]["PROCESSED_IMAGE_DATA_PATH"]
        if validate_path(path=processed_data_path):
            train_dataloader = load(
                filename=os.path.join(processed_data_path, "train_dataloader.pkl")
            )

            X, y, _ = next(iter(train_dataloader))

            num_of_rows = X.size(0) // 2
            num_of_columns = X.size(0) // num_of_rows

            plt.figure(figsize=(20, 10))

            plt.suptitle("ImageX and ImageY for CCGAN".capitalize())
            plt.axis("off")

            for index, image in enumerate(X):
                imageX = image.permute(1, 2, 0).detach().numpy()
                imageY = image.permute(1, 2, 0).detach().numpy()

                imageX = (imageX - imageX.min()) / (imageX.max() - imageX.min())
                imageY = (imageY - imageY.min()) / (imageY.max() - imageY.min())

                plt.subplot(2 * num_of_rows, 2 * num_of_columns, 2 * index + 1)
                plt.imshow(imageX)
                plt.title("X")
                plt.axis("off")

                plt.subplot(2 * num_of_rows, 2 * num_of_columns, 2 * index + 2)
                plt.imshow(imageY)
                plt.title("Y")
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(config()["path"]["FILES_PATH"], "image.jpeg"))
            plt.show()

            print(
                "Image is saved in the folder {}".format(config()["path"]["FILES_PATH"])
            )

        else:
            raise CustomException(
                "Cannot be imported processed path as it is not found".capitalize()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader for the CCGAN".title())
    parser.add_argument(
        "--image_path",
        type=str,
        default=config()["dataloader"]["image_path"],
        help="Batch size for the dataloader".capitalize(),
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=config()["dataloader"]["channels"],
        help="Number of channels".capitalize(),
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=config()["dataloader"]["image_size"],
        help="Image size".capitalize(),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config()["dataloader"]["batch_size"],
        help="Batch size".capitalize(),
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=config()["dataloader"]["split_size"],
        help="Split ratio".capitalize(),
    )
    args = parser.parse_args()

    loader = Loader(
        image_path=args.image_path,
        channels=args.channels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        split_size=args.split_size,
    )
    # try:
    #     loader.unzip_folder()
    # except CustomException as e:
    #     print(e)
    # except Exception as e:
    #     print(e)

    try:
        loader.create_dataloader()
    except CustomException as e:
        print("An error occurred: ", e)
        traceback.print_exc()
    except Exception as e:
        print("An error occurred: ", e)
        traceback.print_exc()

    try:
        Loader.plot_images()
    except CustomException as e:
        print("An error occurred: ", e)
        traceback.print_exc()
    except Exception as e:
        print("An error occurred: ", e)
        traceback.print_exc()
