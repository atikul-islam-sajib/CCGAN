import os
import cv2
import zipfile
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import config, dump, load, validate_path, CustomException


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
        self.lr_image: list = []

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

        for index, image in enumerate(os.listdir(self.X)):
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
                self.lr_image.append(self.lr_imageX)

        assert len(self.independent) == len(self.dependent) == len(self.lr_image)


if __name__ == "__main__":
    loader = Loader(image_path="data/raw/dataset.zip")
    # try:
    #     loader.unzip_folder()
    # except CustomException as e:
    #     print(e)
    # except Exception as e:
    #     print(e)

    loader.feature_extractor()
