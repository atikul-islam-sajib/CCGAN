import zipfile

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

    def unzip_folder(self):
        if validate_path(path=self.CONFIG["path"]["RAW_IMAGE_DATA_PATH"]):
            with zipfile.ZipFile(file=self.image_path, mode="r") as zip_file:
                zip_file.extractall(path=self.CONFIG["path"]["RAW_IMAGE_DATA_PATH"])

        else:
            raise CustomException("Raw data path cannot be found".capitalize())


if __name__ == "__main__":
    loader = Loader(image_path="data/raw/dataset.zip")
    try:
        loader.unzip_folder()
    except CustomException as e:
        print(e)
    except Exception as e:
        print(e)
