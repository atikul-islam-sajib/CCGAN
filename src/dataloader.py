import zipfile


class Loader:
    def __init__(self, image_size: int = 128, channels: int = 3, batch_size: int = 8, split_size: float = 0.20):
        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.split_size = split_size
        
    def unzip_folder(self):
        pass
        