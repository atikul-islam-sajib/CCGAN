import os
import yaml
import torch
import joblib
from pymongo.mongo_client import MongoClient


def parse_tuple(string):
    return tuple(map(int, string.strip("()").split(",")))


class CustomException(Exception):
    def __init__(self, message: str):
        self.message = message


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)

    else:
        CustomException("Cannot be possble to dump the value".capitalize())


def load(filename: str):
    if isinstance(filename, str):
        return joblib.load(filename=filename)

    else:
        CustomException("Cannot be possble to load the value".capitalize())


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)


def validate_path(path: str):
    if isinstance(path, str):
        if os.path.exists(path):
            return True
        else:
            return False

    else:
        CustomException("Cannot be possble to validate the path".capitalize())


def connect_database():
    USERNAME = config()["database"]["USERNAME"]
    PASSWORD = config()["database"]["PASSWORD"]
    if (USERNAME is not None) and (PASSWORD is not None):
        uri = f"mongodb+srv://{USERNAME}:{PASSWORD}@cluster0.ym14neq.mongodb.net/?appName=Cluster0"

    client = MongoClient(uri)

    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return True, client

    except Exception as e:
        print(e)
        return False, _


def clean_folder():
    processed_path = config()["path"]["PROCESSED_IMAGE_DATA_PATH"]
    files_path = config()["path"]["FILES_PATH"]
    train_models = config()["path"]["FILES_PATH"]
    best_model = config()["path"]["BEST_MODEL"]
    metrics_path = config()["path"]["METRICS_PATH"]
    train_output_images = config()["path"]["TRAIN_OUTPUT_IMAGES"]
    valid_output_images = config()["path"]["TEST_OUTPUT_IMAGES"]

    for path in [
        processed_path,
        files_path,
        train_models,
        best_model,
        metrics_path,
        train_output_images,
        valid_output_images,
    ]:
        if validate_path(path=path):
            for file in os.path.listdir(path):
                os.remove(os.path.join(path, file))

            print("{}: deleted all the old files".capitalize())

        else:
            raise FileNotFoundError(
                "{} path is not found for cleaning....".capitalize()
            )


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def device_init(device="cpu"):
    if device == "cpu":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


description = """
Fetch and process documents
X = []
y = []
lr = []

database = client['CCGAN']
collection = database['train_data'] # Use "valid_data" for validation data as well

for document in collection.find({}):
    if "imageX" in document.keys():
        print("Processing imageX")
        X.append(torch.tensor(document["imageX"]).view(-1, 3, 128, 128))
    elif "imageY" in document.keys():
        print("Processing imageY")
        y.append(torch.tensor(document["imageY"]).view(-1, 3, 128, 128))
    else:
        print("Processing lowerY")
        lr.append(torch.tensor(document["lowerY"]).view(-1, 3, 32, 32))

# Verify the tensors by printing their sizes
for tensor_X, tensor_y, tensor_lr in zip(X, y, lr):
    print(f"X size: {tensor_X.size()}, y size: {tensor_y.size()}, lr size: {tensor_lr.size()}")
"""
