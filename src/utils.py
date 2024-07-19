import os
import yaml
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
