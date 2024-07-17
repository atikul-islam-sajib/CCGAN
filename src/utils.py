import os
import yaml
import joblib
class CustomException(Exception):
    def __init__(self, message: str):
        self.message = message
        
def dump(value = None, filename = None):
    if (value is not None) and (filename is not None):
        joblib.dump(value=value, filename=filename)
        
    else:
        CustomException("Cannot be possble to dump the value".capitalize())
        
def load(filename:str):
    if isinstance(filename, str):
        return joblib.load(filename=filename)
    
    else:
        CustomException("Cannot be possble to load the value".capitalize())
        
def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
    
def validate_path(path:str):
    if isinstance(path, str):
        if os.path.exists(path):
            return True
        else:
            return False
        
    else:
        CustomException("Cannot be possble to validate the path".capitalize())