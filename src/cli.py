import sys
import argparse
import traceback

sys.path.append("./")

from dataloader import Loader
from generator import Generator
from utils import config, CustomException

def cli():
    parser = argparse.ArgumentParser(description="CLI for the CCGAN - train & test".title())
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
    parser.add_argument(
        "--mode",
        type=str,
        default="train", # To test use "test"
        help="""To train and test the model: to train set -> "train" and for test set -> "test" """.capitalize(),
    )
    parser.add_argument(
        "--database",
        type=bool,
        default=config()["dataloader"]["mongoDB"],
        help="Database".capitalize(),
    )
    args = parser.parse_args()
    
    if args.mode == "train":

        loader = Loader(
            image_path=args.image_path,
            channels=args.channels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )

        try:
            loader.unzip_folder()
        except CustomException as e:
            print("An error occurred: ", e)
            traceback.print_exc()
        except Exception as e:
            print("An error occurred: ", e)
            traceback.print_exc()

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

        try:
            Loader.dataset_details()
        except Exception as e:
            print("An error occurred: ", e)
            traceback.print_exc()

        if args.database:
            try:
                loader.store_database()
            except CustomException as e:
                print("An error occurred: ", e)
                traceback.print_exc()
            except Exception as e:
                print("An error occurred: ", e)
                traceback.print_exc()
                
    elif args.mode == "test":
        pass

if __name__ == "__main__":
    cli()
