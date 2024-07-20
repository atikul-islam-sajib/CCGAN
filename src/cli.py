import sys
import argparse
import traceback

sys.path.append("./")

from trainer import Trainer
from dataloader import Loader
from generator import Generator
from utils import config, CustomException


def cli():
    parser = argparse.ArgumentParser(
        description="CLI for the CCGAN - train & test".title()
    )
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
        default="train",  # To test use "test"
        help="""To train and test the model: to train set -> "train" and for test set -> "test" """.capitalize(),
    )
    parser.add_argument(
        "--database",
        type=bool,
        default=config()["dataloader"]["mongoDB"],
        help="Database".capitalize(),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config()["trainer"]["epochs"],
        help="Number of epochs to train the model".capitalize(),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config()["trainer"]["lr"],
        help="Learning rate for the model".capitalize(),
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=config()["trainer"]["beta1"],
        help="Beta1 for Adam".capitalize(),
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=config()["trainer"]["beta2"],
        help="Beta2 for Adam".capitalize(),
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=config()["trainer"]["momentum"],
        help="Momentum for SGD".capitalize(),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config()["trainer"]["weight_decay"],
        help="Weight decay for the model".capitalize(),
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=config()["trainer"]["step_size"],
        help="Step size for the scheduler".capitalize(),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=config()["trainer"]["gamma"],
        help="Gamma for the scheduler".capitalize(),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config()["trainer"]["device"],
        help="Device to train the model on".capitalize(),
    )
    parser.add_argument(
        "--adam",
        type=bool,
        default=config()["trainer"]["adam"],
        help="Use Adam optimizer".capitalize(),
    )
    parser.add_argument(
        "--SGD",
        type=bool,
        default=config()["trainer"]["SGD"],
        help="Use SGD optimizer".capitalize(),
    )
    parser.add_argument(
        "--pixelLoss",
        type=bool,
        default=config()["trainer"]["pixelLoss"],
        help="Use pixel loss".capitalize(),
    )
    parser.add_argument(
        "--l1_regularization",
        type=bool,
        default=config()["trainer"]["l1_regularization"],
        help="Use L1 loss".capitalize(),
    )
    parser.add_argument(
        "--l2_regularization",
        type=bool,
        default=config()["trainer"]["l2_regularization"],
        help="Use L2 loss".capitalize(),
    )
    parser.add_argument(
        "--elasticnet_regularization",
        type=bool,
        default=config()["trainer"]["elasticnet_regularization"],
        help="Use elastic loss".capitalize(),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=config()["trainer"]["lr_scheduler"],
        help="Use lr scheduler".capitalize(),
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=config()["trainer"]["verbose"],
        help="Verbose".capitalize(),
    )
    parser.add_argument(
        "--mlflow",
        type=bool,
        default=config()["trainer"]["mlflow"],
        help="Use mlflow".capitalize(),
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

        try:
            trainer = Trainer(
                epochs=args.epochs,
                lr=args.lr,
                beta1=args.beta1,
                beta2=args.beta2,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                step_size=args.step_size,
                gamma=args.gamma,
                device=args.device,
                adam=args.adam,
                SGD=args.SGD,
                pixelLoss=args.pixelLoss,
                l1_regularization=args.l1_regularization,
                l2_regularization=args.l2_regularization,
                elasticnet_regularization=args.elasticnet_regularization,
                lr_scheduler=args.lr_scheduler,
                verbose=args.verbose,
                mlflow=args.mlflow,
            )
        except Exception as e:
            print("An error occurred: ", e)
            traceback.print_exc()

        else:
            trainer.train()

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
