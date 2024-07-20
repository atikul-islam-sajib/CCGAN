# CCGAN
CCGAN (Context Conditional GAN) is a deep learning model designed to generate high-quality images based on specific contextual information. It leverages a conditional Generative Adversarial Network (GAN) framework, where both the generator and discriminator are conditioned on additional context data to improve the realism and relevance of the generated images. CCGAN is particularly useful in scenarios where generating contextually accurate images is crucial, such as in image-to-image translation, super-resolution, and other image synthesis tasks.

<img src="https://github.com/atikul-islam-sajib/Research-Assistant-Work-/blob/main/5-Figure2-1.png" alt="CCGAN Architecture" height="300" width="1200">



## Getting Started

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                       |
| ---- | -------------------------------------------- | ------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/CCGAN.git** |
| 2    | Navigate into the project directory.         | **cd CCGAN**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                           |

## Project Structure
```
.
├── Dockerfile
├── LICENSE
├── README.md
├── artifacts/
│   ├── checkpoints/
│   ├── files/
│   ├── metrics/
│   └── outputs/
├── config.yml
├── data/
│   ├── processed/
│   └── raw/
├── dvc.lock
├── dvc.yaml
├── logs/
├── mlruns/
├── mypy.ini
├── requirements.txt
├── research/
│   ├── files/
│   └── notebooks/
├── setup.py
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── dataloader.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── helper.py
│   ├── loss.py
│   ├── tester.py
│   ├── trainer.py
│   └── utils.py
└── unittest/
    └── test.py
...  ...
...  ...
```

### Dataset Organization for CCGAN

The dataset is organized into three categories for CCGAN. Each category directly contains paired images and their corresponding images, stored together to simplify the association between images.

## Directory Structure:

```
dataset/  # Folder name must be 'dataset'
├── X/  # Contains the inpaint data
│   ├── 1.png
│   ├── 2.png
│   ├── ...
├── y/  # Contains the clean data
│   ├── 1.png
│   ├── 2.png
│   ├── ...
```

### Important Notes:
- The folder name must be `dataset`.
- `X` folder will contain the inpaint data.
- `y` folder will contain the clean data.
- Each image in `X` must be paired with a corresponding image in `y` (e.g., `X/1.png` pairs with `y/1.png`).

### User Guide Notebook - CLI

UPLOAD LATER(MAY BE TOMORROW.......)
<!-- For detailed documentation on the implementation and usage, visit the -> [VAE Notebook - CLI](https://github.com/atikul-islam-sajib/VAE-Pytorch/blob/main/research/notebooks/ModelTrain_CLI.ipynb). -->

### Command Line Interface

The project is controlled via a command line interface (CLI) which allows for running different operational modes such as training, testing, and inference.

#### CLI Arguments
| Argument          | Description                                  | Type   | Default |
|-------------------|----------------------------------------------|--------|---------|
| `--image_path`    | Path to the image dataset                    | str    | None    |
| `--batch_size`    | Number of images per batch                   | int    | 1       |
| `--image_size`    | Size to resize images to                     | int    | 64      |
| `--epochs`        | Number of training epochs                    | int    | 100     |
| `--lr`            | Learning rate                                | float  | 0.0002  |
| `--lr_scheduler`| Enable learning rate scheduler              | bool   | False   |
| `--device`        | Computation device ('cuda', 'mps', 'cpu')    | str    | 'mps'   |
| `--adam`          | Use Adam optimizer                           | bool   | True    |
| `--SGD`           | Use Stochastic Gradient Descent optimizer    | bool   | False   |
| `--beta1`         | Beta1 parameter for Adam optimizer           | float  | 0.5     |
| `--train`         | Flag to initiate training mode               | action | N/A     |
| `--model`         | Path to a saved model for testing            | str    | None    |
| `--test`          | Flag to initiate testing mode                | action | N/A     |

### CLI Command Examples

| Task                     | CUDA Command                                                                                                              | CPU Command                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | `python cli.py --mode train --image_path "/path/to/dataset.zip" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --device "cuda"` | `python cli.py --mode train --image_path "/path/to/dataset.zip" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --device "cpu"` |
| **Testing a Model**      | `python cli.py --mode test --model "/path/to/saved_model.pth" --device "cuda"`                                        | `python cli.py --mode test --model "/path/to/saved_model.pth" --device "cpu"`                                        |

All configuration arguments are defined in `config.yml`. If the user does not provide these arguments, default values from the configuration file will be used.

**Configure the Project**:
   Update the `config.yml` file with the appropriate paths and settings. An example `config.yml`:
   ```yaml

        path:
            RAW_IMAGE_DATA_PATH: "./data/raw/"                        # Path to raw image data
            PROCESSED_IMAGE_DATA_PATH: "./data/processed/"            # Path to processed image data
            FILES_PATH: "./artifacts/files/"                          # Path to artifact files
            TRAIN_MODELS: "./artifacts/checkpoints/train_models/"     # Path to saved training models
            BEST_MODEL: "./artifacts/checkpoints/best_model/"         # Path to the best model checkpoint
            METRICS_PATH: "./artifacts/metrics/"                      # Path to saved metrics
            TRAIN_OUTPUT_IMAGES: "./artifacts/outputs/train_images/"  # Path to output images from training
            TEST_OUTPUT_IMAGES: "./artifacts/outputs/test_images/"    # Path to output images from testing



        dataloader:
            image_path: "./data/raw/dataset.zip"                      # Path to the dataset
            channels: 3                                               # Number of image channels (e.g., RGB), it only works for RGB image
            image_size: 128                                           # Possible sizes to which images will be resized (128, 256, 512)
            batch_size: 1                                             # Number of images per batch
            split_size: 0.20                                          # Proportion of the dataset to use for validation
            mongoDB: False                                            # Use MongoDB for dataset storage


        database:
            USERNAME: None             	                              # Set Your USERNAME
            PASSWORD: None                                            # Set Your PASSWORD
            CLUSTER_URL: None                                         # Set Your Cluster e.g. "cluster0.ym14neq.mongodb.net/?appName=Cluster0"

        MLFlow:
            MLFLOW_TRACKING_URL: "https://dagshub.com/atikul-islam-sajib/CCGAN.mlflow"  # Set your own MLFlow Tracking URL
            MLFLOW_TRACKING_USERNAME: "atikul-islam-sajib"                              # Set your own MLFlow Tracking Username
            MLFLOW_TRACKING_PASSWORD: "*********"                                       # Set your own MLFlow Tracking Password
            REPO_NAME: "CCGAN"

        trainer:
            epochs: 2000                      # Number of training epochs
            lr: 0.001                         # Learning rate
            beta1: 0.5                        # Beta1 hyperparameter for Adam optimizer
            beta2: 0.999                      # Beta2 hyperparameter for Adam optimizer
            momentum: 0.90                    # Momentum for optimizers
            weight_decay: 0.001               # Weight decay for regularization
            step_size: 100                    # Step size for learning rate scheduler
            gamma: 0.85                       # Multiplicative factor for learning rate decay
            threshold: 100                    # Threshold for model saving purposes
            device: "cpu"                     # Device to use for training (e.g., MPS, CPU, GPU)
            adam: True                        # Use Adam optimizer
            SGD: False                        # Use SGD optimizer
            pixelLoss: False                  # Use pixel-wise loss
            l1_regularization: False          # Use L1 regularization
            l2_regularization: False          # Use L2 regularization
            elasticnet_regularization: False  # Use Elastic Net regularization
            lr_scheduler: True                # Enable learning rate scheduler
            verbose: True                     # Enable verbose logging
            mlflow: True                      # Enable MLflow tracking


        tester:
            model: "best"                     # Model to use for testing (e.g., best model or "./checkpoints/train_models/netG10.pth")
            dataloader: "valid"               # DataLoader to use for validation/testing e.g "train" or "valid"
            device: "cpu"                     # Device to use for testing (e.g., CPU, GPU)

```

#### Initializing Data Loader - Custom Modules
```python
loader = Loader(image_path="path/to/dataset", batch_size=32, image_size=128)
loader.unzip_folder()
loader.create_dataloader()
```

##### To details about dataset
```python
loader.display_images()        # It will display the images from dataset
```

#### Training the Model
```python
trainer = Trainer(
    epochs=100,                # Number of epochs to train the model
    lr=0.0002,                 # Learning rate for optimizer
    device='cuda',             # Computation device ('cuda', 'mps', 'cpu')
    adam=True,                 # Use Adam optimizer; set to False to use SGD if implemented
    SGD=False,                 # Use Stochastic Gradient Descent optimizer; typically False if Adam is True
    beta1=0.5,                 # Beta1 parameter for Adam optimizer
    lr_scheduler=False,        # Enable a learning rate scheduler
    weight_init=False,         # Enable custom weight initialization for the models
    display=True               # Display training progress and statistics

    ... ... ... 
    ... ... ...                # Check the trainer.py
)

# Start training
trainer.train()
```

#### Testing the Model
```python
tester = Tester(
    device="cuda",            # Specify the device to test the model
    model="best"              # Define the model, see the checkpoints->train_models if used
)
test.test()
```


### Configuration for MLFlow

1. **Generate a Personal Access Token on DagsHub**:
   - Log in to [DagsHub](https://dagshub.com).
   - Go to your user settings and generate a new personal access token under "Personal Access Tokens".


2. **Configuration in config.yml**:
   Ensure the MLFlow configuration is defined in the `config.yml` file. The relevant section might look like this:

   ```yaml
   MLFlow:
     MLFLOW_TRACKING_URL: "https://dagshub.com/<username>/<repo_name>.mlflow"
     MLFLOW_TRACKING_USERNAME: "<your_dagshub_username>"
     MLFLOW_TRACKING_PASSWORD: "<your_dagshub_token>"
   ```

   Make sure to replace `<username>`, `<repo_name>`, `<your_dagshub_username>`, and `<your_dagshub_token>` with your actual DagsHub credentials.

### Running the Training Script

To start training and logging the experiments to DagsHub, run the following command:

```bash
python src/cli.py --mode train 
python src/cli.py --mode test 

```

### Accessing Experiment Tracking

You can access the MLflow experiment tracking UI hosted on DagsHub using the following link:

[VAE Experiment Tracking on DagsHub](https://dagshub.com/atikul-islam-sajib/CCGAN)

### Using MLflow UI Locally

If you prefer to run the MLflow UI locally, use the following command:

```bash
mlflow ui
```


## Contributing
Contributions to improve this implementation of VAE are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).

