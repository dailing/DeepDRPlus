# A deep learning system for predicting personalized time to progression of diabetic retinopathy

## Contents

1. Requirements
2. Environment setup
    * Linux System
    * Docker
3. Preparing the dataset
4. Train and Testing

## Requirements

This software requires a **Linux** system: [**Ubuntu 22.04**](https://ubuntu.com/download/desktop) or  [**Ubuntu 20.04**](https://ubuntu.com/download/desktop) (other versions are not tested)   and  [**Python3.9**](https://www.python.org) (other versions are not supported). This software requires **16GB memory** and **20GB disk** storage (we recommend 32GB memory). The software analyzes a single image in **5 seconds** on **Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz**. Some Linux** packages** are required to run this software as listed below:

```
build-essential
zlib1g-dev
libncurses5-dev
libgdbm-dev
libnss3-dev
libssl-dev
libreadline-dev
libffi-dev
libsqlite3-dev
libsm6
libxrender1
wget
git
```

The **Python packages** needed are listed below. They can also be found in `reqirements.txt`.

```
numpy>=1.22.3
seaborn>=0.11.2
sklearn>=0.0
scikit-learn>=1.1.1
torchvision>=0.13.0
pandas>=1.4.2
albumentations
torch
pymongo
tqdm
pingouin
```

## Native Environment Setup

### Linux System

#### Step 1: download the project
1. Open the terminal in the system, or press Ctrl+Alt+F1 to switch to the command line window.
1. Clone this repo file to the home path.

```
git clone https://github.com/drpredict/DeepDR_Plus.git
```

3. change the current directory to the source directory

```
cd DeepDR_Plus
```

#### Step 2: prepare the running environment and run the code***

1. install dependent Python packages

```
python3 -m pip install --user -r requirements.txt
```

**Supported Image File Format**

JPG, PNG, and TIF formats are supported and tested. Other formats that OpenCV supports should work too. The input image must be 3-channel color fundus images with a small edge resolution larger than 448.


## preparing the dataset

Training data should be put in a CSV file, containing the following columns:
* image: the path of fundus images
* t1: time of the last exam before the interested event
* t2: time of the first exam after the interested event
* e: censored or not, True for not censored(event observed), False for censored(no event observed)

## Train and Testing
For easy use of the code, we provided a simple command-line tool in `train.py`.

All options are listed in the help documents. Refer to `trainer.py` for more details. The following instructions can be used to train models:

1. The following command can be used to train the Fundus model on your data: `python train.py`
1. The hyper-parameters are set with environment variables.
    * load_pretrain: the pre-trained model path to be loaded for fine-tuning.
    * "batch_size": the training batch size
    * "epochs": the number of training epochs
    * "image_size": the input image resolution
    * "lr": the learning rate
    * "device": the device to be used for training
    * "num_workers": the number of workers for data loading
    * "model": the model name, ResNet-18 and ResNet-50 are supported
