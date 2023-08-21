# A deep learning system for predicting personalized time to progression of diabetic retinopathy

## Contents

1. Requirements
2. Environment setup
    * Linux System
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
### Pretrain the network using Moco-V2
We adopted a open-source implementation of [MoCo-v2](https://github.com/facebookresearch/moco-v2) for pre-training.
To pre-train the network, enter the `MoCo-v2` directory and run the following command:
`python main_moco.py`. Optionally, you may need to change configuration parameters stored in `config.py`.
The trained model will be saved in `MoCo-v2/models/resnet50_bs32_queue16384_wd0.0001_t0.2_cos)` directory. We choose the model with the least eval loss as the pretrained model.

### Training the prediction model
For easy use of the code, we provided a simple command-line tool in `train.py`.

All options are listed in the help documents. Refer to `trainer.py` for more details. The following instructions can be used to train models:

1. The hyper-parameters are set with environment variables.
    * load_pretrain: the pre-trained model path to be loaded for fine-tuning.
    * "batch_size": the training batch size
    * "epochs": the number of training epochs
    * "image_size": the input image resolution
    * "lr": the learning rate
    * "device": the device to be used for training
    * "num_workers": the number of workers for data loading
    * "model": the model name, ResNet-18 and ResNet-50 are supported
1. Training fundus model:
    * run `python train_eval_fund.py`, with proper hyper-parameters settings.
    * the evaluation results are saved in `logs/` in a pickle dump. see trainer.py for more details.
    * to run with pretrained model, invoke `load_pretrain=MoCo-v2/models/resnet50_bs32_queue16384_wd0.0001_t0.2_cos/checkpoint_resnet50_bs32_queue16384_wd0.0001_t0.2_cos_epoch-60.pth python train_eval_fund.py `, change the model dump path as needed.
1. Training metadata model or combined model:
    * to train the model, first prepare the dataset, in CSV format containing normalized features as well as event information.
    * the feature names to use for training are provided with command-line arguments.
    * e.g. run `python train_eval_cov.py AGE GENDER HBA1C SBP DBP BMI LDL HDL T2D_dur`, with proper hyper-parameters settings to train the metadata model with AGE, GENDER, HBA1C, SBP, DBP, BMI, LDL, HDL, T2D_dur as covariables.
    * to run the combined model, extract the scores from the fundus model and add it to the CSV file, invoke `python train_eval_combined.py`. with fundus score included in the command-line arguments.
    * the evaluation results are saved in `logs/` as a pickle dump. see trainer.py for more details.
