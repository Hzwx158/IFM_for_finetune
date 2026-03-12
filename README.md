# IFM: Iterative Feature Merging for Parameter-Efficient Fine-Tuning

This code is based on [Adaptformer](https://github.com/ShoufaChen/AdaptFormer)

## 🛠 Quick Start
### 1. Prepare Dataset

You can use either `bash prepare.sh dataset` or `python prepare_dataset.py` command to prepare datasets used for training.

The result will be saved in `./data/xxx`.

### 2. Prepare Pretrained Models

- Choose the models you need in `prepare.sh`
- Download models by `bash prepare.sh download`
- Convert and Merge model weights by `bash prepare.sh train`

### 3. Train model

We create a shell file [run.sh](./run.sh) for training.
- Choose the models and datasets you want for training by setting `MODELS` and `DATASETS`
- Check the end of shell file. Pick the ways you need for training
    + Example 1: `bash run.sh cnn`
    + Example 2: `bash run.sh lora`
- Notice that ResNet models should only be finetuned by `bash run.sh cnn` and ViT models should only be finetuned by commands except `bash run.sh cnn`.

## 🏗 Method

## 📝Citation
...
