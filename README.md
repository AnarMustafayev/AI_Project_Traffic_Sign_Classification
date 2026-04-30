# Traffic Sign Classification

A deep learning project for classifying traffic signs using PyTorch, featuring multiple model architectures, a training pipeline with TensorBoard logging, and a Streamlit web application for real-time inference.

## Overview

This project trains convolutional neural networks to classify traffic signs. It supports four model configurations:

| Model | Type | Description |
|-------|------|-------------|
| ResNet18 | Pretrained | ImageNet-pretrained ResNet18 with frozen backbone, fine-tuned classifier |
| ResNet18 | Simple | ResNet18 trained from scratch |
| VGG16 | Pretrained | ImageNet-pretrained VGG16 with frozen feature layers, fine-tuned classifier |
| VGG16 | Simple | VGG16 trained from scratch |

## Project Structure

```
├── app.py                          # Streamlit inference web app
├── main.py                         # Training entry point
├── analytics.ipynb                 # Analysis and visualization notebook
├── dataset/
│   ├── labels.csv                  # Class ID to label name mapping
│   └── splitting_and_augmentation.ipynb  # Dataset preparation notebook
└── src/
    ├── dataset.py                  # DataLoader creation
    ├── models.py                   # Model definitions (ResNet18, VGG16)
    ├── train.py                    # Training and validation loops
    ├── test.py                     # Test evaluation
    ├── utils.py                    # Model saving utility
    ├── test_for_resnet_pretrained.py
    └── test_for_vgg16.py
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- torchmetrics
- scikit-learn
- tqdm
- tensorboard
- streamlit
- Pillow
- pandas

Install dependencies:

```bash
pip install torch torchvision torchmetrics scikit-learn tqdm tensorboard streamlit Pillow pandas
```

## Dataset

The project expects the dataset to be organized into three splits with one subdirectory per class:

```
balanced_data/
├── train/
│   ├── <class_0>/
│   ├── <class_1>/
│   └── ...
└── val/
    ├── <class_0>/
    ├── <class_1>/
    └── ...
dataset/
└── TEST/
    ├── <class_0>/
    ├── <class_1>/
    └── ...
```

Use `dataset/splitting_and_augmentation.ipynb` to prepare and augment the raw dataset into the required structure.

Images are resized to **128×128** and normalized with mean=0.5 and std=0.5.

## Training

Edit the directory paths and hyperparameters in `main.py`, then run:

```bash
python main.py
```

Key hyperparameters (configured in `main.py`):

| Parameter | Default |
|-----------|---------|
| Batch size | 32 |
| Epochs | 15 |
| Input size | 128×128 |
| Optimizer | Adam (lr=0.0001) or SGD (lr=0.0005, momentum=0.9) |

Model checkpoints are saved to `outputs/models/` after each epoch. TensorBoard logs are written to `outputs/logs/`.

To monitor training:

```bash
tensorboard --logdir outputs/logs
```

Metrics logged per epoch: **Loss**, **Accuracy**, and **F1 Score** (training and validation).

## Inference — Streamlit App

The web app loads a trained model and classifies uploaded traffic sign images.

1. Update `MODEL_PATH` and `TEST_DATA_PATH` in `app.py` to point to your checkpoint and test dataset.
2. Run the app:

```bash
streamlit run app.py
```

3. Upload a `.jpg`, `.jpeg`, or `.png` image. The app will display the image and the predicted traffic sign class.

## Evaluation

The `src/test.py` module evaluates a trained model on the test set and reports:

- **Accuracy**
- **F1 Score (Macro)**

This is called automatically at the end of training in `main.py`, or can be invoked separately.

## Model Architecture Details

**ResNet18 (Pretrained)**  
All backbone layers are frozen. Only the final fully connected layer is replaced and trained for the number of traffic sign classes.

**VGG16 (Pretrained)**  
The `features` layers are frozen. The final layer of the `classifier` head is replaced and trained for the number of classes.

**ResNet18 / VGG16 (Simple)**  
Full networks trained from scratch with a replaced output layer matching the number of classes.
