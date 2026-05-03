# Lung Cancer Detection System 🫁🔬

A complete, production-quality Deep Learning system for classifying lung nodules in CT scans as **Benign** or **Malignant**. This project utilizes **Transfer Learning** via a pre-trained **ResNet-50** architecture and includes full evaluation metrics and **Grad-CAM explainability** to visually interpret the model's predictions.

## 🌟 Key Features

- **Transfer Learning**: Built on top of PyTorch's `ResNet-50`, frozen to extract powerful ImageNet features, with a custom classification head.
- **Robust Data Pipeline**: Automated train-validation splitting (80/20) with dynamic data augmentations (Random Rotations, Horizontal Flips, Resized Cropping) to prevent overfitting.
- **Comprehensive Evaluation**: Computes Accuracy, Sensitivity (Recall), Specificity, F1-Score, and ROC-AUC. Automatically plots Confusion Matrices.
- **Explainable AI (Grad-CAM)**: Generates and overlays heatmaps on the original CT scans to highlight the exact regions the model focused on to make its prediction.
- **Clean Architecture**: Modular codebase separated into configuration, model definition, training/evaluation loops, and utilities.

## 🛠️ Installation

**Prerequisites:** Python 3.8+

1. Clone this repository:
```bash
git clone https://github.com/Champion0G/lcp.git
cd lcp
```

2. Install the required dependencies:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn opencv-python Pillow numpy
```

*(Note: If you have an NVIDIA GPU, ensure you install the CUDA-compatible version of PyTorch for significantly faster training).*

## 📂 Dataset Structure

Before running the code, you must supply a dataset. This project accepts raw PNG/JPG CT scan images.

1. Create a `dataset` folder in the root directory.
2. Inside `dataset`, create two folders: `benign` and `malignant`.
3. Place your images in their respective folders.

Your final structure should look like this:
```text
dataset/
├── benign/
│   ├── image_1.jpg
│   └── image_2.jpg
└── malignant/
    ├── image_1.jpg
    └── image_2.jpg
```

## 🚀 Usage

### 1. Training the Model
To train the model from scratch, simply run:
```bash
python train.py
```
**What this does:**
- Loads images from the `dataset/` folder.
- Trains the ResNet-50 model based on hyperparameters set in `config.py`.
- Automatically saves the best weights to `models/resnet50_best.pth`.
- Generates a training and validation loss/accuracy curve in the `outputs/` folder.

### 2. Evaluating the Model
Once you have trained the model, you can evaluate its performance:
```bash
python evaluate.py
```
**What this does:**
- Evaluates the saved model on the 20% validation split.
- Prints Accuracy, Sensitivity, Specificity, F1-Score, and ROC-AUC to the console.
- Saves a visual Confusion Matrix to `outputs/confusion_matrix.png`.

### 3. Making Predictions & Visualizing Grad-CAM
To predict whether a single, new CT scan image is benign or malignant:
```bash
python predict.py "path/to/your/image.jpg"
```
**What this does:**
- Outputs the predicted class and confidence probabilities.
- Automatically generates an Explainability Heatmap using **Grad-CAM**.
- Saves the visual heatmap overlay to `outputs/prediction_heatmap.png`.

## ⚙️ Configuration
You can easily tweak hyperparameters without changing the core code. Open `config.py` to modify:
- `IMG_SIZE` (default: 224)
- `BATCH_SIZE` (default: 16)
- `EPOCHS` (default: 20)
- `LEARNING_RATE` (default: 1e-4)

---
*Built as a final-year B.Tech Project.*
