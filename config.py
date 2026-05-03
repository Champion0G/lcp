import os
import torch

# Directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Create directories if they do not exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Training Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4

# Class mappings
CLASS_NAMES = ['benign', 'malignant']
NUM_CLASSES = len(CLASS_NAMES)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Configuration
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "resnet50_best.pth")
