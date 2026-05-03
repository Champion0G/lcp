import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from config import *
from resnet_model import LungCancerResNet
from utils.visualization import GradCAM, overlay_heatmap

def predict_single_image(img_path, save_heatmap=True):
    """
    Predicts the class of a single image and generates a Grad-CAM heatmap.
    """
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
        
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Trained model not found at {MODEL_SAVE_PATH}")
        return
        
    # Load Model
    model = LungCancerResNet(num_classes=NUM_CLASSES, freeze_backbone=True)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Validation transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Initialize Grad-CAM
    target_layer = model.get_last_conv_layer()
    cam = GradCAM(model, target_layer)
    
    # Prediction
    with torch.no_grad():
        # Temporarily disable gradient tracking for standard forward pass probability
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        
    pred_idx = torch.argmax(probs).item()
    pred_class = CLASS_NAMES[pred_idx]
    pred_prob = probs[pred_idx].item()
    
    print(f"\n--- Prediction Results ---")
    print(f"Image: {img_path}")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {pred_prob:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  - {name}: {probs[i].item():.4f}")
        
    # Grad-CAM Heatmap
    if save_heatmap:
        print("\nGenerating Grad-CAM heatmap...")
        
        # Enable gradients for Grad-CAM
        with torch.enable_grad():
            heatmap = cam.generate_heatmap(input_tensor, class_idx=pred_idx)
            
        # We need a standard formatted input image (resized) for overlay
        # Saving resized original to temporary to avoid resizing it manually with cv2 here
        temp_resized_path = os.path.join(OUTPUTS_DIR, "temp_resized.jpg")
        image.resize((IMG_SIZE, IMG_SIZE)).save(temp_resized_path)
        
        heatmap_save_path = os.path.join(OUTPUTS_DIR, "prediction_heatmap.png")
        overlay_heatmap(temp_resized_path, heatmap, save_path=heatmap_save_path)
        
        # Clean up temp file
        if os.path.exists(temp_resized_path):
            os.remove(temp_resized_path)
            
        print(f"Saved explainability heatmap to {heatmap_save_path}")
        
    return pred_class, pred_prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Lung Nodule Class")
    parser.add_argument("image_path", type=str, help="Path to the CT scan image")
    parser.add_argument("--no_heatmap", action="store_true", help="Disable Grad-CAM heatmap generation")
    
    args = parser.parse_args()
    predict_single_image(args.image_path, save_heatmap=not args.no_heatmap)
