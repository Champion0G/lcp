import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import torch
import torch.nn.functional as F

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Plots and saves the training and validation loss and accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    """
    Plots and saves the confusion matrix heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

# --- Grad-CAM Implementation ---
class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into the target layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output[0] contains the gradients with respect to the output of the layer
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_image, class_idx=None):
        """
        Generates the Grad-CAM heatmap for the given input image.
        """
        self.model.eval()
        
        # Unfreeze all parameters temporarily so gradients can flow back to the target layer
        for param in self.model.parameters():
            param.requires_grad = True
            
        # Forward pass
        model_output = self.model(input_image)
        
        if class_idx is None:
            class_idx = torch.argmax(model_output, dim=1).item()
            
        # Target score to backpropagate
        target = model_output[0][class_idx]
        
        # Backward pass
        self.model.zero_grad()
        target.backward()
        
        # Get pooled gradients and activations
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0] # [Channels, H, W]
        
        # Weight the channels by corresponding gradients
        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]
            
        # Compute the heatmap (ReLU on the weighted sum)
        heatmap = torch.mean(activations, dim=0).squeeze().detach().cpu()
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap between 0 and 1
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, save_path=None, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the Grad-CAM heatmap on the original image.
    """
    # Read original image
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Could not read image at {img_path}")
        
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to RGB format using the specified colormap
    heatmap_colored = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    
    if save_path:
        # Convert back to BGR for saving with cv2
        save_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
        
    return superimposed_img
