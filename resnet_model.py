import torch
import torch.nn as nn
from torchvision import models

class LungCancerResNet(nn.Module):
    """
    ResNet-50 customized for Lung Cancer Binary Classification (Benign vs Malignant).
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(LungCancerResNet, self).__init__()
        
        # Load pre-trained ResNet-50
        # We use weights='IMAGENET1K_V1' for transfer learning
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        if freeze_backbone:
            # Freeze all parameters in the initial layers to preserve ImageNet features
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # Extract the number of features feeding into the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        
        # Replace the final fully connected layer with a custom classification head
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout to prevent overfitting
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        """
        return self.resnet(x)

    def get_last_conv_layer(self):
        """
        Helper method to get the last convolutional layer for Grad-CAM.
        For ResNet-50, this is layer4.
        """
        return self.resnet.layer4

if __name__ == "__main__":
    # Test the model structure
    model = LungCancerResNet()
    print("Model Architecture:")
    print(model)
    
    # Dummy forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}") # Expected: [1, 2]
