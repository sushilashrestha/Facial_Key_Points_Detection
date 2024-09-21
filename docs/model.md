# Model Architecture

The model used for facial keypoint detection is a modified VGG16 architecture, with the following adjustments:

1. **Base Model**: VGG16 pre-trained on ImageNet.
2. **Modified Average Pooling Layer**: The average pooling layer is replaced by a combination of `Conv2D` and `MaxPool` layers to further reduce spatial dimensions.
3. **Fully Connected Layers**:
   - First `Linear` layer with 2048 inputs and 512 outputs.
   - ReLU activation followed by a Dropout layer.
   - Second `Linear` layer outputs 136 values (representing 68 keypoint coordinates).

The model's layers are frozen except for the final fully connected layers, allowing the network to specialize in keypoint prediction without retraining the entire architecture.

### Model Summary

```python
import torch.nn as nn
import torchvision.models as models

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# Freeze all layers except for the classifier
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier
model.avgpool = nn.Sequential(
    nn.Conv2d(512, 512, 3),
    nn.MaxPool2d(2),
    nn.Flatten()
)

model.classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 136),
    nn.Sigmoid()
)
```
