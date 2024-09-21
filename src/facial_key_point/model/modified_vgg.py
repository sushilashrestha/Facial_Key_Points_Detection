import torch
import torch.nn as nn
from torchvision import models 


def get_model(device):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    for layers in model.parameters():
        layers.requires_grad = False

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
    return model.to(device=device)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device=device)

