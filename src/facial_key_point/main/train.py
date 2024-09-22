#inbuilt packages 
import os
from PIL import Image
from tqdm import tqdm
import json

#datascience packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#PyTorch related packages
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from src.facial_key_point.config.config import configuration
from src.facial_key_point.datasets.datasets import FaceKeyPointData
from src.facial_key_point.model.modified_vgg import get_model
from src.facial_key_point.utils.utils import train, plot_curve, visualization

def main():

    saved_path = os.path.join(os.getcwd(), 'dump', configuration.get('saved_path'))
    model_path = os.path.join(saved_path, 'model.pth')
    hyperparam_path = os.path.join(saved_path, 'hyperparam.json')
    train_curve_path = os.path.join(saved_path, 'train_curve.png')
    vis_result_path = os.path.join(saved_path, 'vis_result.png')

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = FaceKeyPointData(csv_path = configuration.get('training_data_csvpath'), split='training', device=device)
    test_dataset = FaceKeyPointData(csv_path = configuration.get('test_data_csvpath'), split='test', device=device)

    train_dataloader = DataLoader(train_dataset, batch_size=configuration.get('batch_size'), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=configuration.get('batch_size'), shuffle=False)

    model = get_model(device = device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr= configuration.get('learning_rate'))

    train_loss, test_loss = train(configuration.get('n_epoch'), train_dataloader, test_dataloader, model, criterion, optimizer)

    plot_curve (train_loss, test_loss, train_curve_path)
    visualization('face.jpg', model, vis_result_path, configuration.get('model_input_size'), device=device)

    with open(hyperparam_path, 'w') as f:
        json.dump(configuration, f)
    torch.save(model, model_path)

if __name__ == '__main__':
    main()
