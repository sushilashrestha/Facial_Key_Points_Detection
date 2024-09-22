import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class FaceKeyPointData(Dataset):
    def __init__(self, csv_path='data\\training_frames_keypoints.csv', split='training', device=torch.device('cpu'), model_input_size=(224, 224)):
        super(FaceKeyPointData, self).__init__()
        self.csv_path = csv_path
        self.split = split
        self.df = pd.read_csv(self.csv_path)
        self.normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
        self.device = device 
        self.model_input_size = model_input_size
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img, original_size = self.get_img(index)
        key_points = self.get_keypoints(index = index, original_size = original_size)
        return img, key_points
    
    def get_img(self, index):
        img_path = os.path.join(os.getcwd(), 'data', self.split, self.df.iloc[index, 0])
        img = Image.open(img_path).convert('RGB')
        original_size = img.size

    #preprocessing image 
        img = img.resize(self.model_input_size, Image.Resampling.BILINEAR)
        img = np.asarray(img) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)     #transposing
        img = self.normalize(img).float()
        return img.to(self.device), original_size
    
    def get_keypoints(self, index, original_size):
        kp = self.df.iloc[index, 1:].to_numpy().astype(np.float32)
        kp_x = np.array(kp[0::2]) / original_size[0]
        kp_y = np.array(kp[1::2]) / original_size[1]
        kp = np.concatenate((kp_x, kp_y))
        return torch.tensor(kp).to(self.device)
    
    def load_img(self, index):
        img_path = os.path.join(os.getcwd(), 'data', self.split, self.df.iloc[index, 0])
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.model_input_size, Image.Resampling.BILINEAR)
        return np.asarray(img) / 255.0

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_data = FaceKeyPointData(device=device)
    test_data = FaceKeyPointData(csv_path='data\\test_frames_keypoints.csv', split='test', device=device)
    #print(training_data[0])
