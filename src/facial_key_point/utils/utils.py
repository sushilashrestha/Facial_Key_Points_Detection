import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from tqdm import tqdm 

import matplotlib.pyplot as plt

def train_batch(imgs, kps, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    #forward pass
    kps_pred = model(imgs)
    loss = criterion(kps_pred, kps)

    #backward pass
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def test_batch(imgs, kps, model, criterion):
    model.eval()

    #forward pass
    kps_pred = model(imgs)
    loss = criterion(kps_pred, kps)

    return loss

def train(n_epoch, train_dataloader, test_dataloader, model, criterion, optimizer):
    train_loss = []
    test_loss = []

    for epoch in range (1, n_epoch+1):
        epoch_train_loss, epoch_test_loss = 0, 0

        #train
        for images, kps in tqdm (train_dataloader, desc= f'Training {epoch}/{n_epoch}'):
            images = images.float()  # Convert images to float
            loss = train_batch(images, kps, model, criterion, optimizer)
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_dataloader)
        train_loss.append(epoch_train_loss)

        
        #validation
        for images, kps in tqdm (test_dataloader, desc= f'Testing {epoch}/{n_epoch}'):
            images = images.float()  # Convert images to float
            loss = test_batch(images, kps, model, criterion)
            epoch_test_loss += loss.item()
        epoch_test_loss /= len(test_dataloader)
        test_loss.append(epoch_test_loss)

        print(f'Epoch {epoch}/{n_epoch} : Train Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}')
    return train_loss, test_loss

def plot_curve(train_loss, test_loss, train_curve_path):
    epochs = np.arange(len(train_loss))

    plt.figure()
    plt.plot(epochs, train_loss, 'b', label='Train Loss')
    plt.plot(epochs, test_loss, 'r', label='Test Loss')
    plt.title("Train and Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(train_curve_path)

def load_image(img_path, model_input_size, device):
    normalize = transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )
    img = Image.open(img_path).convert('RGB')
    original_size = img.size

    #preprocessing image 
    img = img.resize(model_input_size, Image.Resampling.BILINEAR)
    img = img_disp = np.asarray(img) / 255.0
    img = torch.tensor(img).permute(2, 0, 1)     #transposing
    img = normalize(img).float()
    return img.to(device), img_disp

def visualization(img_path, model, vis_result_path, model_input_size, device):
    img_tensor, img_disp = load_image(img_path, model_input_size, device)

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(img_disp)

    plt.subplot(122)
    plt.title("Image with Facial Keypoints")
    plt.imshow(img_disp)

    kp_s = model(img_tensor[None]).flatten().detach().cpu()
    plt.scatter(kp_s[:68] * model_input_size[0], kp_s[68:] * model_input_size[1], c='y', s=2)
    plt.savefig(vis_result_path)