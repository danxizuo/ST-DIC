import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import scipy.io as sio
import scipy
from timm.models.swin_transformer import SwinTransformer
from model_strain import SwinTransformerSys

class DeformationDataset(Dataset):
    def __init__(self, original_dir, transformed_dir, strain_dir, transform=None):
        self.original_dir = original_dir
        self.transformed_dir = transformed_dir
        self.strain_dir = strain_dir
        self.transform = transform
        self.num_images = 85700
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        original_image_path = os.path.join(self.original_dir, f'reference{idx + 1}.bmp')
        transformed_image_path = os.path.join(self.transformed_dir, f'deformation{idx + 1}.bmp')
        strain_path = os.path.join(self.strain_dir, f'deformation{idx + 1}.mat')

        original_image = Image.open(original_image_path).convert('L')
        transformed_image = Image.open(transformed_image_path).convert('L')
        strain_data = scipy.io.loadmat(strain_path)

        strain = strain_data['E']  # Key: 'E' 
        strain = np.reshape(strain, (3, 128, 128))

        if self.transform:
            original_image = self.transform(original_image)
            transformed_image = self.transform(transformed_image)
        strain = torch.tensor(strain, dtype=torch.float32)

        return original_image, transformed_image, strain

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128), antialias=True)  # Add antialias parameter
])


# Data loader
dataset = DeformationDataset(
    original_dir=r'E:\SwinT_UNET_data\----datasetDICNET\int_noise\ref',
    transformed_dir=r'E:\SwinT_UNET_data\----datasetDICNET\int_noise\def',
    strain_dir=r'E:\SwinT_UNET_data\----datasetDICNET\label\deformation',
    transform=transform
)
save_dir = r'E:\SwinT_UNET_data\strain_pth'
os.makedirs(save_dir, exist_ok=True)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs=15, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0.0
        batch_count = 0
        for i, (original, transformed, strain) in enumerate(dataloader):
            original, transformed, strain = original.to(device), transformed.to(device), strain.to(device)

            optimizer.zero_grad()
            outputs = model(original, transformed)

            loss = criterion(outputs, strain)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            batch_count += 1

            if batch_count == 100:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(dataloader)}], Loss: {running_loss / 100}')
                running_loss = 0.0
                batch_count = 0

        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader)}, LR: {scheduler.get_last_lr()}')
        torch.save(model.state_dict(), os.path.join(save_dir, f'strain_{epoch+1}.pth'))


if __name__ == '__main__':

    model = SwinTransformerSys()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs=300,
                         device='cuda' if torch.cuda.is_available() else 'cpu')