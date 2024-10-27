import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
import scipy.io as sio
import scipy
from timm.models.swin_transformer import SwinTransformer
from model_dispalcement import SwinTransformerSys

class DeformationDataset(Dataset):
    def __init__(self, original_dir, transformed_dir, displacement_dir, transform=None):
        self.original_dir = original_dir
        self.transformed_dir = transformed_dir
        self.displacement_dir = displacement_dir
        self.transform = transform
        self.num_images = 85700

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        original_image_path = os.path.join(self.original_dir, f'reference{idx + 1}.bmp')
        transformed_image_path = os.path.join(self.transformed_dir, f'deformation{idx + 1}.bmp')
        displacement_path = os.path.join(self.displacement_dir, f'displacement{idx + 1}.mat')

        original_image = Image.open(original_image_path).convert('L')
        transformed_image = Image.open(transformed_image_path).convert('L')
        displacement_data = scipy.io.loadmat(displacement_path)

        displacement = displacement_data['uu']  # key: 'uu' 
        displacement = np.reshape(displacement, (2, 128, 128))

        if self.transform:
            original_image = self.transform(original_image)
            transformed_image = self.transform(transformed_image)
        displacement = torch.tensor(displacement, dtype=torch.float32)

        return original_image, transformed_image, displacement

# Enhanced transform with data augmentation (optional)
transform = transforms.Compose([
    transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize if necessary
])

# Initialize the DICNET_dataset
full_dataset = DeformationDataset(
    original_dir=r'E:\SwinT_UNET_data\----datasetDICNET\int_noise\ref',
    transformed_dir=r'E:\SwinT_UNET_data\----datasetDICNET\int_noise\def',
    displacement_dir=r'E:\SwinT_UNET_data\----datasetDICNET\label\displacement',
    transform=transform
)

# Split the dataset
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Data loaders
batch_size = 16
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

save_dir = r'E:\SwinT_UNET_data\displacement_PTH'
os.makedirs(save_dir, exist_ok=True)

def train_with_scheduler(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=15, device='cuda'):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_loss = 0.0
        batch_count = 0

        for i, (original, transformed, displacement) in enumerate(train_loader):
            original, transformed, displacement = original.to(device), transformed.to(device), displacement.to(device)

            optimizer.zero_grad()
            outputs = model(original, transformed)
            loss = criterion(outputs, displacement)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()
            batch_count += 1

            if batch_count == 100:
                print(f'Epoch [{epoch + 36}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100}')
                running_loss = 0.0
                batch_count = 0

        scheduler.step()

        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for original, transformed, displacement in test_loader:
                original, transformed, displacement = original.to(device), transformed.to(device), displacement.to(device)
                outputs = model(original, transformed)
                loss = criterion(outputs, displacement)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)

        print(f'Epoch [{epoch + 36}/{num_epochs}], Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, LR: {scheduler.get_last_lr()}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_epoch_{epoch+36}.pth'))
            print(f'Best model saved at epoch {epoch+36}')

        torch.save(model.state_dict(), os.path.join(save_dir, f'V11_train_test_{epoch+36}.pth'))

if __name__ == '__main__':
    model = SwinTransformerSys()
    pretrained_model_path = r"E:\SwinT_UNET_data\displacement_PTH\best_model_epoch_35.pth"
    if os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        print("Pre-trained model loaded successfully.")
    else:
        print("Pre-trained model not found. Training from scratch.")
        
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Adjust weight_decay as needed
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    train_with_scheduler(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=265, device='cuda' if torch.cuda.is_available() else 'cpu')
