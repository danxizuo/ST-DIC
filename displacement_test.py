import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# from SwinT_CNN import DICSwinTransformer
# from SwinT_ResNet import DICSwinTransformer
# from Swin_Unet_model_128 import SwinTransformerSys
# from Swin_Unet_V14 import SwinTransformerSys
#from Swin_Unet_V13_interpolation import SwinTransformerSys
#from Swin_Unet_V11 import SwinTransformerSys
# from Swin_Unet_V13_skipc import SwinTransformerSys
from displacement_model import SwinTransformerSys
# 自定义数据集类
class ImagePatchDataset(Dataset):
    def __init__(self, def_image_path, ref_image_path, patch_size=128, stride=120, transform=None):
        self.def_image = Image.open(def_image_path).convert('L')
        self.ref_image = Image.open(ref_image_path).convert('L')
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.patches = self.extract_patches()

    def extract_patches(self):
        patches = []
        width, height = self.def_image.size
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                if x + self.patch_size > width:
                    x = width - self.patch_size
                if y + self.patch_size > height:
                    y = height - self.patch_size
                def_patch = self.def_image.crop((x, y, x + self.patch_size, y + self.patch_size))
                ref_patch = self.ref_image.crop((x, y, x + self.patch_size, y + self.patch_size))
                patches.append((def_patch, ref_patch, x, y))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        def_patch, ref_patch, x, y = self.patches[idx]
        if self.transform:
            def_patch = self.transform(def_patch)
            ref_patch = self.transform(ref_patch)
        return def_patch, ref_patch, x, y


transform = transforms.Compose([
    transforms.ToTensor()
])


# 前向计算并保存结果
def forward_and_save_results(model, dataloader, image_size, patch_size, stride, output_csv, device='cuda'):
    model.to(device)
    model.eval()

    width, height = image_size
    disp_x = np.zeros((height, width))
    disp_y = np.zeros((height, width))
    count = np.zeros((height, width))

    with torch.no_grad():
        for i, (def_patch, ref_patch, x, y) in enumerate(dataloader):
            def_patch = def_patch.to(device)
            ref_patch = ref_patch.to(device)

            # 分别传递给模型
            output = model(def_patch, ref_patch)
            output = output.squeeze(0).cpu().numpy()

            for j in range(patch_size):
                for k in range(patch_size):
                    disp_x[y + j, x + k] += output[0, j, k]
                    disp_y[y + j, x + k] += output[1, j, k]
                    count[y + j, x + k] += 1

    # 避免除以零
    count[count == 0] = 1e-5
    disp_x /= count
    disp_y /= count

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'X', 'Y', 'Disp_X', 'Disp_Y'])

        index = 0
        for y in range(height):
            for x in range(width):
                writer.writerow([index, x, y, disp_x[y, x], disp_y[y, x]])
                index += 1

    return disp_x, disp_y

def plot_displacement(disp_x, disp_y, image_size, title):
    disp_error = disp_y
    disp_error_normalized = np.clip(disp_error, -0.5, 0.5)
    fig, ax = plt.subplots(figsize=(40, 5))
    cax = ax.imshow(disp_error_normalized, cmap='jet', vmin=-0.5, vmax=0.5, aspect='auto')
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(cax, ax=ax, orientation='vertical')
    plt.show()

if __name__ == '__main__':
    def_image_path = 'D:/OneDrive - ahu.edu.cn/-----A论文/-----CorrelationNet/DIC-Net-main/Demo/Star5/Star5Def.bmp'
    ref_image_path = 'D:/OneDrive - ahu.edu.cn/-----A论文/-----CorrelationNet/DIC-Net-main/Demo/Star5/Star5Ref.bmp'

    patch_size = 128
    stride = 100
    image_size = (4000, 501)

    dataset = ImagePatchDataset(def_image_path, ref_image_path, patch_size, stride, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SwinTransformerSys()
    # model.load_state_dict(torch.load('E:\SwinT_UNET_data/SwinT_UNET_model_epoch_8.pth'))
    # model_path = 'E:\SwinT_UNET_data\displacement_PTH/SwinT_UNET_model_epoch_322.pth'
    # model_path ="E:\SwinT_UNET_data\displacement_PTH\V21_236_SwinT_UNET_model_epoch_289.pth"
    model_path = r"E:\SwinT_UNET_data\displacement_PTH\full_dataset_best_model_epoch_266.pth"
    # model_path = r"E:\SwinT_UNET_data\displacement_PTH\2262SwinT_UNET_model_epoch_258.pth"
    model.load_state_dict(torch.load(model_path))
    output_csv = 'predicted_displacements.csv'
    disp_x, disp_y = forward_and_save_results(model, dataloader, image_size, patch_size, stride, output_csv,
                                              device='cuda')
    plot_displacement(disp_x, disp_y, image_size, title=os.path.basename(model_path))
    print(f'Results saved to {output_csv}')
