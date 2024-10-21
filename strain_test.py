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
# from SwinT_ResNet import DICSwinTransformer
# from Swin_UNET_strain import SwinTransformerSys
# from Swin_Unet_V11_Strain import SwinTransformerSys
# from Swin_Unet_V13_Strain import SwinTransformerSys
# from Swin_Unet_V21_Strain import SwinTransformerSys
#from Swin_Unet_V11_simpleStrain import SwinTransformerSys
# from Swin_Unet_V21_simpleStrain import SwinTransformerSys
from V11_revised_Sobel_strain import SwinTransformerSys

def forward_and_save_results(model, dataloader, image_size, patch_size, stride, output_csv, device='cuda'):
    model.to(device)
    model.eval()

    width, height = image_size
    strain_x = np.zeros((height, width))
    strain_y = np.zeros((height, width))
    strain_xy = np.zeros((height, width))
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
                    strain_x[y + j, x + k] += output[0, j, k]
                    strain_y[y + j, x + k] += output[1, j, k]
                    strain_xy[y + j, x + k]+= output[2, j, k]
                    count[y + j, x + k] += 1

    # 避免除以零
    count[count == 0] = 1e-5
    strain_x /= count
    strain_y /= count
    strain_xy/= count

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'X', 'Y', 'strain_X', 'strain_Y', 'strain_XY'])

        index = 0
        for y in range(height):
            for x in range(width):
                writer.writerow([index, x, y, strain_x[y, x], strain_y[y, x], strain_xy[y, x]])
                index += 1

    return strain_x, strain_y, strain_xy

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

def forward_and_save_results(model, dataloader, image_size, patch_size, stride, output_csv, device='cuda', discard_pixels=2):
    model.to(device)
    model.eval()

    width, height = image_size
    strain_x = np.zeros((height, width))
    strain_y = np.zeros((height, width))
    strain_xy = np.zeros((height, width))
    count = np.zeros((height, width))

    with torch.no_grad():
        for i, (def_patch, ref_patch, x, y) in enumerate(dataloader):
            def_patch = def_patch.to(device)
            ref_patch = ref_patch.to(device)

            output = model(def_patch, ref_patch)
            output = output.squeeze(0).cpu().numpy()

            for j in range(patch_size):
                for k in range(patch_size):
                    if (x == 0 and k < discard_pixels) or (y == 0 and j < discard_pixels) or \
                       (x + patch_size == width and k >= patch_size - discard_pixels) or \
                       (y + patch_size == height and j >= patch_size - discard_pixels):
                        strain_x[y + j, x + k] += output[0, j, k]
                        strain_y[y + j, x + k] += output[1, j, k]
                        strain_xy[y + j, x + k] += output[2, j, k]
                        count[y + j, x + k] += 1
                    else:
                        if (j >= discard_pixels and j < patch_size - discard_pixels and
                            k >= discard_pixels and k < patch_size - discard_pixels):
                            strain_x[y + j, x + k] += output[0, j, k]
                            strain_y[y + j, x + k] += output[1, j, k]
                            strain_xy[y + j, x + k] += output[2, j, k]
                            count[y + j, x + k] += 1

    count[count == 0] = 1e-5
    strain_x /= count
    strain_y /= count
    strain_xy /= count

    # with open(output_csv, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Index', 'X', 'Y', 'strain_X', 'strain_Y', 'strain_XY'])
    #
    #     index = 0
    #     for y in range(height):
    #         for x in range(width):
    #             writer.writerow([index, x, y, strain_x[y, x], strain_y[y, x], strain_xy[y, x]])
    #             index += 1

    return strain_x, strain_y, strain_xy
#

def plot_displacement(disp_x, disp_y, image_size, title):
    disp_error = disp_y
    disp_error_normalized = np.clip(disp_error, -0.5, 0.5)

    fig, ax = plt.subplots(figsize=(40, 5))
    cax = ax.imshow(disp_error_normalized, cmap='jet', vmin=-0.05, vmax=0.05, aspect='auto')
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(cax, ax=ax, orientation='vertical')
    plt.show()

if __name__ == '__main__':
    def_image_path = 'D:\OneDrive - ahu.edu.cn\-----A论文\-----CorrelationNet\DICchallenge\Star6StrainNoisy\Star6StrainNoisy/DIC_Challenge_Star_Strain_Noise_Def.tif'
    ref_image_path = 'D:\OneDrive - ahu.edu.cn\-----A论文\-----CorrelationNet\DICchallenge\Star6StrainNoisy\Star6StrainNoisy/DIC_Challenge_Star_Strain_Noise_Ref.tif'
    # def_image_path = "D:\OneDrive - ahu.edu.cn\-----A论文\-----CorrelationNet\DICchallenge\Star3NoNoiseStrain\DIC_Challenge_Wave_Deformed_Strain.tif"
    # ref_image_path = "D:\OneDrive - ahu.edu.cn\-----A论文\-----CorrelationNet\DICchallenge\Star3NoNoiseStrain\DIC_Challenge_Wave_Reference_Strain.tif"
    # def_image_path = "E:\SwinT_UNET_data\strain_images_cropped\series_step_182_right.bmp"
    # ref_image_path = "E:\SwinT_UNET_data\strain_images_cropped\series_step_200_right.bmp"
    patch_size = 128
    stride = 120
    image_size = (4000, 501)
    # image_size = (256, 1536)
    dataset = ImagePatchDataset(def_image_path, ref_image_path, patch_size, stride, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = SwinTransformerSys()
    model_path = r"E:\SwinT_UNET_data\strain_pth\V11_sobel_strain_best_model.pth" # Swin_Unet_strain_2262
    model.load_state_dict(torch.load(model_path))
    output_csv = 'predicted_displacements.csv'
    disp_x, disp_y, strain_xy = forward_and_save_results(model, dataloader, image_size, patch_size, stride, output_csv,
                                              device='cuda')

    # plot_displacement(disp_x, disp_y, strain_xy, image_size)
    plot_displacement(disp_x, disp_y, image_size, title=os.path.basename(model_path))
    print(f'Results saved to {output_csv}')
