# ST-DIC
This repository contains the implementation of the Swin Transformer-based Digital Image Correlation (ST-DIC) method for displacement and strain measurement
# Swin Transformer System Documentation

## Overview

This codebase implements a Swin Transformer-based system for processing and analyzing deformation data. The system is designed to handle tasks such as displacement and strain estimation from image data, leveraging the capabilities of the Swin Transformer architecture.

## Components

### 1. Model Components

#### SwinTransformerSys

The `SwinTransformerSys` class is the core model that utilizes the Swin Transformer architecture for processing image data. It is designed to handle both displacement and strain estimation tasks.

- **Initialization**: The model is initialized with parameters such as image size, patch size, number of channels, number of classes, and various hyperparameters related to the Swin Transformer architecture.
- **Forward Pass**: The model processes two input images (e.g., original and deformed) and outputs the estimated displacement or strain fields.


```497:657:model_strain.py
class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=128, patch_size=4, in_chans=2, num_classes=2,
                 # Assuming output is a 2D displacement vector
                 embed_dim=96, depths=[2, 3, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)
        self.strain_layer = StrainLayer()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Decoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C

        return x
    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = self.output(x)

        return x

    def forward(self, img1, img2):
        # Concatenate the two images along the channel dimension
        x = torch.cat([img1, img2], dim=1)  # B, 2, H, W
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        x = self.strain_layer(x)
```


### 2. Dataset Handling

#### DeformationDataset

The `DeformationDataset` class is a custom PyTorch dataset for loading and processing deformation data. It handles loading images and corresponding displacement or strain data from specified directories.

- **Initialization**: Takes directories for original images, transformed images, and displacement/strain data.
- **Data Loading**: Loads images and data, applies transformations, and returns them as tensors.


```15:43:train_displacement.py
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
```


### 3. Training Scripts

#### train_displacement.py

This script is responsible for training the Swin Transformer model for displacement estimation.

- **Data Loading**: Utilizes the `DeformationDataset` to load training and testing data.
- **Model Training**: Configures the model, loss function, optimizer, and learning rate scheduler. It then trains the model over a specified number of epochs.


```127:139:train_displacement.py
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
```


#### train_strain.py

This script is similar to `train_displacement.py` but is tailored for training the model for strain estimation.

- **Data Loading**: Uses the `DeformationDataset` to load strain data.
- **Model Training**: Sets up the model, loss function, optimizer, and scheduler, and trains the model.


```93:100:train_strain.py
if __name__ == '__main__':

    model = SwinTransformerSys()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs=300,
                         device='cuda' if torch.cuda.is_available() else 'cpu')
```


### 4. Utility Functions

#### Window Partition

The `window_partition` function is used to partition an input tensor into smaller windows, which is a crucial step in the Swin Transformer architecture.


```25:45:model_strain.py
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # 检查高度和宽度是否可以被window_size整除
    assert H % window_size == 0, f"Height {H} is not divisible by window_size {window_size}"
    assert W % window_size == 0, f"Width {W} is not divisible by window_size {window_size}"

    # 调整形状
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # 调整维度顺序
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows
```


## Usage

1. **Data Preparation**: Ensure that your dataset is organized with directories for original images, transformed images, and displacement/strain data.

2. **Training**: Use the provided training scripts (`train_displacement.py` and `train_strain.py`) to train the model on your dataset. Adjust hyperparameters as needed.

3. **Inference**: Once trained, the model can be used to predict displacement or strain fields from new image data.

## Dependencies

- PyTorch
- torchvision
- timm
- einops
- numpy
- scipy
- PIL
- OpenCV (cv2)

## Installation

To install the required dependencies, you can use the following command:

```bash
pip install torch torchvision timm einops numpy scipy pillow opencv-python
```

## Conclusion

This codebase provides a comprehensive framework for using Swin Transformers in deformation analysis tasks. By following the provided structure and scripts, users can train and deploy models for displacement and strain estimation efficiently.
