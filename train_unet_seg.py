import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop
import torch.nn.functional as F
# --- Corrected 3D U-Net Model ---
# This version includes a cropping function to handle spatial mismatches
# in the skip connections, resolving the RuntimeError.

class UNet3D_with_cropping(nn.Module):
    """
    A standard 3D U-Net architecture for volumetric medical image segmentation.
    This version includes a cropping mechanism to ensure that feature maps from
    the encoder and decoder paths have matching spatial dimensions for concatenation.
    """
    def __init__(self, in_channels, out_channels, n_filters=16):
        super(UNet3D_with_cropping, self).__init__()

        # Helper for a double convolution block
        def _conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder path
        self.enc1 = _conv_block(in_channels, n_filters)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _conv_block(n_filters, n_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = _conv_block(n_filters * 2, n_filters * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = _conv_block(n_filters * 4, n_filters * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = _conv_block(n_filters * 8, n_filters * 16)

        # Decoder path
        self.up4 = nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2)
        self.dec4 = _conv_block(n_filters * 16, n_filters * 8)
        self.up3 = nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(n_filters * 8, n_filters * 4)
        self.up2 = nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(n_filters * 4, n_filters * 2)
        self.up1 = nn.ConvTranspose3d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.dec1 = _conv_block(n_filters * 2, n_filters)

        # Final output layer
        self.final_conv = nn.Conv3d(n_filters, out_channels, kernel_size=1)

    def crop_and_concat(self, upsampled, bypass):
        """
        Crops the bypass tensor to match the spatial dimensions of the upsampled tensor,
        then concatenates them along the channel dimension.
        """
        # Get the spatial dimensions (D, H, W) of the upsampled tensor
        target_size = upsampled.shape[2:]
        # Crop the bypass tensor using its own spatial dimensions
        # NOTE: torchvision.transforms.functional.center_crop expects (H, W),
        # so we need a custom approach for 3D or apply it slice-wise.
        # A simpler and direct way is to calculate slicing indices.
        
        c_h, c_w, c_d = bypass.shape[2], bypass.shape[3], bypass.shape[4]
        t_h, t_w, t_d = target_size[0], target_size[1], target_size[2]
        
        delta_h = c_h - t_h
        delta_w = c_w - t_w
        delta_d = c_d - t_d
        
        h_start, w_start, d_start = delta_h // 2, delta_w // 2, delta_d // 2
        
        bypass_cropped = bypass[:, :, h_start:h_start + t_h, w_start:w_start + t_w, d_start:d_start + t_d]
        
        return torch.cat((upsampled, bypass_cropped), dim=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with cropping
        d4 = self.up4(b)
        # The crop_and_concat function handles the potential size mismatch
        d4 = self.crop_and_concat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.crop_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.crop_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.crop_and_concat(d1, e1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

class UNet3D_no_cropping(nn.Module):
    """
    A standard 3D U-Net architecture for volumetric medical image segmentation.
    This version includes a cropping mechanism for skip connections and a final
    interpolation layer to match the output size to the input size.
    """
    def __init__(self, in_channels, out_channels, n_filters=16):
        super(UNet3D_no_cropping, self).__init__()

        # Helper for a double convolution block
        def _conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder path
        self.enc1 = _conv_block(in_channels, n_filters)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = _conv_block(n_filters, n_filters * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = _conv_block(n_filters * 2, n_filters * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = _conv_block(n_filters * 4, n_filters * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = _conv_block(n_filters * 8, n_filters * 16)

        # Decoder path
        self.up4 = nn.ConvTranspose3d(n_filters * 16, n_filters * 8, kernel_size=2, stride=2)
        self.dec4 = _conv_block(n_filters * 16, n_filters * 8)
        self.up3 = nn.ConvTranspose3d(n_filters * 8, n_filters * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(n_filters * 8, n_filters * 4)
        self.up2 = nn.ConvTranspose3d(n_filters * 4, n_filters * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(n_filters * 4, n_filters * 2)
        self.up1 = nn.ConvTranspose3d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.dec1 = _conv_block(n_filters * 2, n_filters)

        # Final output layer
        self.final_conv = nn.Conv3d(n_filters, out_channels, kernel_size=1)

    def crop_and_concat(self, upsampled, bypass):
        """
        Crops the bypass tensor to match the spatial dimensions of the upsampled tensor,
        then concatenates them along the channel dimension.
        Assumes PyTorch's (B, C, D, H, W) tensor format.
        """
        target_dims = upsampled.shape[2:]
        bypass_dims = bypass.shape[2:]
        
        crop_indices = [
            slice((bypass_dim - target_dim) // 2, (bypass_dim - target_dim) // 2 + target_dim)
            for bypass_dim, target_dim in zip(bypass_dims, target_dims)
        ]
        
        final_slice = (slice(None), slice(None)) + tuple(crop_indices)
        bypass_cropped = bypass[final_slice]
        
        return torch.cat((upsampled, bypass_cropped), dim=1)

    def forward(self, x):
        # --- Capture original input size ---
        input_shape = x.shape[2:] # (D, H, W)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with cropping
        d4 = self.up4(b)
        d4 = self.crop_and_concat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self.crop_and_concat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.crop_and_concat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.crop_and_concat(d1, e1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)

        # --- Interpolate to original size ---
        # This ensures the output size is always the same as the input size.
        if out.shape[2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode='trilinear', align_corners=False)

        return out
