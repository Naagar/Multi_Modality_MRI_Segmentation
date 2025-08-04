# The model is evaluated using:
    # Dice coefficient (at multiple thresholds)
    # Mean Absolute Error (MAE)
    # Peak Signal-to-Noise Ratio (PSNR)
    # Structural Similarity Index (SSIM)
    # Relative Absolute Volume Difference (RAVD)

import torch
# main.py
# A comprehensive script for 3D multi-modal brain tumor segmentation
# using a 3D U-Net, PyTorch, and Weights & Biases for logging.
import time
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import wandb
import warnings

from dataloader import BraTS2021Dataset
from unet_3d import UNet3D_with_cropping, UNet3D_no_cropping

# --- Suppress specific warnings from scikit-image ---
warnings.filterwarnings("ignore", category=UserWarning, message="Inputs have mismatched dtype")

# dataset shape and size
# The dataset is expected to have 4 modalities (T1, T1c, T2, FLAIR)
# and a binary segmentation mask. Each modality is a 3D volume.
# The expected input shape is (B, 4, D, H, W)
# where B is the batch size, D is the depth (number of slices), H is the height, and W is the width of the 3D volume.
# The segmentation mask is expected to have the shape (B, 1, D, H, W) for binary segmentation

# --- 1. Initialize Weights & Biases ---
# Before running, log in to your wandb account in your terminal: `wandb login`
try:
    wandb.init(project="glioma_seg_baseline_unet", resume="allow")
    print("Weights & Biases initialized successfully.")
except Exception as e:
    print(f"Could not initialize Weights & Biases. Running without logging. Error: {e}")
    # Create a mock wandb object if initialization fails
    class MockWandB:
        def __init__(self):
            self.config = {}
        def init(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def watch(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
    wandb = MockWandB()


# --- 2. Model Architecture: 3D U-Net ---
class UNet3D(nn.Module):
    """
    A standard 3D U-Net architecture for volumetric medical image segmentation.
    This model takes a 4-channel 3D input (T1, T1c, T2, FLAIR) and outputs a 1-channel segmentation map.
    """
    def __init__(self, in_channels, out_channels, n_filters=16):
        super(UNet3D, self).__init__()

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

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

def center_crop_3d(tensor, target_shape):
    """
    Center crops a 5D tensor (B, C, D, H, W) to a target 3D shape (D, H, W).
    """
    d_diff = tensor.shape[2] - target_shape[2]
    h_diff = tensor.shape[3] - target_shape[3]
    w_diff = tensor.shape[4] - target_shape[4]

    d_start = d_diff // 2
    h_start = h_diff // 2
    w_start = w_diff // 2

    return tensor[:, :, d_start:d_start + target_shape[0], h_start:h_start + target_shape[1], w_start:w_start + target_shape[2]]

# --- 3. Data Handling ---
class GliomaDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the BraTS-like Glioma dataset.
    It loads 4 MRI modalities (T1, T1c, T2, FLAIR) and the segmentation mask.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        # Assumes data is organized in subdirectories, one for each patient
        self.patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient_id)
        
        # Find the file for each modality
        # Using glob is robust to different naming conventions (e.g., t1ce vs t1c)
        modalities = {
            't1': glob.glob(os.path.join(patient_dir, '*_t1.nii.gz'))[0],
            't1c': glob.glob(os.path.join(patient_dir, '*_t1c*.nii.gz'))[0],
            't2': glob.glob(os.path.join(patient_dir, '*_t2.nii.gz'))[0],
            'flair': glob.glob(os.path.join(patient_dir, '*_flair.nii.gz'))[0],
        }
        
        # Load, normalize, and stack the modalities
        image_stack = []
        for mod in ['t1', 't1c', 't2', 'flair']:
            img = nib.load(modalities[mod]).get_fdata()
            img = self.normalize(img)
            image_stack.append(img)
        
        image = np.stack(image_stack, axis=0)
        
        # Load the segmentation mask
        seg_path = glob.glob(os.path.join(patient_dir, '*_seg.nii.gz'))[0]
        segmentation = nib.load(seg_path).get_fdata()
        # Convert multi-class labels to binary (tumor vs. non-tumor)
        segmentation = (segmentation > 0).astype(np.float32)

        sample = {'image': image, 'segmentation': segmentation}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def normalize(self, data):
        """Normalize image intensity to the range [0, 1]."""
        # Add a small epsilon to avoid division by zero
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

class ToTensor3D:
    """Converts numpy arrays in a sample to Tensors."""
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        # Add a channel dimension to the segmentation mask
        segmentation = np.expand_dims(segmentation, axis=0)
        # print(f"ToTensor3D: image shape {image.shape}, segmentation shape {segmentation.shape}")
        return {
            'image': torch.from_numpy(image).float(),
            'segmentation': torch.from_numpy(segmentation).float()
        }


# --- 4. Evaluation Metrics ---
def dice_coefficient(y_pred, y_true, threshold=0.5, smooth=1e-6):
    y_pred = (torch.sigmoid(y_pred) > threshold).float()
    y_true = y_true.float()
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def mean_absolute_error(y_pred, y_true):
    return torch.mean(torch.abs(torch.sigmoid(y_pred) - y_true)).item()

def relative_absolute_volume_difference(y_pred, y_true, threshold=0.5):
    pred_vol = torch.sum((torch.sigmoid(y_pred) > threshold).float())
    true_vol = torch.sum(y_true.float())
    if true_vol == 0:
        return 0.0 if pred_vol == 0 else float('inf')
    return torch.abs(pred_vol - true_vol).item() / true_vol.item()

def get_slice_wise_metrics(y_pred, y_true):
    """Calculate PSNR and SSIM on a slice-by-slice basis and average."""
    y_pred_np = y_pred.cpu().numpy().squeeze()
    y_true_np = y_true.cpu().numpy().squeeze()
    
    psnr_scores = []
    ssim_scores = []
    
    # Iterate over the depth dimension
    for z in range(y_pred_np.shape[2]):
        pred_slice = y_pred_np[:, :, z]
        true_slice = y_true_np[:, :, z]
        
        # Avoid calculating on empty slices
        if np.sum(true_slice) > 0:
            psnr_scores.append(peak_signal_noise_ratio(true_slice, pred_slice, data_range=1.0))
            ssim_scores.append(structural_similarity(true_slice, pred_slice, data_range=1.0))
            
    return np.mean(psnr_scores) if psnr_scores else 0.0, np.mean(ssim_scores) if ssim_scores else 0.0


# --- 5. Training & Testing Functions ---
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['segmentation'].to(device, non_blocking=True)

        masks = batch['segmentation'].to(device) 

        # Check if the mask has 6 dimensions
        if masks.dim() == 6:
            # Squeeze out the dimension at index 2
            masks = torch.squeeze(masks, dim=2)

        optimizer.zero_grad()
        outputs = model(images)
        # print("train_epoch shape of output_mask and  gt_mask, ", outputs.shape, masks.shape)

        loss = loss_fn(outputs, masks)
        # masks_cropped = center_crop_3d(masks, outputs.shape[2:])
        # print("train_epoch shape of cropped gt_mask, ", masks_cropped.shape)
        # if masks_cropped.shape != outputs.shape:
            # raise ValueError(f"Target size {masks_cropped.shape} must be the same as input size {outputs.shape}")

        # loss = loss_fn(outputs, masks_cropped)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    return running_loss / len(loader.dataset)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_metrics = {
        'dice@0.4': [], 'dice@0.5': [], 'dice@0.6': [],
        'mae': [], 'psnr': [], 'ssim': [], 'ravd': []
    }
    dice_thresholds = [0.4, 0.5, 0.6]

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['segmentation'].to(device, non_blocking=True)
            # Check if the mask has 6 dimensions
            if masks.dim() == 6:
                # Squeeze out the dimension at index 2
                masks = torch.squeeze(masks, dim=2)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Calculate metrics for each item in the batch
            for i in range(images.size(0)):
                output_i = outputs[i]
                mask_i = masks[i]
                
                for t in dice_thresholds:
                    all_metrics[f'dice@{t}'].append(dice_coefficient(output_i, mask_i, threshold=t))
                
                all_metrics['mae'].append(mean_absolute_error(output_i, mask_i))
                all_metrics['ravd'].append(relative_absolute_volume_difference(output_i, mask_i))

                # Slice-wise metrics
                psnr_val, ssim_val = get_slice_wise_metrics(torch.sigmoid(output_i), mask_i)
                all_metrics['psnr'].append(psnr_val)
                all_metrics['ssim'].append(ssim_val)

    avg_loss = running_loss / len(loader.dataset)
    avg_metrics = {key: np.mean(val) for key, val in all_metrics.items()}
    
    return avg_loss, avg_metrics

def evaluate_and_visualize_unet(model, loader, device, epoch):
    """
    Evaluates the U-Net model on a test batch, calculates the Dice score,
    and logs a 3D overlay visualization to Weights & Biases.
    """
    model.eval()
    dice_scores = []
    with torch.no_grad():
        # Get one batch from the test loader for visualization
        batch = next(iter(loader))
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)
        
        ''' 
            True Positives: Where the prediction (e.g., purple) overlaps with the ground truth (e.g., green). Green: GT, Purple: Predicted.

            False Positives: Areas where the model predicted a tumor, but there isn't one (purple points with no green underneath).

            False Negatives: Areas of the ground truth that the model missed (green points with no purple).

        '''

        # --- FIX ---
        # Squeeze the mask tensor if it has an extra dimension, ensuring it's 5D.
        # This prevents a shape mismatch with the model's 5D output.
        if masks.dim() > 5:
            masks = masks.squeeze(2)

        # Get model prediction
        outputs = model(images)
        
        # Calculate Dice score
        dice = dice_coefficient(outputs, masks)
        dice_scores.append(dice)
        
        # --- W&B 3D Logging ---
        # Get the first item from the batch for visualization
        pred_mask_np = (torch.sigmoid(outputs[0, 0]) > 0.5).cpu().numpy()
        true_mask_np = masks[0, 0].cpu().numpy().astype(bool)

        # Convert boolean masks to 3D point clouds
        pred_points = np.argwhere(pred_mask_np)
        true_points = np.argwhere(true_mask_np)

        print(f"  -> Found {true_points.shape[0]} ground truth points.")

        # Create class labels for color-coding in W&B
        true_labels = np.ones((true_points.shape[0], 1))
        true_points_with_labels = np.hstack([true_points, true_labels])
        
        log_data = {"Epoch": epoch}
        
        if pred_points.shape[0] > 0:
            print(f"  -> Found {pred_points.shape[0]} prediction points. Logging combined plot.")
            pred_labels = 2 * np.ones((pred_points.shape[0], 1))
            pred_points_with_labels = np.hstack([pred_points, pred_labels])
            
            # print the shape of pred_points_with_labels, true_points_with_labels
            print(f" evaluate_and_visualize_unet -> Predicted points with labels shape: {pred_points_with_labels.shape}")  # 
            print(f"  evaluate_and_visualize_unet -> True points with labels shape: {true_points_with_labels.shape}")       #
            # Combine for a single overlay plot
            combined_points = np.vstack([true_points_with_labels, pred_points_with_labels])
            log_data["Segmentation Overlay 3D, Green: GT, Purple: Predicted."] = wandb.Object3D(combined_points)
        else:
            print("  -> Predicted mask is empty. Logging only ground truth.")
            log_data["Segmentation Overlay 3D, Green: GT, Purple: Pred."] = wandb.Object3D(true_points_with_labels)

        wandb.log(log_data)


    return np.mean(dice_scores)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
bce_loss = nn.BCEWithLogitsLoss()
dice_loss = DiceLoss()

def combined_loss(y_pred, y_true):
    return bce_loss(y_pred, y_true) + dice_loss(y_pred, y_true)
# --- 6. Main Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    # Default config, can be overridden by W&B sweeps
    config = {
        "learning_rate": 1e-4,
        "epochs": 100,
        "batch_size": 4,
        "num_workers": 4,
        "model_filters": 16,
        "train_data_path": "/vol/miltank/datasets/glioma/glioma_public/brats_2021_train/", # <-- UPDATE THIS PATH
        # ssh://cit_tum_sandeep/vol/miltank/datasets/BraTS/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/BraTS-GoAT-00000
        "test_data_path": "/vol/miltank/datasets/glioma/glioma_public/brats_2021_valid/"    # <-- UPDATE THIS PATH
    }
    wandb.config.update(config)
    config = wandb.config # Use wandb.config for sweep compatibility

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # --- Data Loading ---
    print("Setting up data loaders...")
    transform = ToTensor3D()
    train_dataset = BraTS2021Dataset(data_dir=config.train_data_path, transform=transform)
    test_dataset = BraTS2021Dataset(data_dir=config.test_data_path, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                             num_workers=config.num_workers, pin_memory=True)
    print(f"Found {len(train_dataset)} training samples and {len(test_dataset)} test samples.")

    # --- Model, Loss, Optimizer ---
    print("Initializing model...")
    model = UNet3D_with_cropping(in_channels=4, out_channels=1, n_filters=config.model_filters).to(device) #UNet3D
    # model = UNet3D(in_channels=4, out_channels=1, n_filters=config.model_filters).to(device) #UNet3D, UNet3D_no_cropping
    model = UNet3D_no_cropping(in_channels=4, out_channels=1, n_filters=config.model_filters).to(device) #UNet3D, UNet3D_no_cropping
    
    # print the total number of parameters in the model 
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} trainable parameters.")

    # print the model architecture
    print(model)

    # A combination of Dice Loss and BCE Loss often works well for segmentation
    # For simplicity, we start with BCEWithLogitsLoss
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = combined_loss  # Combined loss works good for segmentation tasks
    # loss_fn = nn.BCEWithLogitsLoss() + DiceLoss()  # Uncomment if you have a custom DiceLoss implementation
    # use dice loss for segmentation tasks
    # loss_fn = DiceLoss()  # Uncomment if you have a custom DiceLoss implementation

    # loss function the combination of Dice Loss and BCE Loss
    # loss_fn = nn.BCEWithLogitsLoss() + DiceLoss()  # Uncomment if you have a custom DiceLoss implementation
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Log model architecture and gradients to W&B
    wandb.watch(model, log='all', log_freq=100)

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_dice = -1.0
    for epoch in range(config.epochs):
        epoch_start = time.time()  # Start timer
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device)

        # Visualize and log the model's performance on a test batch
        dice_score = evaluate_and_visualize_unet(model, test_loader, device, epoch)
        
        scheduler.step(test_loss)

        # --- Logging ---
        log_data = {
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Test Loss": test_loss,
            "Learning Rate": optimizer.param_groups[0]['lr']
        }
        # Add all test metrics to the log
        for key, value in test_metrics.items():
            log_data[f"Test {key.upper()}"] = value
        
        # --- Log 3D predictions and ground truth to wandb ---
        # Take first batch from test_loader
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    images = batch['image'].to(device)
                    masks = batch['segmentation'].to(device)
                    if masks.dim() == 6:
                        masks = torch.squeeze(masks, dim=2)
                    outputs = model(images)
                    preds = torch.sigmoid(outputs)  # Apply sigmoid and threshold
                    # Log the first sample in the batch
                    idx = 0
                    pred_np = preds[idx].cpu().numpy().squeeze()
                    mask_np = masks[idx].cpu().numpy().squeeze()
                    img_np = images[idx].cpu().numpy().squeeze()

                    # Log a montage of 8 central slices
                    mid = pred_np.shape[0] // 2
                    slice_idxs = range(mid-4, mid+4)
                    pred_slices = [wandb.Image(pred_np[z], caption=f"Pred Slice {z}") for z in slice_idxs]
                    mask_slices = [wandb.Image(mask_np[z], caption=f"GT Slice {z}") for z in slice_idxs]
                    img_slices = [wandb.Image(img_np[0, z], caption=f"Image Slice {z}") for z in slice_idxs] # first modality

                    wandb.log({
                        "Predicted Slices": pred_slices,
                        "GroundTruth Slices": mask_slices,
                        "Image Slices": img_slices,
                        "Epoch": epoch + 1
                    })
                    # --- W&B 3D Logging ---
                    # Get the first item from the batch for visualization
                    # pred_mask_np = (torch.sigmoid(pred_masks_decoded[0, 0]) > 0.5).cpu().numpy()
                    # pred_mask_np = preds[idx].cpu().numpy().squeeze()
                    # print the shape of pred_mask_np
                    print(f"Predicted mask shape: {pred_np.shape}") # (240, 240, 155)

                    # true_mask_np = masks[0, 0].cpu().numpy().astype(bool)

                    # Convert boolean masks to 3D point clouds for visualization
                    pred_points = np.argwhere(pred_np)
                    # print the shape of pred_points
                    print(f"Predicted points shape: {pred_points.shape}") # (8928000, 3)

                    # print the shape of true_mask_np
                    true_points = np.argwhere(mask_np)
                    print(f"True points shape: {true_points.shape}") # (123259, 3)

                    # Log 3D point clouds to Weights & Biases
                    # This format is understood by W&B for interactive 3D rendering
                    # --- FIX: Conditionally log the prediction only if it's not empty ---
                    log_data = {
                        "Ground Truth 3D": wandb.Object3D(true_points),
                        "Epoch": epoch
                    }
                    if pred_points.shape[0] > 0:
                        print(f"  -> Logging non-empty prediction with {pred_points.shape[0]} points.")
                        log_data["Prediction 3D"] = wandb.Object3D(pred_points)
                    else:
                        print("  -> Predicted mask is empty. Skipping prediction log.")
                        break  # Only log first batch
        
        wandb.log(log_data)
        
        epoch_time = time.time() - epoch_start  # End timer
        print(f"Epoch {epoch+1:03d}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Dice@0.5: {test_metrics['dice@0.5']:.4f} | "
              f"Time: {epoch_time:.2f} sec")

        # --- Save Best Model ---
        if test_metrics['dice@0.5'] > best_dice:
            best_dice = test_metrics['dice@0.5']
            model_path = "best_glioma_model_.pth"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path) # Save to W&B cloud storage
            print(f"  -> New best model saved with Dice@0.5: {best_dice:.4f}")

    print("\n--- Training Finished ---")
    wandb.finish()

