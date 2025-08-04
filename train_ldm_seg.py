# main_ldm.py
# A state-of-the-art pipeline for 3D multi-modal brain tumor segmentation
# using a Latent Diffusion Model (LDM). This involves a two-stage training process:
# 1. Train a 3D Vector-Quantized Variational Autoencoder (VQ-VAE).
# 2. Train a latent diffusion model conditioned on the VQ-VAE's outputs.
# Includes evaluation and 3D visualization logging to Weights & Biases.
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import wandb
import warnings
from tqdm import tqdm
# --- Import from Diffusers library ---
# pip install diffusers transformers accelerate
from diffusers import DDPMScheduler, UNet2DModel
# from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers import UNet2DConditionModel
from dataloader import BraTS2021Dataset
# --- Suppress specific warnings ---
warnings.filterwarnings("ignore", category=UserWarning, message="Inputs have mismatched dtype")

# --- Initialize Weights & Biases (similar to previous script) ---
try:
    wandb.init(project="glioma-segmentation-ldm", resume="allow")
    print("Weights & Biases initialized successfully.")
except Exception as e:
    print(f"Could not initialize W&B. Running without logging. Error: {e}")
    class MockWandB:
        def __init__(self): self.config = {}
        def init(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def watch(self, *args, **kwargs): pass
        def save(self, *args, **kwargs): pass
    wandb = MockWandB()

# --- 1. VQ-VAE Model (Encoder & Decoder) ---
# This autoencoder learns to compress the 3D MRI data into a compact latent space.

class VectorQuantizer(nn.Module):
    """
    Improved VectorQuantizer module with commitment loss.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: (B, C, D, H, W)
        # Permute to (B, D, H, W, C) for flattening
        inputs_permuted = inputs.permute(0, 2, 3, 4, 1).contiguous()
        inputs_flat = inputs_permuted.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(inputs_flat**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs_flat, self.embedding.weight.t()))
            
        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and un-flatten
        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized_flat.view(inputs_permuted.shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs_permuted)
        q_latent_loss = F.mse_loss(quantized, inputs_permuted.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs_permuted + (quantized - inputs_permuted).detach() # Straight-through estimator
        
        # Permute back to (B, C, D, H, W) format for the decoder.
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        
        return quantized, loss

class VQVAE(nn.Module):
    """
    A 3D VQ-VAE for MRI data. Now includes a final interpolation step
    to ensure the output size matches the input size.
    """
    def __init__(self, in_channels=4, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, embedding_dim, kernel_size=3, padding=1)
        )
        
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # --- FIX ---
        # Capture original input size to ensure output matches
        input_shape = x.shape[2:] # (D, H, W)
        z = self.encoder(x)
        z_quantized, vq_loss = self.vq(z)
        x_recon = self.decoder(z_quantized)
        # --- FIX ---
        # Interpolate output to match original input size if they differ
        if x_recon.shape[2:] != input_shape:
            x_recon = F.interpolate(x_recon, size=input_shape, mode='trilinear', align_corners=False)

        return x_recon, vq_loss

    def encode(self, x):
        z = self.encoder(x)
        z_quantized, _ = self.vq(z)
        return z_quantized
    
    # def decode(self, z):
    #     return self.decoder(z)

    def decode(self, z, target_shape):
        """
        Decodes a latent tensor 'z' and resizes the output to the provided 'target_shape'.
        Args:
            z (torch.Tensor): The latent tensor to decode.
            target_shape (tuple): The target spatial shape (D, H, W).
        """
        x_recon_latent = self.decoder(z)
        # Resize to the provided target shape to ensure it matches the ground truth.
        return F.interpolate(x_recon_latent, size=target_shape, mode='trilinear', align_corners=False)

# --- 2. Latent Diffusion Model ---

class LatentDiffusionModel(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        # --- FIX: Use UNet2DModel, which is simpler and doesn't require cross-attention arguments ---
        self.unet = UNet2DModel(
            in_channels=latent_channels * 2, # Seg latent + MRI latent
            out_channels=latent_channels,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, noisy_latent, timestep, mri_latent):
        b, c, d, h, w = noisy_latent.shape
        # Treat depth as batch dimension for the 2D U-Net
        noisy_latent_flat = noisy_latent.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        mri_latent_flat = mri_latent.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        
        # Concatenate noisy segmentation latent with conditioning MRI latent
        model_input = torch.cat([noisy_latent_flat, mri_latent_flat], dim=1)
        
        # Predict the noise (UNet2DModel takes sample and timestep)
        noise_pred_flat = self.unet(model_input, timestep, return_dict=False)[0]
        
        # Reshape back to 3D
        noise_pred = noise_pred_flat.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)
        return noise_pred

# --- 3. Data Handling (from previous script) ---
# (GliomaDataset and ToTensor3D classes are identical to the previous script)
class GliomaDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_size=(128, 128, 128)):
        self.data_dir = data_dir
        self.patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.transform = transform
        self.image_size = image_size # Target size for resizing

    def __len__(self): return len(self.patients)
    
    def __getitem__(self, idx):
        patient_id = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient_id)
        modalities = {
            't1': glob.glob(os.path.join(patient_dir, '*_t1.nii.gz'))[0],
            't1c': glob.glob(os.path.join(patient_dir, '*_t1c*.nii.gz'))[0],
            't2': glob.glob(os.path.join(patient_dir, '*_t2.nii.gz'))[0],
            'flair': glob.glob(os.path.join(patient_dir, '*_flair.nii.gz'))[0],
        }
        image_stack = []
        for mod in ['t1', 't1c', 't2', 'flair']:
            img = self.normalize(nib.load(modalities[mod]).get_fdata())
            image_stack.append(img)
        image = np.stack(image_stack, axis=0)
        
        seg_path = glob.glob(os.path.join(patient_dir, '*_seg.nii.gz'))[0]
        segmentation = (nib.load(seg_path).get_fdata() > 0).astype(np.float32)
        
        # Note: A real implementation would need robust resizing/cropping.
        # Here we assume data is preprocessed to a consistent size or use a placeholder.
        # For simplicity, we'll just convert to tensor.
        sample = {'image': image, 'segmentation': segmentation}
        if self.transform: sample = self.transform(sample)
        return sample

    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

class ToTensor3D:
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        # Permute from (C, H, W, D) to (C, D, H, W) for PyTorch 3D conv
        image = torch.from_numpy(image).permute(0, 3, 1, 2)
        segmentation = torch.from_numpy(segmentation).permute(0, 3, 1, 2)
        return {'image': image, 'segmentation': segmentation}

# --- 4. Training and Evaluation Functions ---

def dice_coefficient(y_pred, y_true, smooth=1e-6):
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    y_true = y_true.float()
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    return ((2. * intersection + smooth) / (union + smooth)).item()

def train_vqvae_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc="Training VQ-VAE"):
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)
        optimizer.zero_grad()
        recon_masks, vq_loss = model(images)
        
        # --- FIX ---
        # The model now handles resizing, so we can directly compute the loss.
        # The incorrect interpolation of the mask is removed.
        recon_loss = F.mse_loss(recon_masks, masks)
        loss = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def train_ldm_epoch(ldm, vqvae, loader, optimizer, scheduler, device):
    ldm.train()
    vqvae.eval() # VQ-VAE is frozen
    running_loss = 0.0
    for batch in tqdm(loader, desc="Training LDM"):
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)

        with torch.no_grad():
            # Encode inputs and masks into the latent space
            mri_latents = vqvae.encode(images)
            # The VQ-VAE was trained to reconstruct seg from MRI, so we use it to encode seg too
            seg_latents = vqvae.encode(torch.cat([masks]*4, dim=1)) # Create 4-channel mask
            
            # --- FIX ---
            # Resize latents to a power of 2 (e.g., 64x64) to avoid U-Net rounding errors.
            # The UNet2DModel is sensitive to input sizes that are not highly divisible by 2.
            # Original latent shape H, W is likely 60x60. We resize to 64x64.
            target_size = (64, 64)
            # Interpolate expects (B, C, D, H, W), which is our current format
            mri_latents = F.interpolate(mri_latents, size=(mri_latents.shape[2],) + target_size, mode='trilinear', align_corners=False)
            seg_latents = F.interpolate(seg_latents, size=(seg_latents.shape[2],) + target_size, mode='trilinear', align_corners=False)


        # Sample noise and timesteps
        noise = torch.randn_like(seg_latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (seg_latents.shape[0],), device=device).long()
        noisy_seg_latents = scheduler.add_noise(seg_latents, noise, timesteps)

        optimizer.zero_grad()

        # For the 2D UNet, timesteps need to be broadcasted correctly
        # when depth is treated as the batch dimension.
        b, c, d, h, w = noisy_seg_latents.shape
        timestep_b = timesteps.repeat_interleave(d)

        # Predict the noise
        noise_pred = ldm(noisy_seg_latents, timestep_b, mri_latents)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# Evaluation metrics (dice_coefficient, etc.) would be the same as the previous script
# For brevity, they are omitted here but should be included for a full evaluation.
# --- Evaluation and Visualization Function ---
# def evaluate_and_visualize_ldm(ldm, vqvae, loader, scheduler, device, epoch):
    ldm.eval()
    vqvae.eval()
    dice_scores = []
    
    with torch.no_grad():
        # Get one batch for visualization
        batch = next(iter(loader))
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)

        # Encode MRI to get conditioning latent
        mri_latents = vqvae.encode(images)
        target_size = (64, 64)
        mri_latents = F.interpolate(mri_latents, size=(mri_latents.shape[2],) + target_size, mode='trilinear', align_corners=False)
        
        # Start with random noise for segmentation latent
        seg_latents_noisy = torch.randn_like(mri_latents)

        # Denoising loop
        scheduler.set_timesteps(50) # Fewer steps for faster inference
        for t in tqdm(scheduler.timesteps, desc="Evaluating"):
            b, c, d, h, w = seg_latents_noisy.shape
            timestep_b = t.unsqueeze(0).repeat(b * d).to(device)
            
            noise_pred = ldm(seg_latents_noisy, timestep_b, mri_latents)
            seg_latents_noisy = scheduler.step(noise_pred, t, seg_latents_noisy).prev_sample

        # --- FIX: Pass the ground truth mask's shape to the decode function ---
        target_shape = masks.shape[2:] # Get the (D, H, W) of the ground truth
        pred_masks_decoded = vqvae.decode(seg_latents_noisy, target_shape)

        # Calculate Dice score
        dice = dice_coefficient(pred_masks_decoded, masks)
        dice_scores.append(dice)

        # --- W&B 3D Logging ---
        # Get the first item from the batch for visualization
        pred_mask_np = (torch.sigmoid(pred_masks_decoded[0, 0]) > 0.5).cpu().numpy()
        true_mask_np = masks[0, 0].cpu().numpy().astype(bool)

        # Log to wandb
        wandb.log({
            "3D Segmentation": wandb.Object3D({
                "type": "segmentation",
                "predictions": {
                    "mask_data": pred_mask_np,
                    "class_labels": {1: "Tumor"}
                },
                "ground_truth": {
                    "mask_data": true_mask_np,
                    "class_labels": {1: "Tumor"}
                }
            }),
            "Epoch": epoch
        })

    return np.mean(dice_scores)


def evaluate_and_visualize_ldm(ldm, vqvae, loader, scheduler, device, epoch):
    ldm.eval()
    vqvae.eval()
    dice_scores = []
    
    with torch.no_grad():
        # Get one batch for visualization
        batch = next(iter(loader))
        images = batch['image'].to(device)
        masks = batch['segmentation'].to(device)

        if masks.dim() > 5:
           masks = masks.squeeze(2)

        # Encode MRI to get conditioning latent
        mri_latents = vqvae.encode(images)
        target_size = (64, 64)
        mri_latents = F.interpolate(mri_latents, size=(mri_latents.shape[2],) + target_size, mode='trilinear', align_corners=False)
        
        # Start with random noise for segmentation latent
        seg_latents_noisy = torch.randn_like(mri_latents)

        # Denoising loop
        scheduler.set_timesteps(200) # Fewer steps for faster inference
        for t in tqdm(scheduler.timesteps, desc="Evaluating"):
            b, c, d, h, w = seg_latents_noisy.shape
            timestep_b = t.unsqueeze(0).repeat(b * d).to(device)
            
            noise_pred = ldm(seg_latents_noisy, timestep_b, mri_latents)
            seg_latents_noisy = scheduler.step(noise_pred, t, seg_latents_noisy).prev_sample

        target_shape = masks.shape[2:] # Get the (C, H, W) of the ground truth
        pred_masks_decoded = vqvae.decode(seg_latents_noisy, target_shape)

        # print the shape of pred_masks_decoded     
        # print(f"Predicted masks shape: {pred_masks_decoded.shape}")

        # get the one mask from the batch
        pred_masks_decoded_one_mask = pred_masks_decoded.unsqueeze(0)  # Add batch dimension

        # Calculate Dice score
        dice = dice_coefficient(pred_masks_decoded, masks)
        dice_scores.append(dice)
        

        # --- W&B 3D Logging ---
        # Get the first item from the batch for visualization
        pred_mask_np = (torch.sigmoid(pred_masks_decoded[0, 0]) > 0.5).cpu().numpy()
        # print the shape of pred_mask_np
        print(f"Predicted mask shape: {pred_mask_np.shape}")

        # get the total number of non-zero elements in the predicted mask
        num_nonzero_pred = np.sum(pred_mask_np > 0.7).item()
        print(f"Number of non-zero elements in predicted mask: {num_nonzero_pred}")
        # get the total number of zero elements in the predicted mask   
        num_zero_pred = np.sum(pred_mask_np <= 0.5).item()
        print(f"Number of zero elements in predicted mask: {num_zero_pred}")

        # invert the pred mask 
        pred_mask_np = np.logical_not(pred_mask_np)
        
        # print the number of mask elements with the value between 0.1-0.2, 0.2-.3, ..., 0.9-1.0
        for i in range(1, 10):
            mask_range = (pred_mask_np > (i-1)/10) & (pred_mask_np <= i/10)
            num_elements = np.sum(mask_range).item()
            print(f"Number of elements in range {i-1}/10 to {i}/10: {num_elements}") 

        true_mask_np = masks[0, 0].cpu().numpy().astype(bool)

        # Convert boolean masks to 3D point clouds for visualization
        pred_points = np.argwhere(pred_mask_np)
        # print the shape of pred_points
        print(f"Predicted points shape: {pred_points.shape}")

        # print the shape of true_mask_np
        true_points = np.argwhere(true_mask_np)
        print(f"True points shape: {true_points.shape}")


        # Log point clouds to Weights & Biases
        # This format is understood by W&B for interactive 3D rendering
        print(f"  -> Found {true_points.shape[0]} ground truth points.")
        # true_labels = np.ones((true_points.shape[0], 1))
        # true_points_with_labels = np.hstack([true_points, true_labels])

        log_data = {"Epoch": epoch}
        
        # if pred_points.shape[0] > 0:
        #     print(f"  -> Found {pred_points.shape[0]} prediction points. Logging combined plot.")
        #     pred_labels = 2 * np.ones((pred_points.shape[0], 1))
        #     pred_points_with_labels = np.hstack([pred_points, pred_labels])
        #     combined_points = np.vstack([true_points_with_labels, pred_points_with_labels])
        #     log_data["Segmentation Overlay 3D"] = wandb.Object3D(combined_points)
        # else:
        #     print("  -> Predicted mask is empty. Logging only ground truth.")
        #     log_data["Segmentation Overlay 3D"] = wandb.Object3D(true_points_with_labels)
        # --- 2D Slice Visualization Logging (7 Middle Slices) ---
        middle_slice_idx = images.shape[2] // 2
        slice_visualizations = []
        # Define the range of slices to visualize (middle slice +/- 3)
        start_slice = max(0, middle_slice_idx - 3)
        end_slice = min(images.shape[2], middle_slice_idx + 4)

        for i in range(start_slice, end_slice):
            flair_slice = images[0, 0, i, :, :].cpu().numpy()
            slice_viz = wandb.Image(
                flair_slice,
                caption=f"Slice {i}",
                masks={
                    "predictions": {
                        "mask_data": pred_mask_np[i, :, :],
                        "class_labels": {1: "Tumor"}
                    },
                    "ground_truth": {
                        "mask_data": true_mask_np[i, :, :],
                        "class_labels": {1: "Tumor"}
                    },
                }
            )
            slice_visualizations.append(slice_viz)
        
        log_data["Slice Visualizations (a)"] = slice_visualizations

        # --- 2D Slice Visualization Logging (7 Middle Slices x 3 Views) ---
        middle_slice_idx = images.shape[2] // 2
        slice_visualizations_seprate = []
        start_slice = max(0, middle_slice_idx - 3)
        end_slice = min(images.shape[2], middle_slice_idx + 4)

        for i in range(start_slice, end_slice):
            flair_slice = images[0, 0, i, :, :].cpu().numpy()
            
            # 1. Just the FLAIR slice
            slice_visualizations_seprate.append(wandb.Image(flair_slice, caption=f"FLAIR Slice {i}"))
            
            # 2. FLAIR slice with Ground Truth mask
            slice_visualizations_seprate.append(wandb.Image(
                flair_slice,
                caption=f"Ground Truth Slice {i}",
                masks={"ground_truth": {"mask_data": true_mask_np[i, :, :], "class_labels": {1: "Tumor"}}}
            ))

            # 3. FLAIR slice with Prediction mask
            slice_visualizations_seprate.append(wandb.Image(
                flair_slice,
                caption=f"Prediction Slice {i}",
                masks={"predictions": {"mask_data": pred_mask_np[i, :, :], "class_labels": {1: "Tumor"}}}
            ))
            # 4. Prediction and Ground Truth overlay    
            # combined_mask = np.logical_or(pred_mask_np[i, :, :], true_mask_np[i, :, :])
            # slice_visualizations_seprate.append(wandb.Image(
                
            #     caption=f"Overlay Slice {i}",
            #     masks={
            #         "predictions": {"mask_data": pred_mask_np[i, :, :], "class_labels": {1: "Tumor"}},
            #         "ground_truth": {"mask_data": true_mask_np[i, :, :], "class_labels": {1: "Tumor"}}
            #     }
            # ))
            # 5. Only Prediction mask
            slice_visualizations_seprate.append(wandb.Image(
                flair_slice,
                caption=f"Prediction Only Slice {i}",
                masks={"predictions": {"mask_data": pred_mask_np[i, :, :], "class_labels": {1: "Tumor"}}}
            ))
        
        log_data["Slice Visualizations (b)"] = slice_visualizations_seprate
        
        wandb.log(log_data)

        # wandb.log(log_data)

    return np.mean(dice_scores)


# Dice loss, 

# --- 5. Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    config = {
        "mode": "train_ldm",  # "train_vqvae" or "train_ldm"
        "learning_rate": 1e-4,
        "epochs": 50,
        "batch_size": 6,
        "num_workers": 8,
        "latent_dim": 64,
        "num_embeddings": 512,
        "train_data_path": "/vol/miltank/datasets/glioma/glioma_public/brats_2021_train/", # <-- UPDATE THIS PATH
        "test_data_path": "/vol/miltank/datasets/glioma/glioma_public/brats_2021_valid/",    # <-- UPDATE THIS PATH
        "vqvae_path": "vqvae_glioma.pth",
        "ldm_path": "ldm_glioma.pth"
    }
    wandb.config.update(config)
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = ToTensor3D()
    train_dataset = BraTS2021Dataset(data_dir=config.train_data_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=True)
    
    # Create a separate loader for testing/validation
    test_dataset = BraTS2021Dataset(data_dir=config.test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, 
                             num_workers=config.num_workers, pin_memory=True)

    # --- STAGE 1: TRAIN VQ-VAE ---
    if config.mode == "train_vqvae":
        print("\n--- STAGE 1: Training VQ-VAE ---")
        vqvae = VQVAE(in_channels=4, embedding_dim=config.latent_dim, num_embeddings=config.num_embeddings).to(device)
        # print total number of parameters in VQ-VAE
        total_params = sum(p.numel() for p in vqvae.parameters() if p.requires_grad)
        print(f"Total parameters in VQ-VAE: {total_params}")
        optimizer = optim.Adam(vqvae.parameters(), lr=config.learning_rate)

        for epoch in range(config.epochs):
            loss = train_vqvae_epoch(vqvae, train_loader, optimizer, device)
            print(f"Epoch {epoch+1}/{config.epochs} | VQ-VAE Loss: {loss:.4f}")
            wandb.log({"Epoch": epoch, "VQ-VAE Loss": loss})
        
        torch.save(vqvae.state_dict(), config.vqvae_path)
        print(f"VQ-VAE model saved to {config.vqvae_path}")

    # --- STAGE 2: TRAIN LATENT DIFFUSION MODEL ---
    elif config.mode == "train_ldm":
        print("\n--- STAGE 2: Training Latent Diffusion Model ---")
        # Load the pre-trained (from Stage 1) VQ-VAE
        vqvae = VQVAE(in_channels=4, embedding_dim=config.latent_dim, num_embeddings=config.num_embeddings).to(device)
        try:
            vqvae.load_state_dict(torch.load(config.vqvae_path, map_location=device))
            print(f"Loaded pre-trained VQ-VAE from {config.vqvae_path}")
        except FileNotFoundError:
            print(f"ERROR: VQ-VAE weights not found at {config.vqvae_path}. Please run in 'train_vqvae' mode first.")
            exit()
        vqvae.eval() # Freeze VQ-VAE

        # Initialize LDM and noise scheduler
        ldm = LatentDiffusionModel(latent_channels=config.latent_dim).to(device)
        # print total number of parameters in LDM in millions
        total_params_ldm = sum(p.numel() for p in ldm.parameters() if p.requires_grad) / 1e6
        print(f"Total parameters in LDM: {total_params_ldm:.2f} million")
        wandb.watch(ldm, log="all", log_freq=10)  # Log gradients and parameters
        
        print("LDM architecture:")
        print(ldm)

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        optimizer = optim.Adam(ldm.parameters(), lr=config.learning_rate)
        best_dice = -1.0

        for epoch in range(config.epochs):
            loss = train_ldm_epoch(ldm, vqvae, train_loader, optimizer, noise_scheduler, device)
            print(f"Epoch {epoch+1}/{config.epochs} | LDM Loss: {loss:.4f}")
            wandb.log({"Epoch": epoch, "LDM Loss": loss})
            
            # Evaluate after each epoch
            current_dice = evaluate_and_visualize_ldm(ldm, vqvae, test_loader, noise_scheduler, device, epoch)
            print(f"Epoch {epoch+1}/{config.epochs} | Validation Dice: {current_dice:.4f}")
            wandb.log({"Validation Dice": current_dice, "Epoch": epoch})

            # Save best model and log 3D visualization
            if current_dice > best_dice:
                best_dice = current_dice
                print(f"  -> New best model found! Saving with Dice: {best_dice:.4f}")
                torch.save(ldm.state_dict(), config.ldm_path)
                print(f"LDM model saved to {config.ldm_path}")
                wandb.save(config.ldm_path) # Save to W&B cloud storage
                # The 3D logging is now handled inside the evaluation function
                # and will be logged for this epoch.

        # torch.save(ldm.state_dict(), config.ldm_path)
        # print(f"LDM model saved to {config.ldm_path}")

    print("\n--- Pipeline Finished ---")
    wandb.finish()






#### Trashed code for reference, not used in the final script ####

# class LatentDiffusionModel_old(nn.Module):
#     def __init__(self, latent_channels):
#         super().__init__()
#         self.unet = UNet2DConditionModel(
#             in_channels=latent_channels * 2, # Seg latent + MRI latent
#             out_channels=latent_channels,
#             block_out_channels=(32, 64, 128, 256),
#             cross_attention_dim=None,
#             attention_head_dim=8,
#         )

#     def forward(self, noisy_latent, timestep, mri_latent):
#         b, c, d, h, w = noisy_latent.shape
#         noisy_latent_flat = noisy_latent.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
#         mri_latent_flat = mri_latent.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
#         model_input = torch.cat([noisy_latent_flat, mri_latent_flat], dim=1)
#         noise_pred_flat = self.unet(model_input, timestep, return_dict=False)[0]
#         noise_pred = noise_pred_flat.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)
#         return noise_pred
        
# def train_vqvae_epoch(model, loader, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     for batch in tqdm(loader, desc="Training VQ-VAE"):
#         # We train the VQ-VAE to reconstruct the segmentation mask from the input MRI
#         images = batch['image'].to(device)
#         masks = batch['segmentation'].to(device)
        
#         optimizer.zero_grad()
#         recon_masks, vq_loss = model(images)
#         recon_loss = F.mse_loss(recon_masks, masks)
#         loss = recon_loss + vq_loss
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     return running_loss / len(loader)

# class ToTensor3D:
#     def __call__(self, sample):
#         image, segmentation = sample['image'], sample['segmentation']
#         return {
#             'image': torch.from_numpy(image).float(),
#             'segmentation': torch.from_numpy(segmentation).float().unsqueeze(0)
#         }


# class VectorQuantizer(nn.Module):
#     """
#     Improved VectorQuantizer module with commitment loss and EMA updates.
#     """
#     def forward(self, inputs):
#         # inputs: (B, C, D, H, W)
#         # Permute to (B, D, H, W, C) for flattening
#         inputs_permuted = inputs.permute(0, 2, 3, 4, 1).contiguous()
#         inputs_flat = inputs_permuted.view(-1, self.embedding_dim)
        
#         # Calculate distances
#         distances = (torch.sum(inputs_flat**2, dim=1, keepdim=True) 
#                      + torch.sum(self.embedding.weight**2, dim=1)
#                      - 2 * torch.matmul(inputs_flat, self.embedding.weight.t()))
            
#         # Find closest embeddings
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#         encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
        
#         # Quantize and un-flatten
#         quantized_flat = torch.matmul(encodings, self.embedding.weight)
#         quantized = quantized_flat.view(inputs_permuted.shape)
        
#         # Loss
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs_permuted)
#         q_latent_loss = F.mse_loss(quantized, inputs_permuted.detach())
#         loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
#         quantized = inputs_permuted + (quantized - inputs_permuted).detach() # Straight-through estimator
        
#         # --- FIX ---
#         # Permute back to (B, C, D, H, W) format for the decoder.
#         # The previous version had a buggy permutation here.
#         quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        
#         return quantized, loss

# class VQVAE(nn.Module):
#     """
#     A 3D VQ-VAE for MRI data.
#     """
#     def __init__(self, in_channels=4, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
#         super(VQVAE, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(64, embedding_dim, kernel_size=3, padding=1)
#         )
        
#         self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose3d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(32, 1, kernel_size=3, padding=1) # Output is single-channel segmentation
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         z_quantized, vq_loss = self.vq(z)
#         x_recon = self.decoder(z_quantized)
#         return x_recon, vq_loss

#     def encode(self, x):
#         z = self.encoder(x)
#         z_quantized, _ = self.vq(z)
#         return z_quantized

#     def decode(self, z):
#         return self.decoder(z)

# class LatentDiffusionModel(nn.Module):
#     def __init__(self, latent_channels):
#         super().__init__()
#         # We use a standard 2D U-Net from diffusers and apply it slice-wise
#         # The conditioning (MRI latent) will be concatenated to the input
#         self.unet = UNet2DConditionModel(
#             in_channels=latent_channels * 2, # Seg latent + MRI latent
#             out_channels=latent_channels,
#             block_out_channels=(32, 64, 128, 256),
#             cross_attention_dim=None, # No text conditioning
#             attention_head_dim=8,
#         )

#     def forward(self, noisy_latent, timestep, mri_latent):
#         # B, C, D, H, W
#         b, c, d, h, w = noisy_latent.shape
        
#         # Treat depth as batch dimension
#         noisy_latent_flat = noisy_latent.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
#         mri_latent_flat = mri_latent.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        
#         # Concatenate noisy segmentation latent with conditioning MRI latent
#         model_input = torch.cat([noisy_latent_flat, mri_latent_flat], dim=1)
        
#         # Predict noise
#         noise_pred_flat = self.unet(model_input, timestep, return_dict=False)[0]
        
#         # Reshape back to 3D
#         noise_pred = noise_pred_flat.reshape(b, d, c, h, w).permute(0, 2, 1, 3, 4)
#         return noise_pred
# class VQVAE(nn.Module):
#     """
#     A 3D VQ-VAE for MRI data.
#     """
#     def __init__(self, in_channels=4, embedding_dim=64, num_embeddings=512, commitment_cost=0.25):
#         super().__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(64, embedding_dim, kernel_size=3, padding=1)
#         )
        
#         self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose3d(embedding_dim, 64, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(32, 1, kernel_size=3, padding=1) # Output is single-channel segmentation
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         z_quantized, vq_loss = self.vq(z)
#         x_recon = self.decoder(z_quantized)
#         return x_recon, vq_loss

#     def encode(self, x):
#         z = self.encoder(x)
#         z_quantized, _ = self.vq(z)
#         return z_quantized

#     def decode(self, z):
#         return self.decoder(z)



# def evaluate_and_visualize_ldm(ldm, vqvae, loader, scheduler, device, epoch):
#     ldm.eval()
#     vqvae.eval()
#     dice_scores = []
    
#     with torch.no_grad():
#         # Get one batch for visualization
#         batch = next(iter(loader))
#         images = batch['image'].to(device)
#         masks = batch['segmentation'].to(device)

#         # Encode MRI to get conditioning latent
#         mri_latents = vqvae.encode(images)
#         target_size = (64, 64)
#         mri_latents = F.interpolate(mri_latents, size=(mri_latents.shape[2],) + target_size, mode='trilinear', align_corners=False)
        
#         # Start with random noise for segmentation latent
#         seg_latents_noisy = torch.randn_like(mri_latents)

#         # Denoising loop
#         scheduler.set_timesteps(50) # Fewer steps for faster inference
#         for t in tqdm(scheduler.timesteps, desc="Evaluating"):
#             b, c, d, h, w = seg_latents_noisy.shape
#             timestep_b = t.unsqueeze(0).repeat(b * d).to(device)
            
#             noise_pred = ldm(seg_latents_noisy, timestep_b, mri_latents)
#             seg_latents_noisy = scheduler.step(noise_pred, t, seg_latents_noisy).prev_sample

#         # Decode the final denoised latent
#         # Resize back to the original latent size before decoding if necessary
#         # This depends on the VQ-VAE decoder's expectation
#         pred_masks_decoded = vqvae.decode(seg_latents_noisy)

#         # Calculate Dice score
#         dice = dice_coefficient(pred_masks_decoded, masks)
#         dice_scores.append(dice)

#         # --- W&B 3D Logging ---
#         # Get the first item from the batch for visualization
#         pred_mask_np = (torch.sigmoid(pred_masks_decoded[0, 0]) > 0.5).cpu().numpy()
#         true_mask_np = masks[0, 0].cpu().numpy().astype(bool)

#         # Log to wandb
#         wandb.log({
#             "3D Segmentation": wandb.Object3D({
#                 "type": "segmentation",
#                 "predictions": {
#                     "mask_data": pred_mask_np,
#                     "class_labels": {1: "Tumor"}
#                 },
#                 "ground_truth": {
#                     "mask_data": true_mask_np,
#                     "class_labels": {1: "Tumor"}
#                 }
#             }),
#             "Epoch": epoch
#         })

#     return np.mean(dice_scores)
