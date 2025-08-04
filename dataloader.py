import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from tqdm import tqdm

class BraTS2021Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading the BraTS 2021 dataset
    with the specific directory structure provided. This version includes
    a check to ensure all required files exist for each patient.
    
    Structure expected:
    - data_dir
      - BraTS2021_00000
        - preop
          - sub-BraTS2021_00000_ses-preop_space-sri_flair.nii.gz
          - ... (and all other modalities)
      - BraTS2021_00002
        - ...
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Path to the directory with all the patient folders (e.g., '.../brats_2021_train').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # --- MODIFICATION START ---
        # Scan the directory and create a list of valid patient samples
        # that have all the required files.
        self.patients = []
        all_patient_folders = sorted([d for d in os.listdir(data_dir) if d.startswith('BraTS2021_') and os.path.isdir(os.path.join(data_dir, d))])
        
        print("Scanning dataset for complete samples...")
        for patient_id in tqdm(all_patient_folders, desc="Verifying patients"):
            patient_dir = os.path.join(self.data_dir, patient_id, 'preop')
            
            # Check if all 5 required files exist
            flair_path = glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_flair.nii.gz'))
            t1_path    = glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t1.nii.gz'))
            t1c_path   = glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t1c.nii.gz'))
            t2_path    = glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t2.nii.gz'))
            seg_path   = glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_seg.nii.gz'))
            
            if all([flair_path, t1_path, t1c_path, t2_path, seg_path]):
                self.patients.append(patient_id) # Add patient only if all files are found
        
        print(f"Found {len(self.patients)} complete patient samples out of {len(all_patient_folders)} total folders.")
        # --- MODIFICATION END ---


    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # Get the patient ID from the pre-verified list
        patient_id = self.patients[idx]
        patient_dir = os.path.join(self.data_dir, patient_id, 'preop')
        
        # Now we can safely use [0] because we've already verified the files exist
        modalities = {
            'flair': glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_flair.nii.gz'))[0],
            't1':    glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t1.nii.gz'))[0],
            't1c':   glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t1c.nii.gz'))[0],
            't2':    glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t2.nii.gz'))[0],
            'seg':   glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_seg.nii.gz'))[0]
        }
        
        # Load, normalize, and stack the MRI modalities
        image_stack = []
        for mod_name in ['flair', 't1', 't1c', 't2']:
            img = nib.load(modalities[mod_name]).get_fdata(dtype=np.float32)
            img_normalized = self.normalize(img)
            image_stack.append(img_normalized)
        
        image = np.stack(image_stack, axis=0) # Shape: (4, H, W, D)
        
        # Load the segmentation mask
        segmentation = nib.load(modalities['seg']).get_fdata(dtype=np.float32)
        segmentation_binary = (segmentation > 0).astype(np.float32)
        segmentation_binary = np.expand_dims(segmentation_binary, axis=0) # Shape: (1, H, W, D)

        sample = {'image': image, 'segmentation': segmentation_binary}

        # print the shapes of the loaded data for debugging
        # print(f"Loaded patient {patient_id}: image shape {image.shape}, segmentation shape {segmentation_binary.shape}")

        if self.transform:
            sample = self.transform(sample)

        return sample

    def normalize(self, data):
        """Normalize image intensity to the range [0, 1]."""
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)


class BraTS2021Dataset_old(Dataset):
    """
    Custom PyTorch Dataset for loading the BraTS 2021 dataset
    with the specific directory structure provided.
    
    Structure expected:
    - data_dir
      - BraTS2021_00000
        - preop
          - sub-BraTS2021_00000_ses-preop_space-sri_flair.nii.gz
          - sub-BraTS2021_00000_ses-preop_space-sri_t1.nii.gz
          - sub-BraTS2021_00000_ses-preop_space-sri_t1c.nii.gz
          - sub-BraTS2021_00000_ses-preop_space-sri_t2.nii.gz
          - sub-BraTS2021_00000_ses-preop_space-sri_seg.nii.gz
      - BraTS2021_00002
        - ...
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Path to the directory with all the patient folders (e.g., '.../brats_2021_train').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        # Get a list of all patient directories
        self.patients = sorted([d for d in os.listdir(data_dir) if d.startswith('BraTS2021_') and os.path.isdir(os.path.join(data_dir, d))])
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        # Get the patient ID from the list
        patient_id = self.patients[idx]
        # Construct the path to the preop directory
        patient_dir = os.path.join(self.data_dir, patient_id, 'preop')
        
        # Use glob to find the files robustly, matching the specific naming convention
        modalities = {
            'flair': glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_flair.nii.gz'))[0],
            't1':    glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t1.nii.gz'))[0],
            't1c':   glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t1c.nii.gz'))[0],
            't2':    glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_t2.nii.gz'))[0],
            'seg':   glob.glob(os.path.join(patient_dir, f'sub-{patient_id}_*_seg.nii.gz'))[0]
        }
        
        # Load, normalize, and stack the MRI modalities
        image_stack = []
        for mod_name in ['flair', 't1', 't1c', 't2']:
            img = nib.load(modalities[mod_name]).get_fdata(dtype=np.float32)
            img_normalized = self.normalize(img)
            image_stack.append(img_normalized)
        
        image = np.stack(image_stack, axis=0) # Shape: (4, H, W, D)
        
        # Load the segmentation mask
        segmentation = nib.load(modalities['seg']).get_fdata(dtype=np.float32)
        # Convert multi-class labels to a binary mask (tumor vs. non-tumor)
        # BraTS labels: 1=necrotic/non-enhancing, 2=edema, 4=enhancing
        segmentation_binary = (segmentation > 0).astype(np.float32)
        
        # Add a channel dimension to the segmentation mask
        segmentation_binary = np.expand_dims(segmentation_binary, axis=0) # Shape: (1, H, W, D)

        sample = {'image': image, 'segmentation': segmentation_binary}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def normalize(self, data):
        """Normalize image intensity to the range [0, 1]."""
        # Add a small epsilon to avoid division by zero
        return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

class ToTensor3D:
    """Converts numpy arrays in a sample to Tensors and permutes dimensions for PyTorch."""
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        
        # PyTorch expects channels first: (C, D, H, W)
        image = torch.from_numpy(image).permute(0, 3, 1, 2)
        segmentation = torch.from_numpy(segmentation).permute(0, 3, 1, 2)
        
        return {
            'image': image,
            'segmentation': segmentation
        }

# --- Example Usage ---
if __name__ == '__main__':
    # !!! IMPORTANT !!!
    # UPDATE THIS PATH to point to your 'brats_2021_train' directory
    DATASET_PATH = '/vol/miltank/datasets/glioma/glioma_public/brats_2021_valid/'

    print(f"Checking dataset path: {DATASET_PATH}")
    if not os.path.isdir(DATASET_PATH):
        print("\nERROR: Dataset directory not found.")
        print("Please update the 'DATASET_PATH' variable in the script to the correct location.")
    else:
        print("Dataset directory found. Initializing dataset...")
        
        # Initialize the dataset with the custom ToTensor transform
        brats_dataset = BraTS2021Dataset(data_dir=DATASET_PATH, transform=ToTensor3D())
        
        print(f"\nSuccessfully initialized dataset.")
        print(f"Number of patients found: {len(brats_dataset)}")
        
        if len(brats_dataset) > 0:
            # Create a DataLoader to handle batching
            data_loader = DataLoader(brats_dataset, batch_size=2, shuffle=True, num_workers=0)
            
            # Get one sample batch from the data loader
            sample_batch = next(iter(data_loader))
            
            # Inspect the shapes of the output tensors
            images_batch, segmentations_batch = sample_batch['image'], sample_batch['segmentation']
            
            print("\n--- Inspecting a sample batch ---")
            print(f"Batch of images shape: {images_batch.shape}")
            print(f"Batch of segmentations shape: {segmentations_batch.shape}")
            print("\nExpected shapes are (Batch Size, Channels, Depth, Height, Width)")
            print("Image channels should be 4 (flair, t1, t1c, t2).")
            print("Segmentation channels should be 1 (binary mask).")

            # You can now use this 'data_loader' in your PyTorch training loop
            print("\nDataLoader is ready to be used in a training loop.")

