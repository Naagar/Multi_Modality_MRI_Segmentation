import numpy
# load modules to read and write images
import nibabel as nib
import os

from torch.utils.data import Dataset, DataLoader
# import nibabel as nib

# load the dataset
class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = nib.load(img_path).get_fdata()
        
        if self.transform:
            image = self.transform(image)
        
        return image
# Example usage
if __name__ == "__main__":
    # data_dir = 'path/to/your/data'  # Replace with your actual data directory # ssh://cit_tum_sandeep/vol/miltank/datasets/glioma/glioma_public/brats_2021_train
    data_dir = '/vol/miltank/datasets/glioma/glioma_public/brats_2021_train/BraTS2021_00000/preop/'  # Replace with your actual data directory 
    print("Loading dataset from:", data_dir) # ssh://cit_tum_sandeep/vol/miltank/datasets/glioma/glioma_public/brats_2021_train/BraTS2021_00000/preop/sub-BraTS2021_00000_ses-preop_space-sri_seg.nii.gz
    dataset = BraTSDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch.shape)  # Print the shape of the batch
        # Here you can add code to process the batch further
