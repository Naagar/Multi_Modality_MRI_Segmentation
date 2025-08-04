# 3D MRI Brain Tumor Segmentation using U-Net and Latent Diffusion Models

This repository provides the official PyTorch implementations for two powerful deep learning architectures for 3D glioma segmentation from multi-modal MRI scans (T1, T1c, T2, FLAIR). The project includes:

1.  A robust **3D U-Net**, serving as a strong and widely-used baseline for medical image segmentation.
2.  A state-of-the-art **Conditional Latent Diffusion Model (LDM)**, which uses a two-stage training process for high-fidelity segmentation.

Both pipelines are fully integrated with **Weights & Biases** for real-time monitoring, evaluation, and advanced 3D and 2D slice visualization.

## Features

* **Two Complete Model Implementations:** Train and evaluate either a classic 3D U-Net or a cutting-edge Latent Diffusion Model.
* **State-of-the-Art LDM:** The LDM uses a two-stage training process (VQ-VAE pre-training + conditional diffusion) for efficient and high-quality results.
* **Comprehensive Evaluation:** Automatically calculates the Dice coefficient to measure segmentation accuracy.
* **Advanced Visualization:** Logs detailed and interactive visualizations to Weights & Biases for in-depth qualitative analysis, including:
    * **3D Point Cloud Overlays:** A color-coded 3D plot showing the ground truth and prediction masks overlaid for direct comparison.
    * **2D Slice Galleries:** A gallery of the 7 middle slices showing the FLAIR image, ground truth mask, and predicted mask as separate, clear views.
* **Modular Code:** All scripts are organized into clear, reusable components for data loading, model definitions, and training loops.

## Dataset

This project is designed to work with a dataset that follows the BraTS 2021 structure. The data loader expects the following directory layout:

```
data/
└── train/
    ├── BraTS2021_00000/
    │   └── preop/
    │       ├── sub-BraTS2021_00000_..._flair.nii.gz
    │       ├── sub-BraTS2021_00000_..._t1.nii.gz
    │       ├── sub-BraTS2021_00000_..._t1c.nii.gz
    │       ├── sub-BraTS2021_00000_..._t2.nii.gz
    │       └── sub-BraTS2021_00000_..._seg.nii.gz
    ├── BraTS2021_00001/
    │   └── ...
    └── ...
```

The scripts will automatically scan the directory and only use patient folders that contain all five required `.nii.gz` files.

## Requirements

The code is built with PyTorch. You can install all necessary dependencies using the following steps.

1.  **Install PyTorch with CUDA support:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

2.  **Install other dependencies:**
    ```bash
    pip install diffusers transformers accelerate nibabel scikit-image wandb tqdm
    ```

## How to Run the Training Pipelines

You can choose to train either the U-Net or the Latent Diffusion Model.

---

### Part A: Training the 3D U-Net

This is the straightforward, single-stage baseline model.

1.  **Update data paths:** In `main_unet.py`, ensure the `train_data_path` and `test_data_path` in the `config` dictionary point to your dataset.
2.  **Run the script:**
    ```bash
    python main_unet.py
    ```
3.  **Result:** The script will train the U-Net, log metrics and visualizations to W&B, and save the best-performing model as `unet_glioma.pth` based on the validation Dice score.

---

### Part B: Training the Latent Diffusion Model (LDM)

This is a more advanced, two-stage process. You must run the script twice.

#### Stage 1: Train the VQ-VAE

This stage trains the autoencoder to learn a compact latent space for the MRI and segmentation data.

1.  **Configure the script:** Open `main_ldm.py` and set the `mode` in the `config` dictionary:
    ```python
    config = {
        "mode": "train_vqvae",
        # ... other parameters
    }
    ```
2.  **Update data paths:** Ensure `train_data_path` and `test_data_path` are correct.
3.  **Run the script:**
    ```bash
    python main_ldm.py
    ```
4.  **Result:** This will create a `vqvae_glioma.pth` file containing the trained autoencoder weights.

#### Stage 2: Train the Conditional LDM

This stage trains the main generative model to produce segmentation masks.

1.  **Configure the script:** Change the `mode` to `"train_ldm"`:
    ```python
    config = {
        "mode": "train_ldm",
        # ... other parameters
    }
    ```
2.  **Run the script:**
    ```bash
    python main_ldm.py
    ```
3.  **Result:** The script will first load the `vqvae_glioma.pth` weights. Then, it will train the `LatentDiffusionModel`. It will save the best-performing model as `ldm_glioma.pth` based on the validation Dice score.

## Evaluation and Visualization

Both the U-Net and LDM pipelines use **Weights & Biases** for extensive logging. After each evaluation step, the scripts will log:

1.  **Metrics:** The **Validation Dice Score** is logged to track quantitative performance.
2.  **3D Overlay Plot:** An interactive 3D point cloud is logged under `Segmentation Overlay 3D`. This plot combines the ground truth (e.g., green) and the prediction (e.g., purple) for easy comparison of true positives, false positives, and false negatives.
3.  **2D Slice Gallery:** A gallery of images is logged under `Slice Visualizations`. For the 7 middle slices of the brain volume, this includes:
    * The FLAIR MRI slice for anatomical context.
    * The ground truth mask overlaid on the FLAIR slice.
    * The predicted mask overlaid on the FLAIR slice.

This combination of quantitative metrics and rich visualizations allows for a thorough analysis of model performance.
