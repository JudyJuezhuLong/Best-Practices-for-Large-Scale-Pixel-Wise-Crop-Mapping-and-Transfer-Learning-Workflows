# Supplementary Material

This repository includes step-by-step experiment code, extended methodological details, and summarized experiment results. The full workflow and evaluation pipeline are organized into clearly structured modules corresponding to our large-scale crop classification study.

## 🔁 Workflow Overview

A series of modular, comparative experiments spanning satellite image preprocessing, trusted sample generation, model selection, transfer learning, and crop type map prediction consists of the following modules:

### (0) 📚 Library and Core Functions
All the experiments were conducted using Python 3.9.18 with PyTorch 1.11.0, running on an Ubuntu 20.04 system with two NVIDIA A100-PCIE-40GB GPUs.
- Edited from the [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library.git)
- Located in:
  ```
  ├── examples/
  ├── tllib/
  └── tllib.egg-info/
  ```

### (1) 🛰️ Time Series Generation
- Preprocessing of satellite imagery (denoising, reconstruction) was conducted using Google Earth Engine (GEE).
- GEE url (https://code.earthengine.google.com/d5deb3dd74d784d4804249d649699bda).

### (2) ✅ Trusted Sample Generation
- See:
  ```
  └── trusted_pixel_selection/
  ```

### (3) 🌾 Crop Classification (Source Domains)
- Site-specific 10-fold cross-validation models:
  ```
  ├── 28_29_10_folds/
  ├── 28_31_10_folds/
  └── 27_30_10_folds/
  ```

### (4) 🔄 Direct Model Transfer
- Experiments where models trained from scratch were applied to new domains:
  ```
  ├── 27_30_temporal_train/
  ├── 28_29_temporal_train/
  ├── 28_31_temporal_train/
  ├── 27_30_sensor_train/
  ├── 28_29_sensor_train/
  ├── 28_31_sensor_train/
  ├── 28_31_spatial_train/
  ├── 22_34_spatial_train/
  ├── serbia_spatial_train/
  └── evaluation_4000_target_sites_trusted_samples/
  ```

### (5) 🔁 Transfer Strategies: Fine-Tuning and Domain Adaptation (DANN)
- Folders correspond to specific source-target region pairs:
  ```
  ├── transfer_spatial_dann_22_34_to_Serbia/
  ├── transfer_spatial_dann_28_29_to_22_34/
  ├── transfer_spatial_dann_28_29_to_28_31/
  ├── transfer_spatial_dann_28_29_to_Serbia/
  ├── transfer_spatial_fine_tuning_28_29_to_22_34/
  ├── transfer_spatial_fine_tuning_28_29_to_28_31/
  └── transfer_spatial_fine_tuning_28_29_to_Serbia/
  ```

### (6) 📊 Evaluation of CDL vs Trusted Samples
- Experiments and evaluations analyzing the impact of label reliability:
  ```
  ├── discussion_28_29_CDLvsTrusted/
  ├── discussion_28_31_CDLvsTrusted/
  ├── discussion_27_30_CDLvsTrusted/
  └── evaluation_4000_target_sites_trusted_samples/
  ```

### (7) 🔬 Four Fine-Tuning Treatments
- Ablation study comparing different fine-tuning strategies:
  ```
  ├── transfer_spatial_fine_tuning_22_34_28_29_to_Serbia_R1/
  ├── transfer_spatial_fine_tuning_22_34_28_29_to_Serbia_R2/
  ├── transfer_spatial_fine_tuning_22_34_28_29_to_Serbia_R3/
  └── transfer_spatial_fine_tuning_22_34_28_29_to_Serbia_R4/
  ```

## 🗂️ How to Use

This code is for reference and reproducibility. Due to size and resource constraints, full execution may require additional setup.

## 🛑 Notes

- All identifiers (e.g., usernames, personal info) have been anonymized.
- Folder names reflect experiment design (e.g., `28_29` indicates the site geographic identifier).
- Figures and visualizations mentioned in the main text are generated using results in the `results/` folder.
