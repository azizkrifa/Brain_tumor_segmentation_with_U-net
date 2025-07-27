# 🧠 Brain Tumor Segmentation with U-Net

Accurate segmentation of brain tumors, especially `gliomas`, is vital for diagnosis and treatment planning. The `BraTS 2024` dataset provides multi-modal MRI scans with expert annotations for tumor subregions.This project uses a `U-Net model` to automatically segment brain tumors from the BraTS 2024 MRI data. U-Net’s architecture effectively captures tumor features to produce precise tumor masks, aiding clinical analysis.

-----

## 1. 📁 Dataset

 ### 1.1 MRI Modalities and Segmentation Labels

  - The BraTS 2024 dataset includes `four MRI modalities` for each patient, providing complementary information to better identify tumor regions:
    
    - **T1-weighted (T1):** Provides detailed `anatomical structure` of the brain.  
    - **T1-contrast enhanced (T1c):** Highlights areas with a disrupted blood-brain barrier, such as `enhancing tumors`.  
    - **T2-weighted (T2w):** Useful for visualizing `edema` and `tumor boundaries`.  
    - **T2f or FLAIR (Fluid-Attenuated Inversion Recovery):** Suppresses fluid signals, making `edema` and `lesions` more visible.

<table align="center" >
  <tr>
    <td colspan="2" align="center">
      <h3>MRI Tests on the Same Patient X</h3>
    </td>
  </tr>

  <tr>
    <td align="center"><b>T1n</b></td>
    <td align="center"><b>T1c</b></td>
  </tr>
  <tr>
    <td><img width="600" height="400" src="https://github.com/user-attachments/assets/a442156b-c73e-4b4e-a279-7257d8ac633d" /></td>
    <td><img width="600" height="400" src="https://github.com/user-attachments/assets/bc7c7471-084f-4347-b3c4-2c27284a7333" /></td>
  </tr>
  <tr>
    <td align="center"><b>T2f</b></td>
    <td align="center"><b>T2w</b></td>
  </tr>
  <tr>
    <td><img width="600" height="400" src="https://github.com/user-attachments/assets/b27657c9-327d-49ae-88c9-06944b982e9e" /></td>
    <td><img width="600" height="400" src="https://github.com/user-attachments/assets/b02966ed-05f7-45ba-97fc-fe25992c8a32" /></td>
  </tr>
</table>

  - Along with these MRI scans, **segmentation masks (seg)** are provided. These masks label each voxel `(3D pixel)` as one of the following classes:
    
    - **0:** Background (non-tumor tissue)  
    - **1:** Necrotic and non-enhancing tumor core  
    - **2:** Peritumoral edema  
    - **3:** Enhancing tumor
    
    This multi-modal data enables the model to learn robust features across different tissue contrasts to accurately segment the tumor subregions.

  ### 1.2 Patient MRI Scan and Segmentation Dimensions

  - Each patient's MRI scan in the BraTS 2024 dataset has a fixed 3D volume size of **(182,218,182)** representing:
    
    - **182 × 218** pixels per slice (height × width), capturing the spatial resolution of each 2D MRI slice.
    - **182** slices in the axial direction, representing the depth or number of cross-sectional images stacked to form the full 3D volume.
    
  - This 3D shape allows us to analyze the brain’s structure slice-by-slice while preserving the volumetric context needed for accurate tumor segmentation.
    
    Below is an example visualization of the segmentation mask for patient X across some slices:
    
  <p align="center">
    <img width="7140" height="4734" alt="seg 02415" src="https://github.com/user-attachments/assets/cc047f52-11ce-4239-851d-d7cef5c5b628" />
  </p>

 ### 1.3 Dataset Structure|Ditribution
  **Note**: The test set was created by `randomly selecting 100 samples` from the original `training se`t to evaluate the model on unseen data while preserving label distribution.
  <p align="center">
    <img width="590" height="390" alt="Sans titre" src="https://github.com/user-attachments/assets/ba8972c5-f62b-445e-8d99-8bdbc90d072d" />
  </p>
  
```bash
  BraTS 2024 dataset/
├── train/
│   ├── BraTS-GLI-00001-000/
│   │   ├── BraTS-GLI-00001-000_t2f.nii.gz
│   │   ├── BraTS-GLI-00001-000_t1n.nii.gz
│   │   ├── BraTS-GLI-00001-000_t1c.nii.gz
│   │   ├── BraTS-GLI-00001-000_t2w.nii.gz
│   │   └── BraTS-GLI-00001-000_seg.nii.gz
│   ├── BraTS-GLI-00002-000/
│   │   └── ...
│   └── ...

├── val/
│   ├── BraTS-GLI-00023-000/
│   │   ├── ... (same structure as train)
│   └── ...

├── test/
│   ├── BraTS-GLI-00041-000/
│   │   ├── BraTS-GLI-00041-000_t2f.nii.gz
│   │   ├── BraTS-GLI-00041-000_t1n.nii.gz
│   │   ├── BraTS-GLI-00041-000_t1c.nii.gz
│   │   ├── BraTS-GLI-00041-000_t2w.nii.gz
│   │   └── BraTS-GLI-00041-000_seg.nii.gz
│   └── ...
```

  
  ### 1.4 Data Preprocessing
  
  To ensure consistency across all MRI scans and prepare the data for model training, we applied the following preprocessing steps to each subject:
  
  #### 1.4.1 Loading NIfTI Files  
  Each subject includes five volumes:  
  - T1 (T1n)  
  - T1 contrast-enhanced (T1c)  
  - T2-weighted (T2w)  
  - FLAIR (T2f)  
  - Segmentation mask  
  
  These files are loaded using the `nibabel` library, which reads `.nii.gz` medical imaging files and returns 3D NumPy arrays.
  
  #### 1.4.2 Normalization  
  For each modality (T1, T1c, T2, FLAIR), we apply `z-score normalization`:  

  <p align="center">
  <img width="299" height="70" alt="image" src="https://github.com/user-attachments/assets/be72d840-f8a7-47c5-ba98-9ba98edf2575" />
  </p>

  where $\mu$ and $\sigma$ are the mean and standard deviation of the image volume. This helps the model converge faster by standardizing intensity values.
  
  #### 1.4.3 Resizing Volumes  
   All modalities are resized from `240×240×155` to a fixed shape of `128×128×128` using linear interpolation (`order=1`). This standard shape ensures uniformity for training in 3D CNNs.
  
  #### 1.4.4 Resizing Segmentation Masks  
   Segmentation masks are resized using nearest-neighbor interpolation (`order=0`) to preserve discrete class labels (e.g., tumor regions). The output mask is cast to `uint8` type.
  
  #### 1.4.5 Multi-modal Stacking  
  The normalized and resized volumes from each modality are stacked along the last axis, resulting in a single 4D volume with shape `128×128×128×4`. This format allows the model to learn from all four modalities simultaneously.





