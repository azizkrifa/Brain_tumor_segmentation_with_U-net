# 🧠 Brain Tumor Segmentation with U-Net

Accurate segmentation of brain tumors, especially `gliomas`, is vital for diagnosis and treatment planning. The `BraTS 2023` dataset provides multi-modal MRI scans with expert annotations for tumor subregions.This project uses a `U-Net model` to automatically segment brain tumors from the BraTS 2023 MRI data. U-Net’s architecture effectively captures tumor features to produce precise tumor masks, aiding clinical analysis.

-----

## 📁 Dataset

- ### MRI Modalities and Segmentation Labels

  - The BraTS 2023 dataset includes `four MRI modalities` for each patient, providing complementary information to better identify tumor regions:
    
    - **T1-weighted (T1):** Provides detailed `anatomical structure` of the brain.  
    - **T1-contrast enhanced (T1c):** Highlights areas with a disrupted blood-brain barrier, such as `enhancing tumors`.  
    - **T2-weighted (T2w):** Useful for visualizing `edema` and `tumor boundaries`.  
    - **T2f or FLAIR (Fluid-Attenuated Inversion Recovery):** Suppresses fluid signals, making `edema` and `lesions` more visible.
   


<table align="center" >
  <tr>
    <td colspan="2" align="center">
      <h3>MRI Tests on the Same Patient</h3>
    </td>
  </tr>

  <tr>
    <td align="center"><b>T1n</b></td>
    <td align="center"><b>T1c</b></td>
  </tr>
  <tr>
    <td><img width="600" height="3558" src="https://github.com/user-attachments/assets/93bd3548-af49-47c6-b8cf-881b033030dc" width="400"/></td>
    <td><img width="600" height="3558" src="https://github.com/user-attachments/assets/838a8056-48f7-4d56-b3d6-593fe11362ad" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><b>T2f</b></td>
    <td align="center"><b>T2w</b></td>
  </tr>
  <tr>
    <td><img width="600" height="3558" src="https://github.com/user-attachments/assets/4b4f825a-c735-4de1-8dd4-3efb4b1d7ebb" width="400"/></td>
    <td><img width="600" height="3558" src="https://github.com/user-attachments/assets/57224994-f811-4102-bfa2-804cb9892012" width="400"/></td>
  </tr>
</table>

  - Along with these MRI scans, **segmentation masks (seg)** are provided. These masks label each voxel `(pixel)` as one of the following classes:
    
    - **0:** Background (non-tumor tissue)  
    - **1:** Necrotic and non-enhancing tumor core  
    - **2:** Peritumoral edema  
    - **3:** Enhancing tumor
    
    This multi-modal data enables the model to learn robust features across different tissue contrasts to accurately segment the tumor subregions.

- ### Patient MRI Scan and Segmentation Dimensions

  - Each patient's MRI scan in the BraTS 2023 dataset has a fixed 3D volume size of **(240, 240, 155)**, representing:
    
    - **240 × 240** pixels per slice (height × width), capturing the spatial resolution of each 2D MRI slice.
    - **155** slices in the axial direction, representing the depth or number of cross-sectional images stacked to form the full 3D volume.
    
  - This 3D shape allows us to analyze the brain’s structure slice-by-slice while preserving the volumetric context needed for accurate tumor segmentation.
    
    Below is an example visualization of the segmentation mask for a patient across some slices:
    
  <p align="center">
    <img width="7034" height="3558" alt="seg_0002" src="https://github.com/user-attachments/assets/77764870-b251-4d58-9a71-520dc7830fa1" />
  </p>




