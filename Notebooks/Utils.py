import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import math
import seaborn as sns
from glob import glob
from IPython import get_ipython


def download_data():
    shell = get_ipython().system
    shell("synapse get syn64314352 --version 1")
    shell("synapse get syn60086071 --version 2 ")
    shell("unzip /content/BraTS2024-BraTS-GLI-AdditionalTrainingData.zip -d /content/BraTS2024")
    shell("unzip /content/BraTS2024-BraTS-GLI-TrainingData.zip -d /content/BraTS2024")
    shell("mv /content/BraTS2024/training_data_additional  /content/BraTS2024/val")
    shell("mv /content/BraTS2024/training_data1_v2  /content/BraTS2024/train")


def display_dataset_distribution(train_dir, val_dir, test_dir):
   
    train_subjects_num = len(sorted(glob(os.path.join(train_dir, "*"))))
    val_subjects_num = len(sorted(glob(os.path.join(val_dir, "*"))))
    test_subjects_num = len(sorted(glob(os.path.join(test_dir, "*"))))

    # Create DataFrame in long format
    df = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "Count": [train_subjects_num, val_subjects_num, test_subjects_num]
    })

    df = df.sort_values(by='Count', ascending=False)

    # Plot
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x='Dataset', y='Count', palette='viridis')
    plt.title('Sample Count per Dataset')

    # Add count labels inside bars
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height / 2,
            f'{int(height)}',
            ha='center', va='center',
            color='white', fontsize=12
        )

    plt.tight_layout()
    plt.show()


def plot_all_modalities(folder_path):

    # Define slice indices to visualize
    slices = list(range(0, 182, 10))
    n_slices = len(slices)
    cols = len(slices)
    rows = 4  # One row for each modality

    # Initialize variables for each modality
    t1n = t1c = t2f = t2w = None

    # Load modalities
    for file in os.listdir(folder_path):
        if file.endswith(".nii.gz") and "seg" not in file.lower() :
            file_path = os.path.join(folder_path, file)
            if "t1n" in file.lower():
                t1n = nib.load(file_path).get_fdata()
            elif "t1c" in file.lower():
                t1c = nib.load(file_path).get_fdata()
            elif "t2f" in file.lower():
                t2f = nib.load(file_path).get_fdata()
            elif "t2w" in file.lower():
                t2w = nib.load(file_path).get_fdata()
            
            print(f"Loaded {file} with shape {nib.load(file_path).get_fdata().shape}")

    # Plotting
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=600)
    modalities = [("T1n", t1n), ("T1c", t1c), ("T2f", t2f), ("T2w", t2w)]

    for row_idx, (label, volume) in enumerate(modalities):
        for col_idx, slice_idx in enumerate(slices):
            ax = axes[row_idx, col_idx]
            ax.imshow(volume[:, :, slice_idx], cmap='magma')
            if row_idx == 0:
                ax.set_title(f"Slice {slice_idx}", fontsize=20)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=20)
            ax.axis('off')

    fig.suptitle("MRI Modalities on the Same Patient", fontsize=30)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_segmentation(seg_path):

    volume=nib.load(seg_path).get_fdata()
    print(f"Loaded {seg_path[45:]} with shape {volume.shape}")

    # Define fixed colors for labels 0–3
    cmap = ListedColormap(['black', 'gray', 'lightgreen', 'red'])

    # Define label names
    labels = {
        0: "Background/Rest of the brain",
        1: "Necrotic/Non-enhancing Tumor",
        2: "Edema",
        3: "Enhancing Tumor"
    }

    patches = [mpatches.Patch(color=cmap.colors[i], label=labels[i]) for i in range(len(labels))]

    slices = list(range(0, 182, 10))
    n_slices = len(slices)

    # Define grid size (e.g., 3 rows x 6 columns)
    cols = 6
    rows = math.ceil(n_slices / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows),dpi=600)
    axes = axes.flatten()

    for ax, slice_idx in zip(axes, slices):
        ax.imshow(volume[:, :, slice_idx], cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f"Slice {slice_idx}")
        ax.axis('off')

    # Hide any unused subplots if number of slices < rows*cols
    for ax in axes[n_slices:]:
        ax.axis('off')

    # Add one legend for the whole figure
    fig.legend(handles=patches, bbox_to_anchor=(0.99, 0.2), fontsize='x-large')

    fig.suptitle("Segmentation Mask for a Patient", fontsize=20)

    plt.tight_layout()
    plt.show()


def display_dataset_distribution(dataset_dir):

    train_subjects_num = len(sorted(glob(os.path.join(dataset_dir, "train", "*"))))
    val_subjects_num = len(sorted(glob(os.path.join(dataset_dir, "val", "*"))))
    test_subjects_num = len(sorted(glob(os.path.join(dataset_dir, "test", "*"))))

    # Create DataFrame in long format
    df = pd.DataFrame({
        "Dataset": ["Train", "Validation", "Test"],
        "Count": [train_subjects_num, val_subjects_num, test_subjects_num]
    })

    df = df.sort_values(by='Count', ascending=False)

    # Plot
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x='Dataset', y='Count', palette='viridis')
    plt.title('Sample Count per Dataset')

    # Add count labels inside bars
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height / 2,
            f'{int(height)}',
            ha='center', va='center',
            color='white', fontsize=12
        )

    plt.tight_layout()
    plt.show()


def plot_training_history():

    # Load the saved history from file
    history = pd.read_csv("Outputs/training_log.csv")

    plt.figure(figsize=(14, 5))

    # Plot multiclass_dice_coefficient
    plt.subplot(1, 2, 1)
    plt.plot(history['multiclass_dice_coefficient'], label='Train multiclass_dice_coefficient', marker='o')
    plt.plot(history['val_multiclass_dice_coefficient'], label='Val multiclass_dice_coefficient', marker='o')
    plt.title('Training vs Validation multiclass_dice_coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('multiclass_dice_coefficient')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
