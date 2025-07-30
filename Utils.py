import os
import numpy as np
import nibabel as nib
import random
import pandas as pd
import matplotlib.pyplot as plt

def load_nifti(path):
    return nib.load(path).get_fdata()

def normalize(volume):
    return (volume - np.mean(volume)) / np.std(volume)

import scipy.ndimage

def resize_volume(volume, target_shape=(128,128,128)):
    # Compute zoom factors for each dimension
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    volume_resized = scipy.ndimage.zoom(volume, zoom=factors, order=1)  # order=1: linear interpolation
    return volume_resized

def preprocess_subject(path):
    subject_id = os.path.basename(path)

    flair = normalize(load_nifti(os.path.join(path, f"{subject_id}-t2f.nii.gz")))
    t1 = normalize(load_nifti(os.path.join(path, f"{subject_id}-t1n.nii.gz")))
    t1ce = normalize(load_nifti(os.path.join(path, f"{subject_id}-t1c.nii.gz")))
    t2 = normalize(load_nifti(os.path.join(path, f"{subject_id}-t2w.nii.gz")))
    seg = load_nifti(os.path.join(path, f"{subject_id}-seg.nii.gz"))

    # Resize each modality
    flair = resize_volume(flair)
    t1 = resize_volume(t1)
    t1ce = resize_volume(t1ce)
    t2 = resize_volume(t2)

    # For segmentation, use order=0 (nearest neighbor) to preserve labels
    seg = scipy.ndimage.zoom(seg,
                             zoom=[128/s for s in seg.shape],
                             order=0).astype(np.uint8)

    # Stack modalities
    image = np.stack([flair, t1, t1ce, t2], axis=-1)

    return image, seg


def data_generator(subject_dirs, batch_size):
    while True:
        random.shuffle(subject_dirs)
        for i in range(0, len(subject_dirs), batch_size):
            batch_images, batch_masks = [], []
            batch_subjects = subject_dirs[i:i+batch_size]
            if len(batch_subjects) == 0:
                continue  # skip empty batch

            for subject_path in batch_subjects:
                image, mask = preprocess_subject(subject_path)
                batch_images.append(image)
                batch_masks.append(mask)

            batch_images = np.array(batch_images)
            batch_masks = np.array(batch_masks)

            if batch_images.size == 0 or batch_masks.size == 0:
                continue  # skip empty batch

            yield batch_images, batch_masks

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
