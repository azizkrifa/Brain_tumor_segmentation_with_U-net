import os
import numpy as np
import nibabel as nib
import scipy.ndimage
import random
import tensorflow as tf
import shutil

# ---------------------------
# Utilities
# ---------------------------


def create_testset(train_dir, test_dir):

    random.seed(42)

    os.makedirs(test_dir, exist_ok=True)
    # List all subject folders in train
    train_subjects = sorted(os.listdir(train_dir))

    # Randomly select 100 for test
    test_subjects = random.sample(train_subjects, 100)

    # Move them to test
    for subject in test_subjects:
        shutil.move(os.path.join(train_dir, subject), os.path.join(test_dir, subject))
        print(f"Moved subject {subject} to test directory.")
        
    print(f"Moved {len(test_subjects)} subjects to test directory.")


def load_nifti(path):
    return nib.load(path).get_fdata()

def normalize(volume):
    return (volume - np.mean(volume)) / np.std(volume)

def resize_volume(volume, target_shape=(128,128,128), order=1):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return scipy.ndimage.zoom(volume, zoom=factors, order=order)

def preprocess_subject(path, target_shape=(128,128,128)):
    subject_id = os.path.basename(path)

    # Load and normalize modalities
    flair = normalize(load_nifti(os.path.join(path, f"{subject_id}-t2f.nii.gz")))
    t1 = normalize(load_nifti(os.path.join(path, f"{subject_id}-t1n.nii.gz")))
    t1ce = normalize(load_nifti(os.path.join(path, f"{subject_id}-t1c.nii.gz")))
    t2 = normalize(load_nifti(os.path.join(path, f"{subject_id}-t2w.nii.gz")))
    seg = load_nifti(os.path.join(path, f"{subject_id}-seg.nii.gz"))

    # Resize modalities
    flair = resize_volume(flair, target_shape)
    t1 = resize_volume(t1, target_shape)
    t1ce = resize_volume(t1ce, target_shape)
    t2 = resize_volume(t2, target_shape)
    seg = resize_volume(seg, target_shape, order=0).astype(np.uint8)  # nearest neighbor for labels

    # Stack modalities as last axis
    image = np.stack([flair, t1, t1ce, t2], axis=-1)

    return image, seg

# ---------------------------
# Simple Augmentations
# ---------------------------

def augment_image(image, mask):
    # Random flip
    if random.random() > 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
    if random.random() > 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
    # Random rotation 90 degrees
    k = random.randint(0, 3)
    image = np.rot90(image, k=k, axes=(0,1))
    mask = np.rot90(mask, k=k, axes=(0,1))
    return image, mask

# ---------------------------
# Data generator compatible with TensorFlow
# ---------------------------

def data_generator(subject_dirs, batch_size, augment=False, shuffle=False):
    while True:
        if shuffle:
            random.shuffle(subject_dirs)

        for i in range(0, len(subject_dirs), batch_size):
            batch_subjects = subject_dirs[i:i+batch_size]
            batch_images = []
            batch_masks = []

            for path in batch_subjects:
                image, mask = preprocess_subject(path)
                if augment:
                    image, mask = augment_image(image, mask)
                batch_images.append(image)
                batch_masks.append(mask)

            # Convert to numpy arrays with float32 for TF
            batch_images = np.array(batch_images, dtype=np.float32)
            batch_masks = np.array(batch_masks, dtype=np.uint8)

            yield batch_images, batch_masks
