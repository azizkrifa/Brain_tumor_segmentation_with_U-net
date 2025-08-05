import os
import numpy as np
import nibabel as nib
import random
import scipy
import shutil


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