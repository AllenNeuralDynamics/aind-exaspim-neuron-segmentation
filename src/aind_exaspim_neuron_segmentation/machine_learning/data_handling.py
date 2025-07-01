"""
Created on Wed June 25 5:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data during training and inference.

"""

from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import random

from aind_exaspim_neuron_segmentation.machine_learning.augmentation import ImageTransforms
from aind_exaspim_neuron_segmentation.utils import img_util


class BaseDataset(Dataset):
    """
    Base dataset class for 3D volumetric patch sampling.
    """
    def __init__(
        self,
        input_img_paths,
        label_mask_paths,
        is_instance_segmentation=True,
        patch_shape=(128, 128, 128),
    ):
        """
        Instantiates a BaseDataset object.

        Parameters
        ----------
        input_img_paths : List[str]
            Paths to input volumetric images.
        label_mask_paths : List[str]
            Paths to corresponding label masks.
        is_instance_segmentation : bool, optional
            Indication of whether the task is instance segmentation. Default
            is True.
        patch_shape : Tuple[int], optional
            Shape of the 3D patch to extract from the images. Default is
            (128, 128, 128).

        Returns
        -------
        None
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.input_img_paths = input_img_paths
        self.label_mask_paths = label_mask_paths
        self.is_instance_segmentation = is_instance_segmentation
        self.patch_shape = patch_shape

        # Load images
        self.input_imgs = self.load_imgs(input_img_paths, is_inputs=True)
        self.label_masks = self.load_imgs(label_mask_paths, is_inputs=False)

    def load_imgs(self, img_paths, is_inputs=True):
        """
        Load a list of volumetric images.

        Parameters
        ----------
        img_paths : List[str]
            Paths to images.
        is_inputs : bool, optional
            If True, prints 'Inputs' during progress. Otherwise prints
            'Label Masks'.

        Returns
        -------
        List[numpy.ndarray]
            Image volumes as NumPy arrays.
        """
        imgs = list()
        desc = "Inputs" if is_inputs else "Label Masks"
        for img_path in tqdm(img_paths, desc=desc):
            img = img_util.read(img_path)
            imgs.append(img)
        return imgs

    # --- Read Image Patches ---
    def get_patch(self, img, center):
        """
        Extract a centered 3D patch from an image.

        Parameters
        ----------
        img : np.ndarray
            Full volumetric image.
        center : Tuple[int]
            Center voxel coordinate of patch.

        Returns
        -------
        np.ndarray
            Image patch with shape (1, D, H, W).
        """
        patch = img_util.get_patch(img, center, self.patch_shape)
        return np.array(patch[np.newaxis, ...])

    def get_input_patch(self, i, center):
        """
        Get a normalized input patch from the i-th image.

        Parameters
        ----------
        i : int
            Index of the image in the dataset.
        center : Tuple[int]
            Center voxel coordinate of patch.

        Returns
        -------
        np.ndarray
            Normalized input patch with shape (1, D, H, W).
        """
        patch = self.get_patch(self.input_imgs[i], center)
        return img_util.normalize(patch)

    def get_label_patch(self, i, center):
        """
        Get a label patch from the i-th label mask.

        Parameters
        ----------
        i : int
            Index of the label mask in the dataset.
        center : Tuple[int]
            Center voxel coordinate of patch.

        Returns
        -------
        np.ndarray
            Label patch with shape (1, D, H, W).
        """
        patch = self.get_patch(self.label_masks[i], center)
        if not self.is_instance_segmentation:
            patch = (patch > 0).astype(int)
        return patch


class TrainDataset(BaseDataset):
    """
    PyTorch dataset for training segmentation models.
    """
    def __init__(
        self,
        input_img_paths,
        label_mask_paths,
        is_instance_segmentation=True,
        patch_shape=(128, 128, 128),
        transform=None
    ):
        """
        Instantiates a TrainDataset object.

        Parameters
        ----------
        input_img_paths : List[str]
            Paths to input volumetric images.
        label_mask_paths : List[str]
            Paths to corresponding label masks.
        is_instance_segmentation : bool, optional
            Indication of whether the task is instance segmentation. Default
            is True.
        patch_shape : Tuple[int], optional
            Shape of the 3D patch to extract from the images. Default is
            (128, 128, 128).
        transform : callable, optional
            A function or callable class for joint image/label augmentation.

        Returns
        -------
        None
        """
        # Call parent class
        super().__init__(
            input_img_paths,
            label_mask_paths,
            is_instance_segmentation=is_instance_segmentation,
            patch_shape=patch_shape
        )

        # Instance attributes
        self.transform = ImageTransforms() if transform else None
        self.wgts = self.compute_wgts()

    def compute_wgts(self):
        """
        Compute sampling weights based on the number of foreground voxels in
        each label mask.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            Normalized weights proportional to foreground pixel counts.
        """
        wgts = list()
        for label_mask in self.label_masks:
            wgts.append((label_mask[:] > 0).sum())
        return wgts / np.sum(wgts)

    # --- Built-In Routines ---
    def __getitem__(self, dummy_input):
        """
        Samples image-label patch pair with optional augmentation.

        Parameters
        ----------
        dummy_input : int
            Unused; included for compatibility with PyTorch's Dataset API.

        Returns
        -------
        Tuple[numpy.ndarray]
            Transformed (or raw) input and label patches as NumPy arrays.
        """
        _, input_patch, label_patch = self.sample_patch()
        if self.transform:
            return self.transform(input_patch, label_patch)
        else:
            return input_patch.copy(), label_patch.copy()

    def __len__(self):
        """
        Return a fixed-length dataset for training.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of samples (5x number of label masks).
        """
        return 4 * len(self.label_masks)

    # --- Patch Sampling ---
    def sample_patch(self):
        """
        Sample a random image patch.

        Parameters
        ----------
        None

        Returns
        -------
        Tuple
            (image index, input patch, label patch), where patches are NumPy
            arrays.
        """
        try:
            # Search for foreground or background patch
            cnt = 0
            is_foreground = np.random.random() > 0.1
            i = np.random.choice(np.arange(len(self.input_imgs)), p=self.wgts)
            while cnt < 25:
                cnt += 1
                center = self.sample_center(self.label_masks[i].shape)
                label_patch = self.get_label_patch(i, center)
                foreground_cnt = (label_patch > 0).sum()
                if foreground_cnt > 5000 and is_foreground:
                    break
                elif foreground_cnt == 0 and not is_foreground:
                    break

            # Reformat patches
            input_patch = self.get_input_patch(i, center)
        except Exception as e:
            print("Exception:", e)
            print(self.label_masks[i].shape, self.input_imgs[i].shape, center)
        return i, input_patch, label_patch

    def sample_center(self, shape):
        """
        Randomly select a patch center within image bounds.

        Parameters
        ----------
        shape : tuple of int
            Shape of the full 3D label/image volume.

        Returns
        -------
        numpy.ndarray
            Coordinate of the center point for a patch.
        """
        idxs = range(3) if len(shape) == 3 else range(2, 5)
        upper = [shape[i] - s // 2 for i, s in zip(idxs, self.patch_shape)]
        lower = [s // 2 for s in self.patch_shape]
        return np.array([random.randint(l, u) for l, u in zip(lower, upper)])


class ValidateDataset(BaseDataset):
    """
    PyTorch dataset for validating segmentation models with deterministically
    sampled 3D patches.
    """
    def __init__(
        self,
        input_img_paths,
        label_mask_paths,
        is_instance_segmentation=True,
        patch_shape=(128, 128, 128),
    ):
        """
        Parameters
        ----------
        input_img_paths : List[str]
            Paths to input volumetric images.
        label_mask_paths : List[str]
            Paths to corresponding label masks.
        is_instance_segmentation : bool, optional
            Indication of whether the task is instance segmentation. Default is
            True.
        patch_shape : Tuple[int], optional
            Shape of the 3D patch to extract from the images. Default is
            (128, 128, 128).

        Returns
        -------
        None
        """
        # Call parent class
        super().__init__(
            input_img_paths,
            label_mask_paths,
            is_instance_segmentation=is_instance_segmentation,
            patch_shape=patch_shape
        )

        # Instance attributes
        self.example_ids = self.generate_examples()

    def generate_examples(self):
        """
        Generate all valid patch centers across all input volumes.

        Parameters
        ----------
        None

        Returns
        -------
        List[tuple]
            List of (image_index, patch_center) tuples for validation patches.
        """
        # Extract foreground/background patches
        foreground, background = list(), list()
        for i in range(len(self.input_imgs)):
            foreground_i, background_i = self.generate_examples_from_img(i)
            foreground.extend(foreground_i)
            background.extend(background_i)

        # Extract examples
        n_background_examples = int(len(foreground) * 0.1)
        background = random.sample(background, n_background_examples)
        foreground.extend(background)
        return foreground

    def generate_examples_from_img(self, i):
        """
        Generate valid patch centers for a single image volume.

        Parameters
        ----------
        i : int
            Index of the image to sample from.

        Returns
        -------
        List[tuple]
            List of (image_index, patch_center) tuples.
        """
        foreground, background = list(), list()
        label_mask = self.label_masks[i]
        for v in img_util.calculate_offsets(label_mask, self.patch_shape):
            center = [v_i + s_i // 2 for v_i, s_i in zip(v, self.patch_shape)]
            patch = self.get_patch(label_mask, center)
            if (patch > 0).sum() > 5000:
                foreground.append((i, center))
            else:
                background.append((i, center))
        return foreground, background

    def __getitem__(self, idx):
        """
        Fetches a single patch from the validation set.

        Parameters
        ----------
        idx : int
            Index of the patch to retrieve.

        Returns
        -------
        tuple
            A tuple (input_patch, label_patch) where both are NumPy arrays.
        """
        i, center = self.example_ids[idx]
        input_patch = self.get_input_patch(i, center)
        label_patch = self.get_label_patch(i, center)
        return input_patch.copy(), label_patch.copy()

    def __len__(self):
        """
        Return the total number of patches in the validation dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of precomputed validation patches.
        """
        return len(self.example_ids)
