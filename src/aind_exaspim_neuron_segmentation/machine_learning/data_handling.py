"""
Created on Wed June 25 5:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for loading data during training.

"""

from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import random

from aind_exaspim_neuron_segmentation import inference
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
        affinity_mode=True,
        brightness_clip=300,
        normalization_percentiles=(1, 99.9),
        patch_shape=(96, 96, 96),
    ):
        """
        Instantiates a BaseDataset object.

        Parameters
        ----------
        input_img_paths : List[str]
            Paths to input volumetric images.
        label_mask_paths : List[str]
            Paths to corresponding label masks.
        affinity_mode : bool, optional
            If True, the model predicts affinities; if False, it predicts
            foreground–background. Default is True.
        brightness_clip : float, optional
            Maximum brightness value for voxel intensities. Default is 1000.
        normalization_percentiles, Tuple[int]
            Lower and upper percentiles used for normalization. Default is
            (1, 99.9).
        patch_shape : Tuple[int], optional
            Shape of 3D patches to extract from the images. Default is
            (96, 96, 96).
        """
        # Call parent class
        super().__init__()

        # Instance attributes
        self.input_img_paths = input_img_paths
        self.label_mask_paths = label_mask_paths
        self.affinity_mode = affinity_mode
        self.brightness_clip = brightness_clip
        self.normalization_percentiles = normalization_percentiles
        self.patch_shape = patch_shape

        # Load images
        self.input_imgs = self._load_imgs(input_img_paths)
        self.label_masks = self._load_imgs(label_mask_paths)

    def _load_imgs(self, img_paths):
        """
        Loads a list of 3D images.

        Parameters
        ----------
        img_paths : List[str]
            Paths to images.

        Returns
        -------
        List[numpy.ndarray]
            Loaded images as NumPy arrays.
        """
        imgs = list()
        for img_path in tqdm(img_paths, desc="Loading Images"):
            img = img_util.read(img_path)
            imgs.append(img)
        return imgs

    # --- Read Image Patches ---
    def get_patch(self, img, center):
        """
        Extracts a centered 3D patch from an image.

        Parameters
        ----------
        img : numpy.ndarray
            Full volumetric image.
        center : Tuple[int]
            Center voxel coordinate of patch.

        Returns
        -------
        numpy.ndarray
            Image patch with shape (1, D, H, W).
        """
        patch = img_util.get_patch(img, center, self.patch_shape)
        return np.array(patch[np.newaxis, ...])

    def get_input_patch(self, i, center):
        """
        Gets a normalized input patch from the i-th image.

        Parameters
        ----------
        i : int
            Index of the image in the dataset.
        center : Tuple[int]
            Center voxel coordinate of patch.

        Returns
        -------
        patch : numpy.ndarray
            Normalized input patch with shape (1, D, H, W).
        """
        patch = self.get_patch(self.input_imgs[i], center)
        patch = np.minimum(patch, self.brightness_clip)
        patch = img_util.normalize(
            patch, percentiles=self.normalization_percentiles
        )
        return patch

    def get_label_patch(self, i, center):
        """
        Gets a label patch from the i-th label mask.

        Parameters
        ----------
        i : int
            Index of the label mask in the dataset.
        center : Tuple[int]
            Center voxel coordinate of patch.

        Returns
        -------
        numpy.ndarray
            Label patch with shape (1, D, H, W).
        """
        patch = self.get_patch(self.label_masks[i], center)
        if not self.affinity_mode:
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
        affinity_mode=True,
        brightness_clip=300,
        normalization_percentiles=(1, 99.9),
        patch_shape=(96, 96, 96),
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
        affinity_mode : bool, optional
            If True, the model predicts affinities; if False, it predicts
            foreground–background. Default is True.
        brightness_clip : float, optional
            Maximum brightness value for voxel intensities. Default is 300.
        normalization_percentiles, Tuple[int]
            Lower and upper percentiles used for normalization. Default is
            (1, 99.5).
        patch_shape : Tuple[int], optional
            Shape of the 3D patch to extract from the images. Default is
            (96, 96, 96).
        transform : callable, optional
            A function or callable class for joint image/label augmentation.
        """
        # Call parent class
        super().__init__(
            input_img_paths,
            label_mask_paths,
            affinity_mode=affinity_mode,
            brightness_clip=brightness_clip,
            patch_shape=patch_shape
        )

        # Instance attributes
        self.transform = ImageTransforms() if transform else None
        self.wgts = self.compute_wgts()

    def compute_wgts(self):
        """
        Computes sampling weights based on the number of foreground voxels in
        each label mask.

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
        # Sample patches
        _, input_patch, label_patch = self.sample_patch()
        if self.transform:
            input_patch, label_patch = self.transform(input_patch, label_patch)

        # Check whether to compute affinity channels
        if self.affinity_mode:
            return input_patch, img_util.get_affinity_channels(label_patch[0])
        else:
            return input_patch, label_patch

    def __len__(self):
        """
        Returns a fixed-length dataset for training.

        Returns
        -------
        int
            Number of samples (4x number of label masks).
        """
        return 4 * len(self.label_masks)

    # --- Patch Sampling ---
    def sample_patch(self):
        """
        Randomly samples a patch from the dataset, biased toward either
        foreground or background regions.

        Returns
        -------
        i : int
            Index of the selected image.
        input_patch : numpy.ndarray
            Extracted input patch centered at the sampled location.
        label_patch : numpy.ndarray
            Extracted label patch centered at the sampled location.
        """
        # Search for foreground or background patch
        cnt = 0
        is_foreground = np.random.random() > 0.15
        i = np.random.choice(np.arange(len(self.input_imgs)), p=self.wgts)
        while cnt < 25:
            # Sample patch
            cnt += 1
            center = self.sample_center(self.label_masks[i].shape)
            label_patch = self.get_label_patch(i, center)

            # Check if patch is foreground or background
            foreground_cnt = (label_patch > 0).sum()
            if foreground_cnt > 10**3 and is_foreground:
                break
            elif foreground_cnt < 10**3 and not is_foreground:
                break

        # Get input patch
        input_patch = self.get_input_patch(i, center)
        return i, input_patch, label_patch

    def sample_center(self, shape):
        """
        Samples a patch center within image bounds.

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
        affinity_mode=True,
        brightness_clip=300,
        normalization_percentiles=(1, 99.5),
        patch_shape=(96, 96, 96),
    ):
        """
        Instantiates a ValidateDataset object.

        Parameters
        ----------
        input_img_paths : List[str]
            Paths to input volumetric images.
        label_mask_paths : List[str]
            Paths to corresponding label masks.
        affinity_mode : bool, optional
            If True, the model predicts affinities; if False, it predicts
            foreground–background. Default is True.
        brightness_clip : float, optional
            Maximum brightness value for voxel intensities. Default is 300.
        normalization_percentiles, Tuple[int]
            Lower and upper percentiles used for normalization. Default is
            (1, 99.5).
        patch_shape : Tuple[int], optional
            Shape of the 3D patches to extract from images. Default is
            (96, 96, 96).
        """
        # Call parent class
        super().__init__(
            input_img_paths,
            label_mask_paths,
            affinity_mode=affinity_mode,
            brightness_clip=brightness_clip,
            patch_shape=patch_shape
        )

        # Instance attributes
        self.example_ids = self.generate_examples()

    def generate_examples(self):
        """
        Generates all valid patch centers across all input volumes.

        Returns
        -------
        foreground : List[Tuple[int]]
            List of (image_index, patch_center) tuples for validation patches.
        """
        # Extract foreground/background patches
        foreground, background = list(), list()
        for i in range(len(self.input_imgs)):
            foreground_i, background_i = self.generate_examples_from_img(i)
            foreground.extend(foreground_i)
            background.extend(background_i)

        # Extract examples used for validation
        val_examples = foreground
        n_background_examples = int(len(foreground) * 0.25)
        background = random.sample(background, n_background_examples)
        val_examples.extend(background)
        return val_examples

    def generate_examples_from_img(self, i):
        """
        Generates valid patch centers for a single image volume.

        Parameters
        ----------
        i : int
            Index of the image to sample from.

        Returns
        -------
        foreground : List[numpy.ndarray]
            List of image patches that contain a sufficiently large foreground
            object.
        background : List[numpy.ndarray]
            List of image patches that do not contain a sufficiently large
            foreground object.
        """
        # Generate patch starts
        label_mask = self.label_masks[i]
        patch_starts = inference.generate_patch_starts(
            label_mask.shape, self.patch_shape, (0, 0, 0)
        )

        # Generate examples
        foreground, background = list(), list()
        for v in patch_starts:
            center = [v_i + s_i // 2 for v_i, s_i in zip(v, self.patch_shape)]
            if img_util.is_contained(center, label_mask.shape[2:], buffer=64):
                patch = self.get_patch(label_mask, center)
                if (patch > 0).sum() > 10**3:
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
        input_patch : numpy.ndarray
            Input image patch.
        affs_patch or label_patch : numpy.ndarray
            Affinity maps of label patch if "affinity_mode" is True;
            otherwise, binary mask of foreground-background.
        """
        # Fetch examples
        i, center = self.example_ids[idx]
        input_patch = self.get_input_patch(i, center)
        label_patch = self.get_label_patch(i, center)

        # Check whether to compute affinity channels
        if self.affinity_mode:
            affs_patch = img_util.get_affinity_channels(label_patch[0])
            return input_patch, affs_patch
        else:
            return input_patch, label_patch

    def __len__(self):
        """
        Returns the total number of patches in the validation dataset.

        Returns
        -------
        int
            Number of validation patches.
        """
        return len(self.example_ids)
