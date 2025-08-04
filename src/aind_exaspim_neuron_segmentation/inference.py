"""
Created on Wed June 28 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Denoising routines for 3D microscopy images using patch-based deep learning
inference. Includes functions to extract overlapping patches, normalize and
batch process them through a model on GPU, and stitch denoised patches back
into a full 3D volume.

"""

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from tqdm import tqdm

import itertools
import numpy as np
import torch
import waterz

from aind_exaspim_neuron_segmentation.utils import img_util


def predict(
    img,
    model,
    batch_size=32,
    patch_size=96,
    overlap=16,
    trim=8,
    verbose=True
):
    """
    Denoises a 3D image by processing patches in batches and running deep
    learning model.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image of shape (1, 1, depth, height, width).
    model : torch.nn.Module
        PyTorch model to perform prediction on patches.
    batch_size : int, optional
        Number of patches to process in a batch. Default is 32.
    patch_size : int, optional
        Size of the cubic patch extracted from the image. Default is 64.
    overlap : int, optional
        Number of voxels to overlap between patches. Default is 16.
    verbose : bool, optional
        Whether to show a tqdm progress bar. Default is True.

    Returns
    -------
    coords : List[Tuple[int]]
        List of (i, j, k) starting coordinates of patches processed.
    preds : List[numpy.ndarray]
        List of predicted patches (3D arrays) matching the patch size.
    """
    # Adjust image dimenions
    while len(img.shape) < 5:
        img = img[np.newaxis, ...]

    # Initializations
    starts_generator = generate_patch_starts(img, patch_size, overlap)
    n_starts = count_patches(img, patch_size, overlap)
    img = img_util.normalize(np.clip(img, 0, 1000))

    # Main
    pbar = tqdm(total=n_starts, desc="Segment") if verbose else None
    segmentation = np.zeros_like(img, dtype=np.float32)
    for i in range(0, n_starts, batch_size):
        # Run model
        starts = list(itertools.islice(starts_generator, batch_size))
        patches = _predict_batch(img, model, starts, patch_size, trim)

        # Store result
        for patch, start in zip(patches, starts):
            start = [max(s + trim, 0) for s in start]
            end = [start[i] + patch.shape[i] for i in range(3)]
            end = [min(e, s) for e, s in zip(end, img.shape[2:])]
            segmentation[
                0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]
            ] = patch[: end[0] - start[0], : end[1] - start[1], : end[2] - start[2]]
        pbar.update(len(starts)) if verbose else None
    return segmentation


def predict_patch(patch, model):
    """
    Denoised a single 3D patch using the provided model.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model used for prediction.
    patch : numpy.ndarray
        3D input patch to denoise.

    Returns
    -------
    numpy.ndarray
        Denoised 3D patch with the same shape as input patch.
    """
    patch = to_tensor(img_util.normalize(patch))
    with torch.no_grad():
        output_tensor = torch.sigmoid(model(patch))
    return np.array(output_tensor.cpu())


def _predict_batch(img, model, starts, patch_size, trim=6):
    # Subroutine
    def process_patch(i):
        start = starts[i]
        end = [min(s + patch_size, d) for s, d in zip(start, (D, H, W))]
        patch = img[0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        return add_padding(patch, patch_size)

    # Process patches
    D, H, W = img.shape[2:]
    inputs = np.empty((len(starts),) + (patch_size,) * 3, dtype=np.float32)
    for i in range(len(starts)):
        inputs[i, ...] = process_patch(i)

    # Run model
    inputs = batch_to_tensor(inputs)
    with torch.no_grad():
        outputs = torch.sigmoid(model(inputs)).cpu().numpy()
    return outputs[:, 0, trim:-trim, trim:-trim, trim:-trim]


def run_watershed(pred):
    # Distance transform
    img = pred > max(threshold_otsu(pred), 0.4)
    distance = ndi.distance_transform_edt(img)

    # Find local maxima to use as markers
    local_maxi = peak_local_max(distance, labels=img, exclude_border=False)

    # Create marker array
    markers = np.zeros_like(img, dtype=int)
    for i, coord in enumerate(local_maxi, start=1):
        markers[tuple(coord)] = i

    # Run watershed
    return watershed(-distance, markers, mask=img)


def run_agglomerative_watershed(pred, thresholds=[0.1, 0.2, 0.3]):
    # Compute foregroun mask
    binary_mask = pred > min(max(threshold_otsu(pred), 0.4), 0.6)

    # Prepare agglomeration input
    restricted_pred = pred.copy()
    restricted_pred[binary_mask == 0] = 0
    pseudo_affs = np.stack(3 * [restricted_pred[0, 0, ...]], axis=0)

    # Agglomeration
    segmentations = waterz.agglomerate(
        pseudo_affs,
        thresholds,
        aff_threshold_low=0.1,
        aff_threshold_high=0.99,
    )
    return list(segmentations)[-1]


# --- Helpers ---
def add_padding(patch, patch_size):
    """
    Pads a 3D patch with zeros to reach the desired patch shape.

    Parameters
    ----------
    patch : numpy.ndarray
        3D array representing the patch to be padded.
    patch_size : int
        Target size for each dimension after padding.

    Returns
    -------
    numpy.ndarray
        Zero-padded patch with shape (patch_size, patch_size, patch_size).
    """
    pad_width = [
        (0, patch_size - patch.shape[0]),
        (0, patch_size - patch.shape[1]),
        (0, patch_size - patch.shape[2]),
    ]
    return np.pad(patch, pad_width, mode="constant", constant_values=0)


def batch_to_tensor(arr):
    """
    Converts a NumPy array containing a batch of inputs to a PyTorch tensor
    and moves it to the GPU.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted, with shape (batch_size, depth, height, width).

    Returns
    -------
    torch.Tensor
        Tensor on GPU, with shape (batch_size, 1, depth, height, width).
    """
    return to_tensor(arr[:, np.newaxis, ...])


def count_patches(img, patch_size, overlap):
    """
    Counts the number of patches within a 3D image for a given patch size
    and overlap between patches.

    Parameters
    ----------
    img : torch.Tensor or numpy.ndarray
        Input image tensor with shape (batch, channels, depth, height, width).
    patch_size : int
        Size of each cubic patch along each spatial dimension.
    overlap : int
        Number of voxels that adjacent patches overlap.

    Returns
    -------
    int
        Number of patches.
    """
    stride = patch_size - overlap
    d_range = range(0, img.shape[2] - patch_size + stride, stride)
    h_range = range(0, img.shape[3] - patch_size + stride, stride)
    w_range = range(0, img.shape[4] - patch_size + stride, stride)
    return len(d_range) * len(h_range) * len(w_range)


def generate_patch_starts(img, patch_size, overlap):
    """
    Generates starting coordinates for 3D patches extracted from an image
    tensor, based on specified patch size and overlap.

    Parameters
    ----------
    img : torch.Tensor or numpy.ndarray
        Input image tensor with shape (batch, channels, depth, height, width).
    patch_size : int
        The size of each cubic patch along each spatial dimension.
    overlap : int
        Number of voxels that adjacent patches overlap.

    Returns
    -------
    generator
        Generates starting coordinates for image patches.
    """
    stride = patch_size - overlap
    for i in range(0, img.shape[2] - patch_size + stride, stride):
        for j in range(0, img.shape[3] - patch_size + stride, stride):
            for k in range(0, img.shape[4] - patch_size + stride, stride):
                yield (i, j, k)


def to_tensor(arr):
    """
    Converts a NumPy array containing to a PyTorch tensor and moves it to the
    GPU.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to be converted.

    Returns
    -------
    torch.Tensor
        Tensor on GPU, with shape (1, 1, depth, height, width).
    """
    while (len(arr.shape)) < 5:
        arr = arr[np.newaxis, ...]
    return torch.tensor(arr).to("cuda", dtype=torch.float)
