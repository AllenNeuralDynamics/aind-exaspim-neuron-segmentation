"""
Created on Wed June 28 12:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Denoising routines for 3D microscopy images using patch-based deep learning
inference. Includes functions to extract overlapping patches, normalize and
batch process them through a model on GPU, and stitch denoised patches back
into a full 3D volume.

"""

from tqdm import tqdm

import itertools
import numpy as np
import torch
import waterz

from aind_exaspim_neuron_segmentation.machine_learning.unet3d import UNet
from aind_exaspim_neuron_segmentation.utils import img_util


def predict(
    img,
    model,
    affinity_mode=True,
    batch_size=32,
    normalization_percentiles=(1, 99.5),
    patch_shape=(128, 128, 128),
    overlap=(32, 32, 32),
    trim=8,
    verbose=True
):
    """
    Predicts affinities or foreground–background maps for a 3D image by
    splitting it into overlapping patches, batching the patches, and
    processing each batch with the model.

    Parameters
    ----------
    img : numpy.ndarray
        Input 3D image of shape (1, 1, depth, height, width).
    model : torch.nn.Module
        PyTorch model used for prediction.
    affinity_mode : bool, optional
        If True, the model predicts affinities; if False, it predicts
        foreground–background. Default is True.
    batch_size : int, optional
        Number of patches to process in a batch. Default is 32.
    normalization_percentiles : Tuple[int], optional
        Lower and upper percentiles used for normalization. Default is
        (0.5, 99.9).
    overlap : Tuple[int], optional
        Number of voxels in overlap between patches along each dimension.
        Default is (16, 16, 16).
    patch_shape : Tuple[int], optional
        Shape of the 3D patch expected by the model. Default is (128, 128, 128).
    verbose : bool, optional
        Whether to show a tqdm progress bar. Default is True.

    Returns
    -------
    pred : numpy.ndarray
        Prediction produced by the given model applied to an image.
    """
    # Preprocess image
    img = img_util.normalize(
        img, normalization_percentiles=normalization_percentiles
    )
    while len(img.shape) < 5:
        img = img[np.newaxis, ...]

    # Initializations
    n_channels = 3 if affinity_mode else 1
    n_patches = count_patches(img, patch_shape, overlap)
    starts_generator = generate_patch_starts(img.shape, patch_shape, overlap)
    pred = np.zeros((n_channels,) + img.shape[2:])

    # Main
    pbar = tqdm(total=n_patches, desc="Predict") if verbose else None
    for _ in range(0, n_patches, batch_size):
        # Extract batch and run model
        starts = list(itertools.islice(starts_generator, batch_size))
        patches = _predict_batch(img, model, starts, patch_shape, trim=trim)

        # Store result
        for patch, start in zip(patches, starts):
            start = [max(s + trim, 0) for s in start]
            end = [start[i] + patch.shape[i] for i in range(3)]
            end = [min(e, s) for e, s in zip(end, img.shape[2:])]
            pred[
                0, 0, start[0]:end[0], start[1]:end[1], start[2]:end[2]
            ] = patch[: end[0] - start[0], : end[1] - start[1], : end[2] - start[2]]
        pbar.update(len(starts)) if verbose else None
    return pred


def predict_patch(patch, model):
    """
    Predicts affinities or foreground–background maps for a single 3D image
    patch.

    Parameters
    ----------
    patch : numpy.ndarray
        3D input patch to process.
    model : torch.nn.Module
        PyTorch model used for prediction.

    Returns
    -------
    numpy.ndarray
        Prediction produced by the given model applied to a single patch.
    """
    patch = to_tensor(img_util.normalize(patch))
    with torch.no_grad():
        output_tensor = torch.sigmoid(model(patch))
    return np.array(output_tensor.cpu())


def _predict_batch(img, model, starts, patch_shape, trim=8):
    # Subroutine
    def process_patch(start):
        s = img_util.get_patch_slices(start, patch_shape, img.shape[2:])
        patch = img[(0, 0, *s)]
        return add_padding(patch, patch_shape)

    # Process patches
    inputs = np.empty((len(starts), 1,) + patch_shape, dtype=np.float32)
    for i in range(len(starts)):
        inputs[i, 0, ...] = process_patch(starts[i])

    # Run model
    inputs = to_tensor(inputs)
    with torch.no_grad():
        outputs = torch.sigmoid(model(inputs)).cpu().numpy()
    return outputs[:, 0, trim:-trim, trim:-trim, trim:-trim]


# --- Segmentation and Skeletonization ---
def affinities_to_segmentation(affinities, thresholds=[0.1, 0.2, 0.3]):
    """
    Converts affinity maps into a segmentation using agglomerative watershed.

    Parameters
    ----------
    affinities : numpy.ndarray
        Affinity map with shape (3, H, W, D). where each channel encodes voxel
        affinities along a spatial axis.
    thresholds : List[float], optional
        List of merge thresholds passed to Waterz. Final segmentation is taken
        from the last threshold in the list. Defaults to [0.1, 0.2, 0.3].

    Returns
    -------
    numpy.ndarray
        A segmentation corresponding to the final agglomeration produced by
        Waterz.
    """
    segmentations = waterz.agglomerate(
        affinities,
        thresholds,
        aff_threshold_low=0.1,
        aff_threshold_high=0.99,
    )
    return list(segmentations)[-1]


# --- Helpers ---
def add_padding(patch, patch_shape):
    """
    Pads a 3D patch with zeros to reach the desired patch shape.

    Parameters
    ----------
    patch : numpy.ndarray
        3D array representing the patch to be padded.
    patch_shape : Tuple[int]
        Shape of the 3D patch expected by model.

    Returns
    -------
    numpy.ndarray
        Zero-padded patch with the specified patch shape.
    """
    pad_width = [(0, ps - s) for ps, s in zip(patch_shape, patch.shape)]
    return np.pad(patch, pad_width, mode="constant", constant_values=0)


def count_patches(img, patch_shape, overlap):
    """
    Counts the number of patches within a 3D image for a given patch size
    and overlap between patches.

    Parameters
    ----------
    img : torch.Tensor or numpy.ndarray
        Input image tensor with shape (batch, channels, depth, height, width).
    patch_shape : Tuple[int], optional
        Shape of the 3D patch expected by the model.
    overlap : Tuple[int], optional
        Number of voxels in overlap between patches along each dimension.

    Returns
    -------
    int
        Number of patches.
    """
    stride = tuple(ps - ov for ps, ov in zip(patch_shape, overlap))
    d_range = range(0, img.shape[2] - patch_shape[0] + stride[0], stride[0])
    h_range = range(0, img.shape[3] - patch_shape[1] + stride[1], stride[1])
    w_range = range(0, img.shape[4] - patch_shape[2] + stride[2], stride[2])
    return len(d_range) * len(h_range) * len(w_range)


def load_model(path, affinity_mode=True, device="cuda"):
    """
    Loads a pretrained UNet model from a file.

    Parameters
    ----------
    path : str
        Path to the saved model weights (e.g., .pt or .pth file).
    affinity_mode : bool, optional
        If True, the model predicts affinities; if False, it predicts
        foreground–background. Default is True.
    device : str, optional
        Device to load the model onto. Default is "cuda".

    Returns
    -------
    torch.nn.Module
        UNet model loaded with weights and set to evaluation mode.
    """
    output_channels = 3 if affinity_mode else 1
    model = UNet(output_channels=output_channels)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate_patch_starts(img_shape, patch_shape, overlap):
    """
    Generates starting coordinates for 3D patches extracted from an image
    tensor, based on specified patch size and overlap.

    Parameters
    ----------
    img : torch.Tensor or numpy.ndarray
        Shape of the input image.
    patch_shape : Tuple[int]
        Shape of the 3D patch expected by the model.
    overlap : Tuple[int]
        Number of voxels in overlap between patches along each dimension.

    Returns
    -------
    generator
        Generates starting coordinates for image patches.
    """
    # Compute the range of starting voxel coordinate along each dimension
    assert len(img_shape) == 5, "Image must have shape (1, 1, D, H, W)"
    stride = tuple(ps - o for ps, o in zip(patch_shape, overlap))
    ranges = [
        range(0, d - ps + s, s)
        for d, ps, s in zip(img_shape[2:], patch_shape, stride)
    ]

    # Generate all starting voxel coordinates
    for start in itertools.product(*ranges):
        yield start


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
        Tensor on GPU, with shape (1, 1, D, H, W).
    """
    while (len(arr.shape)) < 5:
        arr = arr[np.newaxis, ...]
    return torch.tensor(arr).to("cuda", dtype=torch.float)
