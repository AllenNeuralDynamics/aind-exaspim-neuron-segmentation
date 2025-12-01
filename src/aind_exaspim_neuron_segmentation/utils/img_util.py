"""
Created on Wed June 25 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

from fastremap import mask_except, renumber, unique
from matplotlib.colors import ListedColormap

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import s3fs
import tifffile
import zarr

from aind_exaspim_neuron_segmentation.utils import util


# --- Image Reader ---
def read(img_path):
    """
    Reads an image volume from a supported path based on its extension.
    Supported formats:
        - Zarr ('.zarr') from local, GCS, or S3
        - N5 ('.n5') from local or GCS
        - TIFF ('.tif', '.tiff') from local or GCS

    Parameters
    ----------
    img_path : str
        Path to the image. Can be a local or cloud path (gs:// or s3://).

    Returns
    -------
    numpy.ndarray
        Loaded image volume.
    """
    if ".zarr" in img_path:
        return _read_zarr(img_path)
    elif ".n5" in img_path:
        return _read_n5(img_path)
    elif ".tif" in img_path or ".tiff" in img_path:
        return _read_tiff(img_path)
    else:
        raise ValueError(f"Unsupported image format: {img_path}")


def _read_zarr(img_path):
    """
    Reads a Zarr volume from local disk, GCS, or S3.

    Parameters
    ----------
    img_path : str
        Path to a Zarr dataset.

    Returns
    -------
    zarr.hierarchy.Group
        A Zarr group opened in read-only mode.
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(anon=False)
        store = zarr.storage.FSStore(img_path, fs=fs)
    elif _is_s3_path(img_path):
        fs = s3fs.S3FileSystem(config_kwargs={"max_pool_connections": 50})
        store = s3fs.S3Map(root=img_path, s3=fs)
    else:
        store = zarr.DirectoryStore(img_path)
    return zarr.open(store, mode="r")


def _read_n5(img_path):
    """
    Reads an N5 volume from local disk or GCS.

    Parameters
    ----------
    img_path : str
        Path to the N5 directory.

    Returns
    -------
    zarr.hierarchy.Group
        A Zarr group opened in read-only mode.
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(anon=False)
        store = zarr.n5.N5FSStore(img_path, s=fs)
    else:
        store = zarr.n5.N5Store(img_path)
    return zarr.open(store, mode="r")


def _read_tiff(img_path, storage_options=None):
    """
    Reads a TIFF file from local disk or GCS.

    Parameters
    ----------
    img_path : str
        Path to the TIFF file.
    storage_options : dict, optional
        Additional kwargs for GCSFileSystem. Default is None.

    Returns
    -------
    numpy.ndarray
        Image data from the TIFF file.
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(**(storage_options or {}))
        with fs.open(img_path, "rb") as f:
            return tifffile.imread(f)
    else:
        return tifffile.imread(img_path)


def _is_gcs_path(path):
    """
    Checks if the path is a GCS path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is a GCS path.
    """
    return path.startswith("gs://")


def _is_s3_path(path):
    """
    Checks if the path is an S3 path.

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    bool
        Indication of whether the path is an S3 path.
    """
    return path.startswith("s3://")


# --- Compute Affinity Channels ---
def get_affinity_channels(
    label_mask, edges=((1, 0, 0), (0, 1, 0), (0, 0, 1))
):
    """
    Computes affinity channels for a label mask along specified spatial
    offsets.

    Parameters
    ----------
    label_mask : numpy.ndarray
        A 3D integer array where each voxel contains a label ID for its
        corresponding segment.
    edges : Tuple[Tuple[int]], optional
        Offsets (dx, dy, dz) defining the directions along which affinities
        are computed. Default is ((1, 0, 0), (0, 1, 0), (0, 0, 1)).

    Returns
    -------
    affinity_channels : numpy.ndarray
        A 4D array of shape (C, Z, Y, X), where C is the number of affinity
        channels. Each channel is a binary mask indicating voxel affinities
        along that edge direction.
    """
    affinity_channels = np.zeros((3,) + label_mask.shape)
    for i, edge in enumerate(edges):
        affinity_channels[i, ...] = get_affinity_mask(label_mask, edge)
    return affinity_channels


def get_affinity_mask(label_mask, edge):
    """
    Computes affinity mask for label mask based on the given edge affinity
    direction.

    Parameters
    ----------
    label_mask : numpy.ndarray
        Label mask that represents an instance segmentation image.
    edge : Tuple[int]
        Edge affinity direction (e.g (1, 0, 0)).

    Returns
    -------
    aff_mask : numpy.ndarray
        Affinity mask for label mask based on the given edge affinity
        direction.
    """
    # Compute affinity mask
    o1, o2 = get_offset_masks(label_mask, edge)
    aff_mask = (o1 == o2) & (o1 != 0)
    aff_mask = aff_mask.astype(label_mask.dtype)

    # Pad in the axis of the edge only
    axis = edge.index(1)
    pad_width = [(0, 0)] * aff_mask.ndim
    pad_width[axis] = (0, 1)
    aff_mask = np.pad(aff_mask, pad_width, mode="constant", constant_values=0)
    return aff_mask.astype(label_mask.dtype)


def get_offset_masks(label_mask, edge):
    """
    Extracts two subarrays from "label_mask" by using the given edge affinity
    as an offset.

    Parameters
    ----------
    label_mask : torch.Tensor
        Label mask that represents an instance segmentation image.
    edge : Tuple[int]
        Edge affinity direction (e.g (1, 0, 0)).

    Returns
    -------
    offset_mask1 : numpy.ndarray
         Subarray extracted based on the edge affinity.
    offset_mask2 : numpy.ndarray
        Subarray extracted based on the negative of the edge affinity.
    """
    shape = label_mask.shape
    edge = np.array(edge)
    offset1 = np.maximum(edge, 0)
    offset2 = np.maximum(-edge, 0)

    offset_mask1 = label_mask[
        offset1[0]: shape[0] - offset2[0],
        offset1[1]: shape[1] - offset2[1],
        offset1[2]: shape[2] - offset2[2],
    ]
    offset_mask2 = label_mask[
        offset2[0]: shape[0] - offset1[0],
        offset2[1]: shape[1] - offset1[1],
        offset2[2]: shape[2] - offset1[2],
    ]
    return offset_mask1, offset_mask2


# --- Visualization ---
def make_segmentation_colormap(mask, seed=42):
    """
    Creates a matplotlib ListedColormap for a segmentation mask. Ensures label
    0 maps to black and all other labels get distinct random colors.

    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask with integer labels. Assumes label 0 is background.
    seed : int, optional
        Random seed for color reproducibility. Default is 42.

    Returns
    -------
    ListedColormap
        Colormap with black for background and unique colors for other labels.
    """
    n_labels = int(mask.max()) + 1
    rng = np.random.default_rng(seed)
    colors = [(0, 0, 0)]
    colors += list(rng.uniform(0.2, 1.0, size=(n_labels - 1, 3)))
    return ListedColormap(colors)


def plot_mips(img, output_path=None, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input image to generate MIPs from.
    output_path : None or str, optional
        Path to save MIPs as a PNG if provided. Default is None.
    vmax : None or int, optional
        Brightness value used as upper limit of the colormap. Default is None.
    """
    # Initialize plot
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]

    # Plot MIPs
    for i in range(3):
        if len(img.shape) == 5:
            mip = np.max(img[0, 0, ...], axis=i)
        else:
            mip = np.max(img, axis=i)

        axs[i].imshow(mip, vmax=vmax)
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()

    # Save plot (if applicable)
    if output_path:
        plt.savefig(output_path, dpi=200)

    plt.show()
    plt.close(fig)


def plot_segmentation_mips(segmentation, output_path=None):
    """
    Plots maximum intensity projections (MIPs) of a segmentation.

    Parameters
    ----------
    segmentation : numpy.ndarray
        Segmentation whichh can be either:
            - 3D array (Z, Y, X), or
            - 5D array (N, C, Z, Y, X), in which case the first sample
              and first channel are used.
    output_path : None or str, optional
        Path to save MIPs as a PNG if provided. Default is None.
    """
    # Initialize plot
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    cmap = make_segmentation_colormap(segmentation)

    # Plot MIPs
    for i in range(3):
        if len(segmentation.shape) == 5:
            mip = np.max(segmentation[0, 0, ...], axis=i)
        else:
            mip = np.max(segmentation, axis=i)

        axs[i].imshow(mip, cmap=cmap, interpolation="none")
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()

    # Save plot (if applicable)
    if output_path:
        plt.savefig(output_path, dpi=200)

    plt.show()
    plt.close(fig)


# --- Helpers ---
def add_padding(patch, patch_shape):
    """
    Pads a 3D patch with zeros to reach the desired patch shape.

    Parameters
    ----------
    patch : numpy.ndarray
        3D array representing the patch to be padded.
    patch_shape : Tuple[int]
        Shape of the 3D patch expected by a model.

    Returns
    -------
    numpy.ndarray
        Zero-padded patch with the specified patch shape.
    """
    pad_width = [(0, ps - s) for ps, s in zip(patch_shape, patch.shape)]
    return np.pad(patch, pad_width, mode='reflect')


def get_patch(img, center, shape):
    """
    Extracts a patch from an image based on the given voxel coordinate and
    patch shape.

    Parameters
    ----------
    img : ArrayLike
         A Zarr object representing an image.
    center : Tuple[int]
        Center of image patch to be read.
    shape : Tuple[int]
        Shape of image patch to be read.

    Returns
    -------
    numpy.ndarray
        Patch extracted from the given image.
    """
    s = get_slices(center, shape)
    return img[s] if img.ndim == 3 else img[(0, 0, *s)]


def get_patch_slices(start, patch_shape, img_shape):
    """
    Computes slices for a 3D patch within an image, clipped to image
    boundaries.

    Parameters
    ----------
    start : Tuple[int]
        Starting indices (z, y, x) of the patch.
    patch_shape : Tuple[int]
        Desired patch shape (depth, height, width).
    img_shape : Tuple[int]
        Shape of image that the patch is contained within.

    Returns
    -------
    Tuple[slice]
        Slices to index the image: (slice_z, slice_y, slice_x).
    """
    slices = tuple(
        slice(s, min(s + ps, d))
        for s, ps, d in zip(start, patch_shape, img_shape)
    )
    return slices


def get_slices(center, shape):
    """
    Gets the start and end indices of an image patch to be read.

    Parameters
    ----------
    center : Tuple[int]
        Center of image patch to be read.
    shape : Tuple[int]
        Shape of image patch to be read.

    Return
    ------
    Tuple[slice]
        Slice objects used to index into the image.
    """
    start = [c - d // 2 for c, d in zip(center, shape)]
    return tuple(slice(s, s + d) for s, d in zip(start, shape))


def is_contained(voxel, shape, buffer=0):
    """
    Checks whether a voxel is within bounds of a given shape, considering a
    buffer.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinates to be checked.
    shape : tuple of int
        Shape of image volume.
    buffer : int, optional
        Number of voxels to pad the bounds by when checking containment.
        Default 0.

    Returns
    -------
    bool
        True if the voxel is within bounds (with buffer) on all axes, False
        otherwise.
    """
    contained_above = all(0 <= v + buffer < s for v, s in zip(voxel, shape))
    contained_below = all(0 <= v - buffer < s for v, s in zip(voxel, shape))
    return contained_above and contained_below


def list_block_paths(prefix):
    """
    Lists the GCS paths to image blocks associated with a given brain ID.

    Parameters
    ----------
    prefix : str
        Path to directory containing image blocks.

    Returns
    -------
    img_paths : List[str]
        GCS paths (gs://...) to the image blocks.
    """
    img_paths, label_paths = list(), list()
    for block_prefix in util.list_gcs_subprefixes("allen-nd-goog", prefix):
        img_path = util.find_subprefix_with_keyword(
            "allen-nd-goog", block_prefix, "input."
        )
        label_path = util.find_subprefix_with_keyword(
            "allen-nd-goog", block_prefix, "Fill_Label_Mask."
        )
        img_paths.append(f"gs://allen-nd-goog/{img_path}")
        label_paths.append(f"gs://allen-nd-goog/{label_path}")
    return img_paths, label_paths


def normalize(img, apply_clip=True, percentiles=(1, 99.9)):
    """
    Normalizes an image array based on percentile clipping and optionally
    clips the values to the range [0, 1].

    Parameters
    ----------
    img : numpy.ndarray
        Input image array to normalize.
    apply_clip : bool, optional
        Indication of whether to clip image intensities to the range [0, 1].
        Default is True.
    percentiles : Tuple[int], optional
        Lower and upper percentiles used for normalization. Default is
        (1, 99.9).

    Returns
    -------
    numpy.ndarray
        Normalized image with values in [0, 1] if clipping is applied.
    """
    # Normalize image
    mn, mx = np.percentile(img, percentiles)
    img = (img - mn) / (mx - mn + 1e-8)

    # Apply clipping (optional)
    if apply_clip:
        return np.clip(img, 0, 1)
    else:
        return img


def remove_small_segments(label_mask, min_size):
    """
    Removes small segments from a label mask.

    Parameters
    ----------
    label_mask : numpy.ndarray
        Integer array representing a segmentation mask. Each unique
        nonzero value corresponds to a distinct segment.
    min_size : int
        Minimum size (in voxels) for a segment to be kept.

    Returns
    -------
    label_mask : numpy.ndarray
        A new label mask of the same shape as the input, with only
        the retained segments renumbered contiguously. Background
        voxels remain labeled as 0.
    """
    ids, cnts = unique(label_mask, return_counts=True)
    ids = [i for i, cnt in zip(ids, cnts) if cnt > min_size and i != 0]
    ids = mask_except(label_mask, ids)
    label_mask, _ = renumber(ids, preserve_zero=True, in_place=True)
    return label_mask
