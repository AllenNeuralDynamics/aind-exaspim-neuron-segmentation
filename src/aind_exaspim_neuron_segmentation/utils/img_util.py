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
    Read an image volume from a supported path based on its extension.

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
    np.ndarray
        Loaded image volume as a NumPy array.
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
    Read a Zarr volume from local disk, GCS, or S3.

    Parameters
    ----------
    img_path : str
        Path to the Zarr directory.

    Returns
    -------
    np.ndarray
        First array in the Zarr group.
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(anon=False)
        store = zarr.storage.FSStore(img_path, fs=fs)
    elif _is_s3_path(img_path):
        fs = s3fs.S3FileSystem(config_kwargs={"max_pool_connections": 50})
        store = s3fs.S3Map(root=img_path, s3=fs)
    else:
        store = zarr.DirectoryStore(img_path)
    return zarr.open(store, mode="r")[0]


def _read_n5(img_path):
    """
    Read an N5 volume from local disk or GCS.

    Parameters
    ----------
    img_path : str
        Path to the N5 directory.

    Returns
    -------
    np.ndarray
        N5 group volume stored at key "volume".
    """
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(anon=False)
        store = zarr.n5.N5FSStore(img_path, s=fs)
    else:
        store = zarr.n5.N5Store(img_path)
    return zarr.open(store, mode="r")["volume"]


def _read_tiff(img_path, storage_options=None):
    """
    Read a TIFF file from local disk or GCS.

    Parameters
    ----------
    img_path : str
        Path to the TIFF file.
    storage_options : dict, optional
        Additional kwargs for GCSFileSystem.

    Returns
    -------
    np.ndarray
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
    Check if the path is a GCS path.

    Parameters
    ----------
    path : str

    Returns
    -------
    bool
    """
    return path.startswith("gs://")


def _is_s3_path(path):
    """
    Check if the path is an S3 (Amazon S3) path.

    Parameters
    ----------
    path : str

    Returns
    -------
    bool
    """
    return path.startswith("s3://")


# --- Read Patches ---
def get_patch(img, voxel, shape, from_center=True):
    """
    Extracts a patch from an image based on the given voxel coordinate and
    patch shape.

    Parameters
    ----------
    img : zarr.core.Array
         A Zarr object representing an image.
    voxel : Tuple[int]
        Voxel coordinate used to extract patch.
    shape : Tuple[int]
        Shape of the image patch to extract.
    from_center : bool, optional
        Indicates whether the given voxel is the center or top, left, front
        corner of the patch to be extracted.

    Returns
    -------
    numpy.ndarray
        Patch extracted from the given image.
    """
    s, e = get_start_end(voxel, shape, from_center=from_center)
    if len(img.shape) == 5:
        return img[0, 0, s[0]: e[0], s[1]: e[1], s[2]: e[2]]
    else:
        return img[s[0]: e[0], s[1]: e[1], s[2]: e[2]]


def get_start_end(voxel, shape, from_center=True):
    """
    Gets the start and end indices of the image patch to be read.

    Parameters
    ----------
    voxel : Tuple[int]
        Voxel coordinate that specifies either the center or front-top-left
        corner of the patch to be read.
    shape : Tuple[int]
        Shape of the image patch to be read.
    from_center : bool, optional
        Indication of whether the provided coordinates represent the center of
        the patch or the front-top-left corner. Default is True.

    Return
    ------
    Tuple[List[int]]
        Start and end indices of the image patch to be read.
    """
    if from_center:
        start = [voxel[i] - shape[i] // 2 for i in range(3)]
        end = [voxel[i] + shape[i] // 2 for i in range(3)]
    else:
        start = voxel
        end = [voxel[i] + shape[i] for i in range(3)]
    return start, end


# --- Helpers ---
def calculate_offsets(img, window_shape, overlap=(0, 0, 0)):
    """
    Generates a list of 3D coordinates representing the front-top-left corner
    by sliding a window over a 3D image, given a specified window size and
    overlap between adjacent windows.

    Parameters
    ----------
    img : zarr.core.Array
        Input 3D image.
    window_shape : Tuple[int]
        Shape of the sliding window.
    overlap : Tuple[int]
        Overlap between adjacent windows.

    Returns
    -------
    List[Tuple[int]]
        Voxel coordinates representing the front-top-left corner.
    """
    # Calculate stride based on the overlap and window size
    stride = tuple(w - o for w, o in zip(window_shape, overlap))
    i_stride, j_stride, k_stride = stride

    # Get dimensions of the window
    if len(img.shape) == 5:
        _, _, i_dim, j_dim, k_dim = img.shape
    else:
        i_dim, j_dim, k_dim = img.shape
    i_win, j_win, k_win = window_shape

    # Loop over the  with the sliding window
    voxels = []
    for i in range(0, i_dim - i_win + 1, i_stride):
        for j in range(0, j_dim - j_win + 1, j_stride):
            for k in range(0, k_dim - k_win + 1, k_stride):
                voxels.append((i, j, k))
    return voxels


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
    torch.Tensor
        Binary tensor, where each element indicates the affinity for each
        voxel based on the given edge.

    """
    o1, o2 = get_offset_masks(label_mask, edge)
    aff_mask = (o1 == o2) & (o1 != 0)
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
    tuple of torch.Tensor
        A tuple containing two tensors:
        - "arr1": Subarray extracted based on the edge affinity.
        - "arr2": Subarray extracted based on the negative of the edge
                  affinity.
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


def make_segmentation_colormap(mask, seed=42):
    """
    Create a matplotlib ListedColormap for a segmentation mask. Ensures label
    0 maps to black and all other labels get distinct random colors.

    Parameters
    ----------
    mask : np.ndarray
        Segmentation mask with integer labels. Assumes label 0 is background.
    seed : int
        Random seed for color reproducibility.

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


def normalize(img):
    """
    Normalize a NumPy image array based on percentile clipping.

    Parameters
    ----------
    img : np.ndarray
        Input image array to normalize.

    Returns
    -------
    np.ndarray
        Normalized image with values approximately in [0, 1].
    """
    mn, mx = np.percentile(img, 5), np.percentile(img, 99.9)
    return (img - mn) / mx


def plot_mips(img, output_path=None, vmax=None):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : numpy.ndarray
        Input image to generate MIPs from.

    Returns
    -------
    None
    """
    vmax = vmax or np.percentile(img, 99.9)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
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
    if output_path:
        plt.savefig(output_path, dpi=200)
    plt.show()
    plt.close(fig)


def plot_segmentation_mips(mask):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    cmap = make_segmentation_colormap(mask)

    for i in range(3):
        if len(mask.shape) == 5:
            mip = np.max(mask[0, 0, ...], axis=i)
        else:
            mip = np.max(mask, axis=i)

        axs[i].imshow(mip, cmap=cmap, interpolation="none")
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def relabel_by_size(label_mask):
    """
    Relabels a segmentation mask so that larger segments get higher label IDs.
    Background (label 0) remains 0.

    Parameters
    ----------
    label_mask : np.ndarray
        Integer segmentation mask (3D, 4D, or 5D), with 0 as background.

    Returns
    -------
    np.ndarray
        Relabeled mask with largest segments assigned highest IDs.
    """
    flat = label_mask.ravel()
    max_label = flat.max()
    sizes = np.bincount(flat)

    # Exclude background (label 0)
    labels = np.nonzero(sizes)[0]
    labels = labels[labels != 0]

    # Sort labels by size (ascending)
    sorted_labels = labels[np.argsort(sizes[labels])]

    # Assign new labels: 1 for smallest, N for largest
    mapping = np.zeros(max_label + 1, dtype=label_mask.dtype)
    mapping[sorted_labels] = np.arange(1, len(sorted_labels) + 1, dtype=label_mask.dtype)
    return mapping[label_mask]


def remove_small_segments(label_mask, n):
    ids, cnts = unique(label_mask, return_counts=True)
    ids = [i for i, cnt in zip(ids, cnts) if cnt > n and i != 0]
    ids = mask_except(label_mask, ids)
    label_mask, _ = renumber(ids, preserve_zero=True, in_place=True)
    return label_mask


def zero_border_3d(img, border_width=64):
    """
    Zeros out everything within `border_width` voxels of the border of a 3D image.

    Parameters
    ----------
    img : np.ndarray
        A 3D NumPy array.
    border_width : int
        Number of voxels to zero out from each edge. Default is 64.

    Returns
    -------
    np.ndarray
        The input image with the border zeroed out.
    """
    assert img.ndim == 3, "Input image must be 3D"
    img = img.copy()
    z, y, x = img.shape

    img[:border_width, :, :] = 0         # Top
    img[-border_width:, :, :] = 0        # Bottom
    img[:, :border_width, :] = 0         # Front
    img[:, -border_width:, :] = 0        # Back
    img[:, :, :border_width] = 0         # Left
    img[:, :, -border_width:] = 0        # Right
    return img
