"""
Created on Wed June 25 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with images.

"""

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
        Path to GCS directory contain image blocks.

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
