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
    if ".zarr" in img_path:
        return _read_zarr(img_path)
    elif ".n5" in img_path:
        return _read_n5(img_path)
    elif ".tif" in img_path or ".tiff" in img_path:
        return _read_tiff(img_path)
    else:
        raise ValueError(f"Unsupported image format: {img_path}")


def _read_zarr(img_path):
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
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(anon=False)
        store = zarr.n5.N5FSStore(img_path, s=fs)
    else:
        store = zarr.n5.N5Store(img_path)
    return zarr.open(store, mode="r")["volume"]


def _read_tiff(img_path, storage_options=None):
    if _is_gcs_path(img_path):
        fs = gcsfs.GCSFileSystem(**(storage_options or {}))
        with fs.open(img_path, "rb") as f:
            return tifffile.imread(f)
    else:
        return tifffile.imread(img_path)


def _is_gcs_path(path):
    return path.startswith("gs://")


def _is_s3_path(path):
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


def list_block_paths(brain_id):
    """
    Lists the GCS paths to image blocks associated with a given brain ID.

    Parameters
    ----------
    brain_id : str
        Unique identifier for a brain dataset.

    Returns
    -------
    img_paths : List[str]
        GCS paths (gs://...) to the image blocks.
    """
    # Find prefix containing blocks
    bucket_name = "allen-nd-goog"
    prefix = util.find_subprefix_with_keyword(bucket_name, "from_aind/", brain_id)
    prefix += "blocks/"

    # Iterate over blocks
    img_paths, label_paths = list(), list()
    for block_prefix in util.list_gcs_subprefixes("allen-nd-goog", prefix):
        img_path = util.find_subprefix_with_keyword(
            bucket_name, block_prefix, "input"
        )
        label_path = util.find_subprefix_with_keyword(
            bucket_name, block_prefix, "Label"
        )
        img_paths.append(f"gs://{bucket_name}/{img_path}")
        label_paths.append(f"gs://{bucket_name}/{label_path}")
    return img_paths, label_paths


def normalize(img):
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
