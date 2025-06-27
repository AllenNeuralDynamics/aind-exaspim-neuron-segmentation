"""
Created on Wed June 25 3:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Miscellaneous helper routines.

"""

from google.cloud import storage

import os
import shutil


# -- OS Utils ---
def mkdir(path, delete=False):
    """
    Creates a directory at the given path.

    Parameters
    ----------
    path : str
        Path of directory to be created.
    delete : bool, optional
        Indication of whether to delete directory at path if it already
        exists. The default is False.

    Returns
    -------
    None
    """
    if delete:
        rmdir(path)
    if not os.path.exists(path):
        os.mkdir(path)


def rmdir(path):
    """
    Removes the given directory and all of its subdirectories.

    Parameters
    ----------
    path : str
        Path to directory to be removed if it exists.

    Returns
    -------
    None
    """
    if os.path.exists(path):
        shutil.rmtree(path)


# --- GCS utils ---
def find_subprefix_with_keyword(bucket_name, prefix, keyword):
    """
    Finds the first GCS subprefix under a given prefix that contains a
    specified keyword.

    Parameters
    ----------
    bucket_name : str
        Name of the GCS bucket.
    prefix : str
        The prefix to search under.
    keyword : str
        Keyword to look for within the subprefixes.

    Returns
    -------
    str
        First subprefix containing the keyword.
    """
    # Search for prefix
    for subprefix in list_gcs_subprefixes(bucket_name, prefix):
        if keyword in subprefix:
            return subprefix

    # Search for filename
    for path in list_gcs_paths(bucket_name, prefix):
        if keyword in path:
            return path
    raise Exception(f"Prefix with keyword '{keyword}' not found in {prefix}")


def list_gcs_paths(bucket_name, prefix):
    """
    Lists filenames in a GCS bucket path filtered by file extension.

    Parameters
    ----------
    bucket : str
        Name of GCS bucket to be read from.
    prefix : str
        Path to directory in the GCS bucket.

    Returns
    -------
    List[str]
        Path to files in that prefix
    """
    paths = list()
    storage_client = storage.Client()
    for blob in storage_client.list_blobs(bucket_name, prefix=prefix):
        if not blob.name.endswith('/'):
            paths.append(blob.name)
    return paths


def list_gcs_subprefixes(bucket_name, prefix):
    """
    Lists all direct subdirectories of a given prefix in a GCS bucket.

    Parameters
    ----------
    bucket : str
        Name of GCS bucket to be read from.
    prefix : str
        Path to directory in the GCS bucket.

    Returns
    -------
    subdirs : List[str]
         List of direct subdirectories.
    """
    # Load blobs
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(
        bucket_name, prefix=prefix, delimiter="/"
    )
    [blob.name for blob in blobs]

    # Parse directory contents
    prefix_depth = len(prefix.split("/"))
    subdirs = list()
    for prefix in blobs.prefixes:
        is_dir = prefix.endswith("/")
        is_direct_subdir = len(prefix.split("/")) - 1 == prefix_depth
        if is_dir and is_direct_subdir:
            subdirs.append(prefix)
    return subdirs
