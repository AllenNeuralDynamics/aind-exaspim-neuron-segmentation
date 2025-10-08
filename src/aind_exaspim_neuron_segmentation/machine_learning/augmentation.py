"""
Created on Tue Jan 21 14:00:00 2025

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Routines for applying image augmentation during training.

"""

from scipy.ndimage import gaussian_filter, rotate, zoom

import numpy as np
import random
import torchvision.transforms as transforms


# --- Image Transforms ---
class ImageTransforms:
    """
    Class that applies a sequence of transforms to a 3D input image and label
    mask.
    """

    def __init__(self):
        """
        Initializes a GeometricTransforms instance.
        """
        # Instance attributes
        self.geometric_transforms = [
            RandomFlip3D(),
            RandomRotation3D(),
        ]
        self.intensity_transforms1 = transforms.Compose(
                [RandomNoise3D(), RandomContrast3D()]
            )
        self.intensity_transforms2 = transforms.Compose(
                [RandomSmooth3D(), RandomContrast3D()]
            )

    def __call__(self, input_img, label_mask):
        """
        Applies geometric transforms to the input image and label mask.

        Parameters
        ----------
        input_img : numpy.ndarray
            Input image with shape (D, H, W).
        label_mask : numpy.ndarray
            Label mask with shape (D, H, W).

        Returns
        -------
        input_img : numpy.ndarray
            Transformed input image
        label_mask : numpy.ndarray
            Transformed label mask.
        """
        # Geometric transforms
        for transform in self.geometric_transforms:
            input_img, label_mask = transform(input_img, label_mask)

        # Intensity transforms
        if random.random() < 0.5:
            input_img = self.intensity_transforms1(input_img)
        else:
            input_img = self.intensity_transforms2(input_img)
        return input_img, label_mask


# --- Geometric Transforms ---
class RandomFlip3D:
    """
    Randomly flips a 3D image along one or more axes.
    """

    def __init__(self, axes=(0, 1, 2)):
        """
        Initializes a RandomFlip3D transformer.

        Parameters
        ----------
        axes : Tuple[float], optional
            Axes along which to flip the image. Default is (0, 1, 2).
        """
        self.axes = axes

    def __call__(self, input_img, label_mask):
        """
        Applies random flipping to the input image and label mask.

        Parameters
        ----------
        input_img : numpy.ndarray
            Input image with shape (D, H, W).
        label_mask : numpy.ndarray
            Label mask with shape (D, H, W).

        Returns
        -------
        input_img : numpy.ndarray
            Flipped input image.
        label_mask : numpy.ndarray
            Flipped label mask.
        """
        for axis in self.axes:
            if random.random() > 0.5:
                input_img = np.flip(input_img, axis=axis)
                label_mask = np.flip(label_mask, axis=axis)
        return input_img, label_mask


class RandomRotation3D:
    """
    Applies random rotation along a randomly chosen axis.
    """

    def __init__(self, angles=(-45, 45), axes=((0, 1), (0, 2), (1, 2))):
        """
        Initializes a RandomRotation3D transformer.

        Parameters
        ----------
        angles : Tuple[int], optional
            Maximum angle of rotation. Default is (-45, 45).
        axis : Tuple[Tuple[int]], optional
            Axes to apply rotation. Default is ((0, 1), (0, 2), (1, 2)).
        """
        self.angles = angles
        self.axes = axes

    def __call__(self, input_img, label_mask):
        """
        Rotates the input image and label mask.

        Parameters
        ----------
        input_img : numpy.ndarray
            Input image with shape (D, H, W).
        label_mask : numpy.ndarray
            Label mask with shape (D, H, W).

        Returns
        -------
        input_img : numpy.ndarray
            Rotated input image.
        label_mask : numpy.ndarray
            Rotated label mask.
        """
        for axes in self.axes:
            if random.random() > 0.2:
                angle = random.uniform(*self.angles)
                input_img = rotate3d(input_img, angle, axes)
                label_mask = rotate3d(label_mask, angle, axes)
        return input_img, label_mask


class RandomScale3D:
    """
    Applies random scaling along each axis.
    """

    def __init__(self, scale_range=(0.9, 1.1)):
        """
        Initializes a RandomScale3D transformer.

        Parameters
        ----------
        scale_range : Tuple[float], optional
            Range of scaling factors. Default is (0.9, 1.1).
        """
        self.scale_range = scale_range

    def __call__(self, input_img, label_mask):
        """
        Applies random rescaling to the input image and label mask.

        Parameters
        ----------
        input_img : numpy.ndarray
            Input image with shape (D, H, W).
        label_mask : numpy.ndarray
            Label mask with shape (D, H, W).

        Returns
        -------
        input_img : numpy.ndarray
            Rescaled input image.
        label_mask : numpy.ndarray
            Rescaled label mask.
        """
        # Sample new image shape
        alpha = np.random.uniform(self.scale_range[0], self.scale_range[1])
        new_shape = (
            int(input_img.shape[1] * alpha),
            int(input_img.shape[2] * alpha),
            int(input_img.shape[3] * alpha),
        )

        # Compute the zoom factors
        shape = input_img.shape[1:]
        zoom_factors = [
            new_dim / old_dim for old_dim, new_dim in zip(shape, new_shape)
        ]

        # Rescale images
        input_img[0, ...] = zoom(input_img[0, ...], zoom_factors, order=3)
        label_mask[1, ...] = zoom(label_mask[1, ...], zoom_factors, order=3)
        return input_img, label_mask


# --- Intensity Transforms ---
class RandomContrast3D:
    """
    Adjusts the contrast of a 3D image by scaling voxel intensities.
    """

    def __init__(self, factor_range=(0.8, 1.2)):
        """
        Initializes a RandomContrast3D transformer.

        Parameters
        ----------
        factor_range : Tuple[float], optional
            Range of contrast factors. Default is (0.8, 1.2).
        """
        self.factor_range = factor_range

    def __call__(self, img):
        """
        Applies contrast to an image.

        Parameters
        ----------
        img : numpy.ndarray
            image with shape (D, H, W).

        Returns
        -------
        numpy.ndarray
            Contrasted image.
        """
        factor = random.uniform(*self.factor_range)
        return np.clip(img * factor, 0, 1)


class RandomNoise3D:
    """
    Adds random Gaussian noise to a 3D image.
    """

    def __init__(self, max_std=0.15):
        """
        Initializes a RandomNoise3D transformer.

        Parameters
        ----------
        max_std : float, optional
            Maximum standard deviation of the Gaussian noise distribution.
            Default is 0.05.
        """
        self.max_std = max_std

    def __call__(self, img):
        """
        Adds Gaussian noise to an image.

        Parameters
        ----------
        img : numpy.ndarray
            Image to which noise will be added.

        Returns
        -------
        numpy.ndarray
            Noisy image.
        """
        std = self.max_std * random.random()
        noise = np.random.normal(0, std, img.shape)
        return img + noise


class RandomSmooth3D:
    """
    Applies Gaussian smoothing to a 3D image.
    """

    def __init__(self, max_sigma=1.0):
        """
        Initializes a GaussianSmooth3D transformer.

        Parameters
        ----------
        max_sigma : float, optional
            Maximum standard deviation of the Gaussian kernel.
            Default is 1.0.
        """
        self.max_sigma = max_sigma

    def __call__(self, img):
        """
        Applies Gaussian smoothing to an image.

        Parameters
        ----------
        img : numpy.ndarray
            3D image to smooth.

        Returns
        -------
        numpy.ndarray
            Smoothed image.
        """
        sigma = self.max_sigma * random.random()
        return gaussian_filter(img, sigma=sigma)


# --- Helpers ---
def rotate3d(img, angle, axes):
    """
    Rotates a 3D image patch around the specified axes by a given angle.

    Parameters
    ----------
    img : numpy.ndarray
        Image to be rotated.
    angle : float
        Angle (in degrees) by which to rotate the image patch around the
        specified axes.
    axes : Tuple[int]
        Tuple representing the two axes of rotation.

    Returns
    -------
    img : numpy.ndarray
        Rotated image.
    """
    img = rotate(
        img,
        angle,
        axes=axes,
        mode="grid-mirror",
        reshape=False,
        order=0,
    )
    return img
