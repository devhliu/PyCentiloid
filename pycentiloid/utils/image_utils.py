"""
Utility functions for image processing.
"""

import nibabel as nib
import numpy as np

def load_image(image_path):
    """
    Load a NIfTI image.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
        
    Returns
    -------
    nibabel.Nifti1Image
        Loaded image
    """
    return nib.load(image_path)

def save_image(data, affine, output_path):
    """
    Save data as a NIfTI image.
    
    Parameters
    ----------
    data : numpy.ndarray
        Image data
    affine : numpy.ndarray
        Affine transformation matrix
    output_path : str
        Path to save the image
    """
    img = nib.Nifti1Image(data, affine)
    nib.save(img, output_path)

def resample_image(image, target_spacing):
    """
    Resample image to target spacing.
    
    Parameters
    ----------
    image : nibabel.Nifti1Image
        Input image
    target_spacing : tuple
        Target voxel spacing in mm
        
    Returns
    -------
    nibabel.Nifti1Image
        Resampled image
    """
    # Implementation of image resampling
    pass