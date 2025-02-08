"""
Validation utilities for pycentiloid package.
"""

import os
import nibabel as nib

def validate_input_images(image_paths):
    """
    Validate input image paths.
    
    Parameters
    ----------
    image_paths : list
        List of paths to input images
        
    Raises
    ------
    ValueError
        If any image path is invalid or file format is incorrect
    """
    for path in image_paths:
        if not