"""
Segmentation module for pycentiloid package.
"""

import nibabel as nib
import numpy as np
from nipype.interfaces import fsl
from ..utils.validation import validate_input_images

def segment_mri(mri_path, output_dir=None):
    """
    Perform tissue segmentation on MRI image using FSL FAST.
    
    Parameters
    ----------
    mri_path : str
        Path to the MRI image
    output_dir : str, optional
        Directory to save segmentation outputs
        
    Returns
    -------
    dict
        Paths to segmentation outputs (GM, WM, CSF)
    """
    # Validate input
    validate_input_images([mri_path])
    
    # Initialize FSL FAST
    fast = fsl.FAST()
    fast.inputs.in_files = mri_path
    fast.inputs.output_type = 'NIFTI_GZ'
    if output_dir:
        fast.inputs.out_dir = output_dir
    
    # Run segmentation
    result = fast.run()
    
    # Return paths to segmentation outputs
    return {
        'gm': result.outputs.partial_volume_files[1],
        'wm': result.outputs.partial_volume_files[2],
        'csf': result.outputs.partial_volume_files[0]
    }