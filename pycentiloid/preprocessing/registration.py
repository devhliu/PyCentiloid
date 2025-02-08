"""
Registration module for pycentiloid package.
Handles registration between PET and MRI images.
"""

import nibabel as nib
import numpy as np
from nipype.interfaces import ants
from ..utils.validation import validate_input_images

def register_pet_to_mri(pet_path, mri_path, output_path=None):
    """
    Register PET image to MRI space using ANTs registration.
    
    Parameters
    ----------
    pet_path : str
        Path to the PET image in NIfTI format
    mri_path : str
        Path to the MRI image in NIfTI format
    output_path : str, optional
        Path to save the registered PET image
        
    Returns
    -------
    str
        Path to the registered PET image
    """
    # Validate inputs
    validate_input_images([pet_path, mri_path])
    
    # Initialize ANTs registration
    reg = ants.Registration()
    reg.inputs.fixed_image = mri_path
    reg.inputs.moving_image = pet_path
    reg.inputs.output_transform_prefix = "pet2mri_"
    reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
    reg.inputs.number_of_iterations = [[1000, 500, 250],
                                     [1000, 500, 250],
                                     [100, 70, 50]]
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = True
    reg.inputs.initial_moving_transform_com = True
    reg.inputs.metric = ['MI', 'MI', 'CC']
    reg.inputs.metric_weight = [1] * 3
    reg.inputs.radius_or_number_of_bins = [32, 32, 4]
    reg.inputs.sampling_strategy = ['Regular'] * 3
    reg.inputs.sampling_percentage = [0.3, 0.3, 0.3]
    reg.inputs.convergence_threshold = [1.e-8] * 3
    reg.inputs.convergence_window_size = [10] * 3
    reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 3
    reg.inputs.sigma_units = ['vox'] * 3
    reg.inputs.shrink_factors = [[6, 4, 2]] * 3
    reg.inputs.use_estimate_learning_rate_once = [True] * 3
    reg.inputs.use_histogram_matching = [True] * 3
    reg.inputs.output_warped_image = output_path or 'registered_pet.nii.gz'
    
    # Run registration
    reg.run()
    
    return reg.inputs.output_warped_image