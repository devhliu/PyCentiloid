"""
Spatial normalization module for pycentiloid package.
"""

import nibabel as nib
import numpy as np
from nipype.interfaces import ants
from ..utils.validation import validate_input_images

def normalize_to_mni(image_path, template_path, output_path=None):
    """
    Normalize an image to MNI space.
    
    Parameters
    ----------
    image_path : str
        Path to the input image
    template_path : str
        Path to the MNI template
    output_path : str, optional
        Path to save the normalized image
        
    Returns
    -------
    str
        Path to the normalized image
    """
    # Validate inputs
    validate_input_images([image_path, template_path])
    
    # Initialize ANTs normalization
    norm = ants.Registration()
    norm.inputs.fixed_image = template_path
    norm.inputs.moving_image = image_path
    norm.inputs.output_transform_prefix = "to_mni_"
    norm.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    norm.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
    norm.inputs.number_of_iterations = [[1000, 500, 250],
                                      [1000, 500, 250],
                                      [100, 70, 50]]
    norm.inputs.dimension = 3
    norm.inputs.write_composite_transform = True
    norm.inputs.collapse_output_transforms = True
    norm.inputs.initial_moving_transform_com = True
    norm.inputs.metric = ['MI', 'MI', 'CC']
    norm.inputs.metric_weight = [1] * 3
    norm.inputs.radius_or_number_of_bins = [32, 32, 4]
    norm.inputs.sampling_strategy = ['Regular'] * 3
    norm.inputs.sampling_percentage = [0.3, 0.3, 0.3]
    norm.inputs.convergence_threshold = [1.e-8] * 3
    norm.inputs.convergence_window_size = [10] * 3
    norm.inputs.smoothing_sigmas = [[4, 2, 1]] * 3
    norm.inputs.sigma_units = ['vox'] * 3
    norm.inputs.shrink_factors = [[6, 4, 2]] * 3
    norm.inputs.use_estimate_learning_rate_once = [True] * 3
    norm.inputs.use_histogram_matching = [True] * 3
    norm.inputs.output_warped_image = output_path or 'normalized.nii.gz'
    
    # Run normalization
    norm.run()
    
    return norm.inputs.output_warped_image