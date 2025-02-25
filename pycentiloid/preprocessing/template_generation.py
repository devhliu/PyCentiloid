"""
Template generation module for creating modality-specific templates in MNI space.
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from nipype.interfaces import ants
from ..utils.validation import validate_input_images
from ..utils.image_utils import load_image, save_image, resample_image
from ..config import Config

class TemplateBuilder:
    """Class for building modality-specific templates in MNI space."""
    
    MODALITY_PARAMS = {
        'PET': {
            'transforms': ['Rigid', 'Affine', 'SyN'],
            'metrics': ['MI', 'MI', 'CC'],
            'sampling': 0.3,
            'smoothing': [4, 2, 1],
            'convergence': [1000, 500, 250]
        },
        'CT': {
            'transforms': ['Rigid', 'Affine', 'SyN'],
            'metrics': ['MI', 'MI', 'MSQ'],
            'sampling': 0.2,
            'smoothing': [3, 2, 1],
            'convergence': [1000, 500, 250]
        },
        'T1': {
            'transforms': ['Rigid', 'Affine', 'SyN'],
            'metrics': ['MI', 'MI', 'CC'],
            'sampling': 0.2,
            'smoothing': [4, 2, 0],
            'convergence': [1000, 500, 250]
        }
    }
    
    def __init__(self, modality, config=None):
        """
        Initialize template builder.
        
        Parameters
        ----------
        modality : str
            Image modality ('PET', 'CT', 'T1')
        config : Config, optional
            Configuration object
        """
        if modality not in self.MODALITY_PARAMS:
            raise ValueError(f"Unsupported modality: {modality}")
            
        self.config = config or Config()
        self.modality = modality
        self.params = self.MODALITY_PARAMS[modality]
        self.mni_template = self._get_reference_template()
    
    def _get_reference_template(self):
        """Get appropriate reference template for modality."""
        templates = {
            'PET': 'mni152_pet_2mm.nii.gz',
            'CT': 'mni152_ct_2mm.nii.gz',
            'T1': 'mni152_t1_2mm.nii.gz'
        }
        return str(self.config.TEMPLATE_DIR / templates[self.modality])
    
    def create_template(self, image_paths, output_dir=None, iterations=3,
                       target_shape=(91, 109, 91), target_voxel_size=(2, 2, 2)):
        """Create modality-specific template."""
        validate_input_images(image_paths)
        
        output_dir = Path(output_dir or self.config.TEMPLATE_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        template_path = str(output_dir / f'{self.modality.lower()}_template.nii.gz')
        
        # Initial registration to MNI
        mni_registered = self._register_all_to_mni(image_paths, output_dir)
        
        # Create and refine template
        current_template = self._create_initial_template(mni_registered, output_dir,
                                                       target_shape, target_voxel_size)
        
        # Iterative refinement
        for i in range(iterations):
            current_template = self._refine_template(mni_registered, current_template,
                                                   output_dir, i, target_shape,
                                                   target_voxel_size)
        
        # Final registration to MNI
        final_template = self._register_to_mni(current_template, template_path)
        
        return final_template
    
    def _register_all_to_mni(self, image_paths, output_dir):
        """Register all images to MNI space."""
        registered = []
        for img_path in image_paths:
            out_path = str(output_dir / f'mni_{Path(img_path).stem}.nii.gz')
            registered.append(self._register_to_mni(img_path, out_path))
        return registered
    
    def _create_initial_template(self, images, output_dir, shape, voxel_size):
        """Create initial template by averaging."""
        return self._create_average_template(
            images,
            str(output_dir / 'initial_template.nii.gz'),
            shape, voxel_size
        )
    
    def _refine_template(self, images, current_template, output_dir, iteration,
                        shape, voxel_size):
        """Refine template through registration and averaging."""
        registered = []
        for img_path in images:
            out_path = str(output_dir / f'iter{iteration+1}_{Path(img_path).stem}.nii.gz')
            registered.append(self._register_to_template(img_path, current_template, out_path))
        
        return self._create_average_template(
            registered,
            str(output_dir / f'template_iter{iteration+1}.nii.gz'),
            shape, voxel_size
        )
    
    def _register_to_mni(self, image_path, output_path):
        """Register image to MNI space with modality-specific parameters."""
        reg = ants.Registration()
        reg.inputs.fixed_image = self.mni_template
        reg.inputs.moving_image = image_path
        reg.inputs.output_transform_prefix = "to_mni_"
        reg.inputs.transforms = self.params['transforms']
        reg.inputs.transform_parameters = [(0.1,)] * len(self.params['transforms'])
        reg.inputs.number_of_iterations = [self.params['convergence']] * len(self.params['transforms'])
        reg.inputs.dimension = 3
        reg.inputs.write_composite_transform = True
        reg.inputs.collapse_output_transforms = True
        reg.inputs.initial_moving_transform_com = True
        reg.inputs.metric = self.params['metrics']
        reg.inputs.metric_weight = [1] * len(self.params['metrics'])
        reg.inputs.radius_or_number_of_bins = [32] * len(self.params['metrics'])
        reg.inputs.sampling_strategy = ['Regular'] * len(self.params['metrics'])
        reg.inputs.sampling_percentage = [self.params['sampling']] * len(self.params['metrics'])
        reg.inputs.convergence_threshold = [1.e-8] * len(self.params['metrics'])
        reg.inputs.convergence_window_size = [10] * len(self.params['metrics'])
        reg.inputs.smoothing_sigmas = [self.params['smoothing']] * len(self.params['metrics'])
        reg.inputs.sigma_units = ['vox'] * len(self.params['metrics'])
        reg.inputs.shrink_factors = [[4, 2, 1]] * len(self.params['metrics'])
        reg.inputs.use_estimate_learning_rate_once = [True] * len(self.params['metrics'])
        reg.inputs.use_histogram_matching = [True] * len(self.params['metrics'])
        reg.inputs.output_warped_image = output_path
        
        reg.run()
        return reg.inputs.output_warped_image