"""
Registration module for pycentiloid package.
Handles registration between PET and MRI images.
"""

import nibabel as nib
import numpy as np
from nipype.interfaces import ants
from ..utils.validation import validate_input_images

class Registration:
    """Image registration handler for different modalities."""
    
    def __init__(self, modality='PET'):
        """
        Initialize registration handler.
        
        Parameters
        ----------
        modality : str
            Image modality ('PET', 'CT', 'T1')
        """
        self.config = Config()
        self.modality = modality
        self.params = self.config.REGISTRATION.get(modality, self.config.REGISTRATION['DEFAULT'])
    
    def register_to_template(self, image_path: str, template_path: str, 
                           output_path: str = None, resolution: str = '2mm') -> str:
        """
        Register image to template with modality-specific parameters.
        
        Parameters
        ----------
        image_path : str
            Path to input image
        template_path : str
            Path to template image
        output_path : str, optional
            Path to save registered image
        resolution : str
            Target resolution ('1mm', '2mm')
            
        Returns
        -------
        str
            Path to registered image
        """
        validate_input_images([image_path, template_path])
        
        # Ensure consistent resolution
        from ..utils.resolution import match_resolution
        
        # Get image and template
        fixed = ants.image_read(str(template_path))
        moving = ants.image_read(str(image_path))
        
        # Match resolution if needed
        if moving.spacing != fixed.spacing:
            temp_path = str(Path(image_path).parent / f'temp_resampled_{Path(image_path).name}')
            moving = match_resolution(moving, fixed, temp_path)
        
        # Perform registration with modality-specific parameters
        transform = ants.registration(
            fixed=fixed,
            moving=moving,
            **self.params
        )
        
        output_path = output_path or str(Path(image_path).parent / f'registered_{Path(image_path).name}')
        transform['warpedmovout'].to_filename(output_path)
        
        return output_path