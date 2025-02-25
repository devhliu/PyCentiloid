"""
Registration module for pycentiloid package.
Handles registration between PET and MRI images.
"""

import nibabel as nib
import numpy as np
from nipype.interfaces import ants
from ..utils.validation import validate_input_images

class Registration:
    """Image registration handler."""
    
    def __init__(self):
        self.config = Config()
        self.params = self.config.REGISTRATION['DEFAULT']
    
    def register_pet_to_mni(self, pet_path: str, template_path: str, 
                           output_path: str = None) -> str:
        """Register PET image to MNI space."""
        validate_input_images([pet_path, template_path])
        
        fixed = ants.image_read(str(template_path))
        moving = ants.image_read(str(pet_path))
        
        transform = ants.registration(
            fixed=fixed,
            moving=moving,
            **self.params
        )
        
        output_path = output_path or str(Path(pet_path).parent / f'registered_{Path(pet_path).name}')
        transform['warpedmovout'].to_filename(output_path)
        
        return output_path