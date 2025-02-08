"""
Centiloid calculation module for pycentiloid package.
"""

import nibabel as nib
import numpy as np
from ..utils.validation import validate_input_images

class CentiloidCalculator:
    """
    Class for calculating Centiloid values from PET images.
    """
    
    # Tracer-specific parameters
    TRACER_PARAMS = {
        'PIB': {'slope': 1.0, 'intercept': 0.0},  # Reference values
        'FBB': {'slope': 0.985, 'intercept': -0.187},
        'FBP': {'slope': 0.994, 'intercept': -0.0994},
        'FMM': {'slope': 0.979, 'intercept': -0.117}
    }
    
    def __init__(self, tracer):
        """
        Initialize the calculator.
        
        Parameters
        ----------
        tracer : str
            PET tracer type ('PIB', 'FBB', 'FBP', 'FMM')
        """
        if tracer not in self.TRACER_PARAMS:
            raise ValueError(f"Unsupported tracer: {tracer}")
        self.tracer = tracer
        self.params = self.TRACER_PARAMS[tracer]
    
    def calculate(self, pet_path, ref_region_mask, target_region_mask):
        """
        Calculate Centiloid value.
        
        Parameters
        ----------
        pet_path : str
            Path to the normalized PET image
        ref_region_mask : str
            Path to the reference region mask
        target_region_mask : str
            Path to the target region mask
            
        Returns
        -------
        float
            Centiloid value
        """
        # Validate inputs
        validate_input_images([pet_path, ref_region_mask, target_region_mask])
        
        # Load images
        pet_img = nib.load(pet_path)
        ref_mask = nib.load(ref_region_mask)
        target_mask = nib.load(target_region_mask)
        
        pet_data = pet_img.get_fdata()
        ref_data = ref_mask.get_fdata()
        target_data = target_mask.get_fdata()
        
        # Calculate SUVr
        ref_mean = np.mean(pet_data[ref_data > 0])
        target_mean = np.mean(pet_data[target_data > 0])
        suvr = target_mean / ref_mean
        
        # Convert to Centiloid
        centiloid = (suvr - self.params['intercept']) / self.params['slope'] * 100
        
        return centiloid