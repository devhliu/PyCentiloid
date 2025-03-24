"""
Centiloid calculation module for PyCentiloid.

This module provides functions for calculating Centiloid scores from PET images,
based on standardized reference regions and target regions.
"""

import os
import ants
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, Literal

from ..core.tracer import TracerRegistry
from ..preprocessing.registration import register_to_mni
from ..atlas.manager import AtlasManager

class CentiloidCalculator:
    """
    Class for calculating Centiloid scores from PET images.
    """
    
    def __init__(self, 
                 tracer_type: str,
                 resolution: str = '2mm'):
        """
        Initialize Centiloid calculator.
        
        Parameters
        ----------
        tracer_type : str
            Type of tracer ('PIB', 'FBB', 'FBP', 'FMM', 'FBZ')
        resolution : str
            Image resolution ('1mm' or '2mm')
        """
        self.tracer_type = tracer_type
        self.resolution = resolution
        
        # Get tracer information
        self.tracer = TracerRegistry.get_tracer(tracer_type)
        
        # Load reference data
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Load reference data for Centiloid calculation."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Load reference regions
        atlas_manager = AtlasManager(resolution=self.resolution, atlas_type='centiloid')
        
        # Get reference region based on tracer type
        if self.tracer_type in ['PIB', 'FMM']:
            self.reference_region = 'whole_cerebellum'
        elif self.tracer_type == 'FBB':
            self.reference_region = 'cerebellar_cortex'
        elif self.tracer_type == 'FBP':
            self.reference_region = 'cerebellar_white_matter'
        elif self.tracer_type == 'FBZ':
            self.reference_region = 'pons'
        else:
            raise ValueError(f"Unsupported tracer type: {self.tracer_type}")
        
        # Get reference mask
        self.reference_mask = atlas_manager.get_region_mask(self.reference_region)
        
        # Get target region (standard centiloid target)
        self.target_region = 'cortical_composite'
        self.target_mask = atlas_manager.get_region_group_mask(self.target_region)
        
        # Load reference values
        self.reference_values = self._get_reference_values()
    
    def _get_reference_values(self) -> Dict[str, float]:
        """
        Get reference values for Centiloid calculation.
        
        Returns
        -------
        dict
            Dictionary containing reference values
        """
        # Standard reference values from literature
        reference_values = {
            'PIB': {
                'YC_mean': 1.0,
                'YC_sd': 0.067,
                'AD_mean': 2.07,
                'AD_sd': 0.236,
                'm': 100.0 / (2.07 - 1.0),
                'b': -100.0 / (2.07 - 1.0)
            },
            'FBB': {
                'YC_mean': 1.08,
                'YC_sd': 0.076,
                'AD_mean': 1.67,
                'AD_sd': 0.158,
                'm': 100.0 / (1.67 - 1.08),
                'b': -108.0 / (1.67 - 1.08)
            },
            'FBP': {
                'YC_mean': 1.06,
                'YC_sd': 0.054,
                'AD_mean': 1.71,
                'AD_sd': 0.177,
                'm': 100.0 / (1.71 - 1.06),
                'b': -106.0 / (1.71 - 1.06)
            },
            'FMM': {
                'YC_mean': 1.03,
                'YC_sd': 0.065,
                'AD_mean': 1.61,
                'AD_sd': 0.169,
                'm': 100.0 / (1.61 - 1.03),
                'b': -103.0 / (1.61 - 1.03)
            },
            'FBZ': {
                'YC_mean': 1.05,
                'YC_sd': 0.062,
                'AD_mean': 1.58,
                'AD_sd': 0.145,
                'm': 100.0 / (1.58 - 1.05),
                'b': -105.0 / (1.58 - 1.05)
            }
        }
        
        return reference_values.get(self.tracer_type, {})
    
    def calculate(self, 
                 pet_path: Union[str, Path],
                 mri_path: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 register: bool = True) -> Dict[str, float]:
        """
        Calculate Centiloid score from PET image.
        
        Parameters
        ----------
        pet_path : str or Path
            Path to PET image
        mri_path : str or Path, optional
            Path to MRI image for registration
        output_dir : str or Path, optional
            Directory to save intermediate files
        register : bool
            Whether to register PET to MNI space
            
        Returns
        -------
        dict
            Dictionary containing SUVR and Centiloid values
        """
        # Create output directory if needed
        if output_dir is None:
            output_dir = Path(pet_path).parent / "centiloid_output"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Register PET to MNI space if needed
        if register:
            if mri_path:
                # Register using MRI as intermediate
                # First register MRI to MNI
                mri_mni_path = output_dir / "mri_mni.nii.gz"
                register_to_mni(
                    mri_path, 
                    modality='MRI', 
                    output_path=mri_mni_path,
                    resolution=self.resolution
                )
                
                # Then register PET to MRI
                from ..preprocessing.registration import register_pet_to_mri
                pet_mri_path = output_dir / "pet_mri.nii.gz"
                register_pet_to_mri(
                    pet_path,
                    mri_mni_path,
                    output_path=pet_mri_path
                )
                
                # Use registered PET
                pet_registered = pet_mri_path
            else:
                # Direct registration of PET to MNI
                pet_registered = output_dir / "pet_mni.nii.gz"
                register_to_mni(
                    pet_path, 
                    modality='PET', 
                    output_path=pet_registered,
                    resolution=self.resolution
                )
        else:
            # Use original PET (assuming it's already in MNI space)
            pet_registered = pet_path
        
        # Load registered PET
        pet_img = ants.image_read(str(pet_registered))
        
        # Calculate SUVR
        target_mean = self._calculate_mean_intensity(pet_img, self.target_mask)
        reference_mean = self._calculate_mean_intensity(pet_img, self.reference_mask)
        
        if reference_mean == 0:
            raise ValueError("Reference region has zero mean intensity")
        
        suvr = target_mean / reference_mean
        
        # Calculate Centiloid value
        centiloid = self._suvr_to_centiloid(suvr)
        
        # Return results
        return {
            'SUVR': suvr,
            'Centiloid': centiloid,
            'reference_region': self.reference_region,
            'target_region': self.target_region,
            'reference_mean': reference_mean,
            'target_mean': target_mean
        }
    
    def _calculate_mean_intensity(self, 
                                 image: ants.ANTsImage, 
                                 mask: ants.ANTsImage) -> float:
        """
        Calculate mean intensity in a region.
        
        Parameters
        ----------
        image : ants.ANTsImage
            Input image
        mask : ants.ANTsImage
            Region mask
            
        Returns
        -------
        float
            Mean intensity in the region
        """
        # Ensure mask is binary
        mask_binary = (mask.numpy() > 0).astype(np.float32)
        mask_img = ants.from_numpy(mask_binary, origin=mask.origin,
                                  spacing=mask.spacing, direction=mask.direction)
        
        # Calculate mean
        masked_img = image * mask_img
        masked_data = masked_img.numpy()
        mask_data = mask_img.numpy()
        
        # Avoid division by zero
        mask_sum = np.sum(mask_data)
        if mask_sum == 0:
            return 0
        
        mean_intensity = np.sum(masked_data) / mask_sum
        return float(mean_intensity)
    
    def _suvr_to_centiloid(self, suvr: float) -> float:
        """
        Convert SUVR to Centiloid value.
        
        Parameters
        ----------
        suvr : float
            SUVR value
            
        Returns
        -------
        float
            Centiloid value
        """
        # Get conversion parameters
        m = self.reference_values.get('m', 100.0)
        b = self.reference_values.get('b', -100.0)
        
        # Calculate Centiloid
        centiloid = m * suvr + b
        
        return centiloid
    
    def get_reference_values(self) -> Dict[str, float]:
        """
        Get reference values for the current tracer.
        
        Returns
        -------
        dict
            Dictionary containing reference values
        """
        return self.reference_values