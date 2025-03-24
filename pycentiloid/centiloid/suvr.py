"""
SUVr calculation module for PyCentiloid.

This module provides functions for calculating Standardized Uptake Value ratios (SUVr)
from PET images, based on reference regions and target regions.
"""

import os
import ants
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Literal

from ..atlas.manager import AtlasManager
from ..core.tracer import TracerRegistry

class SUVrCalculator:
    """
    Class for calculating SUVr values from PET images.
    """
    
    def __init__(self, 
                 tracer_type: str,
                 resolution: str = '2mm',
                 atlas_type: str = 'standard'):
        """
        Initialize SUVr calculator.
        
        Parameters
        ----------
        tracer_type : str
            Type of tracer ('PIB', 'FBB', 'FBP', 'FMM', 'FBZ')
        resolution : str
            Image resolution ('1mm' or '2mm')
        atlas_type : str
            Atlas type ('standard', 'aal', 'hammers', etc.)
        """
        self.tracer_type = tracer_type
        self.resolution = resolution
        self.atlas_type = atlas_type
        
        # Get tracer information
        self.tracer = TracerRegistry.get_tracer(tracer_type)
        
        # Load atlas
        self.atlas_manager = AtlasManager(resolution=resolution, atlas_type=atlas_type)
        
        # Set reference region based on tracer
        self.reference_region = self.tracer.reference_region
    
    def calculate_suvr(self, 
                      pet_path: Union[str, Path],
                      target_regions: List[str],
                      reference_region: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate SUVr values for target regions.
        
        Parameters
        ----------
        pet_path : str or Path
            Path to PET image
        target_regions : list
            List of target region names
        reference_region : str, optional
            Reference region name (overrides tracer default)
            
        Returns
        -------
        dict
            Dictionary mapping region names to SUVr values
        """
        # Load PET image
        pet_img = ants.image_read(str(pet_path))
        
        # Set reference region
        ref_region = reference_region or self.reference_region
        
        # Get reference mask
        try:
            ref_mask = self.atlas_manager.get_region_mask(ref_region)
        except ValueError:
            # Try as region group
            ref_mask = self.atlas_manager.get_region_group_mask(ref_region)
        
        # Calculate reference mean
        ref_mean = self._calculate_mean_intensity(pet_img, ref_mask)
        
        if ref_mean == 0:
            raise ValueError(f"Reference region '{ref_region}' has zero mean intensity")
        
        # Calculate SUVr for each target region
        suvr_values = {}
        
        for region_name in target_regions:
            try:
                # Try as individual region
                region_mask = self.atlas_manager.get_region_mask(region_name)
                region_mean = self._calculate_mean_intensity(pet_img, region_mask)
                suvr_values[region_name] = region_mean / ref_mean
            except ValueError:
                try:
                    # Try as region group
                    region_mask = self.atlas_manager.get_region_group_mask(region_name)
                    region_mean = self._calculate_mean_intensity(pet_img, region_mask)
                    suvr_values[region_name] = region_mean / ref_mean
                except ValueError:
                    print(f"Warning: Region '{region_name}' not found in atlas")
                    suvr_values[region_name] = float('nan')
        
        # Add reference region information
        suvr_values['reference_region'] = ref_region
        suvr_values['reference_mean'] = ref_mean
        
        return suvr_values
    
    def calculate_all_regions_suvr(self, 
                                  pet_path: Union[str, Path],
                                  reference_region: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate SUVr values for all regions in the atlas.
        
        Parameters
        ----------
        pet_path : str or Path
            Path to PET image
        reference_region : str, optional
            Reference region name (overrides tracer default)
            
        Returns
        -------
        dict
            Dictionary mapping region names to SUVr values
        """
        # Get all regions
        regions = self.atlas_manager.list_regions()
        region_groups = self.atlas_manager.list_region_groups()
        
        # Calculate SUVr for all regions
        all_regions = regions + region_groups
        
        # Remove reference region from target regions
        ref_region = reference_region or self.reference_region
        if ref_region in all_regions:
            all_regions.remove(ref_region)
        
        return self.calculate_suvr(pet_path, all_regions, reference_region)
    
    def calculate_composite_suvr(self, 
                               pet_path: Union[str, Path],
                               composite_name: str = 'cortical_composite',
                               reference_region: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate SUVr for a composite region.
        
        Parameters
        ----------
        pet_path : str or Path
            Path to PET image
        composite_name : str
            Name of the composite region
        reference_region : str, optional
            Reference region name (overrides tracer default)
            
        Returns
        -------
        dict
            Dictionary containing SUVr value for the composite region
        """
        # Calculate SUVr for the composite region
        return self.calculate_suvr(pet_path, [composite_name], reference_region)
    
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