"""
Atlas management module for PyCentiloid.

This module provides functions for loading, manipulating, and visualizing
brain atlases used in the Centiloid calculation process.
"""

import os
import ants
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Any

class AtlasManager:
    """
    Class for managing brain atlases.
    
    Provides functionality for loading atlas data, extracting regions,
    and visualizing atlas information.
    """
    
    def __init__(self, 
                 resolution: str = '2mm',
                 atlas_type: str = 'standard'):
        """
        Initialize atlas manager.
        
        Parameters
        ----------
        resolution : str
            Atlas resolution ('1mm' or '2mm')
        atlas_type : str
            Atlas type ('standard', 'aal', 'hammers', etc.)
        """
        self.resolution = resolution
        self.atlas_type = atlas_type
        
        # Load atlas data
        self._load_atlas()
    
    def _load_atlas(self):
        """Load atlas data."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        atlas_dir = package_dir / 'data' / 'atlases'
        
        # Set atlas paths based on type and resolution
        atlas_subdir = atlas_dir / self.atlas_type / self.resolution
        
        # Check if atlas exists
        if not atlas_subdir.exists():
            raise ValueError(f"Atlas not found: {self.atlas_type} at {self.resolution}")
        
        # Load atlas template and labels
        self.template_path = atlas_subdir / "template.nii.gz"
        self.labels_path = atlas_subdir / "labels.nii.gz"
        self.metadata_path = atlas_subdir / "metadata.json"
        
        # Load images
        self.template = ants.image_read(str(self.template_path))
        self.labels = ants.image_read(str(self.labels_path))
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Extract region information
        self.regions = self.metadata.get('regions', {})
        self.region_groups = self.metadata.get('region_groups', {})
    
    def get_region_mask(self, region_name: str) -> ants.ANTsImage:
        """
        Get mask for a specific region.
        
        Parameters
        ----------
        region_name : str
            Name of the region
            
        Returns
        -------
        ants.ANTsImage
            Binary mask for the region
        """
        if region_name not in self.regions:
            raise ValueError(f"Region not found: {region_name}")
        
        # Get region label
        region_label = self.regions[region_name]['label']
        
        # Create binary mask
        labels_np = self.labels.numpy()
        mask_np = (labels_np == region_label).astype(np.float32)
        
        # Create ANTs image
        mask = ants.from_numpy(mask_np, origin=self.labels.origin,
                              spacing=self.labels.spacing,
                              direction=self.labels.direction)
        
        return mask
    
    def get_region_group_mask(self, group_name: str) -> ants.ANTsImage:
        """
        Get mask for a group of regions.
        
        Parameters
        ----------
        group_name : str
            Name of the region group
            
        Returns
        -------
        ants.ANTsImage
            Binary mask for the region group
        """
        if group_name not in self.region_groups:
            raise ValueError(f"Region group not found: {group_name}")
        
        # Get regions in group
        regions = self.region_groups[group_name]['regions']
        
        # Create empty mask
        mask_np = np.zeros_like(self.labels.numpy(), dtype=np.float32)
        
        # Add each region to mask
        for region_name in regions:
            if region_name in self.regions:
                region_label = self.regions[region_name]['label']
                mask_np[(self.labels.numpy() == region_label)] = 1.0
        
        # Create ANTs image
        mask = ants.from_numpy(mask_np, origin=self.labels.origin,
                              spacing=self.labels.spacing,
                              direction=self.labels.direction)
        
        return mask
    
    def save_region_mask(self, 
                        region_name: str, 
                        output_path: Union[str, Path]) -> str:
        """
        Save mask for a specific region.
        
        Parameters
        ----------
        region_name : str
            Name of the region
        output_path : str or Path
            Path to save mask
            
        Returns
        -------
        str
            Path to saved mask
        """
        mask = self.get_region_mask(region_name)
        mask.to_filename(str(output_path))
        return str(output_path)
    
    def save_region_group_mask(self, 
                              group_name: str, 
                              output_path: Union[str, Path]) -> str:
        """
        Save mask for a group of regions.
        
        Parameters
        ----------
        group_name : str
            Name of the region group
        output_path : str or Path
            Path to save mask
            
        Returns
        -------
        str
            Path to saved mask
        """
        mask = self.get_region_group_mask(group_name)
        mask.to_filename(str(output_path))
        return str(output_path)
    
    def list_regions(self) -> List[str]:
        """
        List all available regions.
        
        Returns
        -------
        list
            List of region names
        """
        return list(self.regions.keys())
    
    def list_region_groups(self) -> List[str]:
        """
        List all available region groups.
        
        Returns
        -------
        list
            List of region group names
        """
        return list(self.region_groups.keys())
    
    def get_region_info(self, region_name: str) -> Dict[str, Any]:
        """
        Get information about a specific region.
        
        Parameters
        ----------
        region_name : str
            Name of the region
            
        Returns
        -------
        dict
            Region information
        """
        if region_name not in self.regions:
            raise ValueError(f"Region not found: {region_name}")
        
        return self.regions[region_name]
    
    def get_region_group_info(self, group_name: str) -> Dict[str, Any]:
        """
        Get information about a specific region group.
        
        Parameters
        ----------
        group_name : str
            Name of the region group
            
        Returns
        -------
        dict
            Region group information
        """
        if group_name not in self.region_groups:
            raise ValueError(f"Region group not found: {group_name}")
        
        return self.region_groups[group_name]
    
    def get_region_by_name(self, name: str) -> Optional[AtlasRegion]:
        """
        Get a region by name.
        
        Parameters
        ----------
        name : str
            Region name
            
        Returns
        -------
        AtlasRegion or None
            Region object if found, None otherwise
        """
        for region in self.regions.values():
            if region.name == name:
                return region
        
        return None
    
    def get_region_group_by_name(self, name: str) -> Optional[RegionGroup]:
        """
        Get a region group by name.
        
        Parameters
        ----------
        name : str
            Group name
            
        Returns
        -------
        RegionGroup or None
            Group object if found, None otherwise
        """
        for group in self.region_groups.values():
            if group.name == name:
                return group
        
        return None
    
    def get_regions_by_hemisphere(self, hemisphere: str) -> List[AtlasRegion]:
        """
        Get regions by hemisphere.
        
        Parameters
        ----------
        hemisphere : str
            Hemisphere ('left', 'right', 'both')
            
        Returns
        -------
        list
            List of regions in the specified hemisphere
        """
        return [
            region for region in self.regions.values()
            if region.hemisphere == hemisphere
        ]
    
    def get_region_volume(self, region_id: str, voxel_volume: Optional[float] = None) -> float:
        """
        Get volume of a region in cubic millimeters.
        
        Parameters
        ----------
        region_id : str
            Region ID
        voxel_volume : float, optional
            Volume of a single voxel in cubic millimeters
            
        Returns
        -------
        float
            Volume of the region in cubic millimeters
        """
        mask = self.get_region_mask(region_id)
        
        # Count voxels
        voxel_count = np.sum(mask.numpy())
        
        # Calculate voxel volume if not provided
        if voxel_volume is None:
            voxel_volume = np.prod(mask.spacing)
        
        # Calculate region volume
        volume = voxel_count * voxel_volume
        
        return float(volume)
    
    def get_region_group_volume(self, group_id: str, voxel_volume: Optional[float] = None) -> float:
        """
        Get volume of a region group in cubic millimeters.
        
        Parameters
        ----------
        group_id : str
            Group ID
        voxel_volume : float, optional
            Volume of a single voxel in cubic millimeters
            
        Returns
        -------
        float
            Volume of the region group in cubic millimeters
        """
        mask = self.get_region_group_mask(group_id)
        
        # Count voxels
        voxel_count = np.sum(mask.numpy())
        
        # Calculate voxel volume if not provided
        if voxel_volume is None:
            voxel_volume = np.prod(mask.spacing)
        
        # Calculate region volume
        volume = voxel_count * voxel_volume
        
        return float(volume)
    
    def create_custom_region_group(self, 
                                  group_id: str,
                                  name: str,
                                  regions: List[str],
                                  metadata: Optional[Dict[str, Any]] = None) -> RegionGroup:
        """
        Create a custom region group.
        
        Parameters
        ----------
        group_id : str
            Unique identifier for the group
        name : str
            Descriptive name of the group
        regions : list
            List of region IDs in the group
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        RegionGroup
            Created region group
        """
        # Validate regions
        for region_id in regions:
            if region_id not in self.regions:
                raise ValueError(f"Region not found: {region_id}")
        
        # Create group
        group = RegionGroup(
            group_id=group_id,
            name=name,
            regions=regions,
            metadata=metadata
        )
        
        # Add to registry
        self.region_groups[group_id] = group
        
        # Save custom groups
        self._save_custom_groups()
        
        return group
    
    def _save_custom_groups(self):
        """Save custom region groups to configuration file."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = package_dir / 'config'
        config_dir.mkdir(exist_ok=True, parents=True)
        
        custom_groups_path = config_dir / f"custom_groups_{self.atlas_type}_{self.resolution}.json"
        
        # Get standard group IDs (this would depend on the atlas)
        standard_ids = set()  # This should be populated with standard group IDs
        
        # Filter custom groups
        custom_groups = [
            group.to_dict() for group_id, group in self.region_groups.items()
            if group_id not in standard_ids
        ]
        
        # Save to file
        with open(custom_groups_path, 'w') as f:
            json.dump(custom_groups, f, indent=2)
    
    def remove_region_group(self, group_id: str):
        """
        Remove a region group.
        
        Parameters
        ----------
        group_id : str
            Group ID
        """
        if group_id not in self.region_groups:
            raise ValueError(f"Region group not found: {group_id}")
        
        # Don't allow removing standard groups (this would depend on the atlas)
        standard_ids = set()  # This should be populated with standard group IDs
        if group_id in standard_ids:
            raise ValueError(f"Cannot remove standard region group: {group_id}")
        
        del self.region_groups[group_id]
        
        # Save custom groups
        self._save_custom_groups()
    
    def get_centiloid_regions(self) -> Dict[str, str]:
        """
        Get standard Centiloid regions.
        
        Returns
        -------
        dict
            Dictionary mapping region names to region IDs
        """
        # Standard Centiloid regions
        centiloid_regions = {
            'whole_cerebellum': 'whole_cerebellum',
            'cerebellar_cortex': 'cerebellar_cortex',
            'cerebellar_white_matter': 'cerebellar_white_matter',
            'pons': 'pons',
            'cortical_composite': 'cortical_composite',
            'global_composite': 'global_composite'
        }
        
        # Filter to only include regions that exist in this atlas
        return {
            name: region_id for name, region_id in centiloid_regions.items()
            if region_id in self.regions or region_id in self.region_groups
        }
    
    def visualize_region(self, region_id: str, output_path: Optional[Union[str, Path]] = None):
        """
        Visualize a region.
        
        Parameters
        ----------
        region_id : str
            Region ID
        output_path : str or Path, optional
            Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            from nilearn import plotting
        except ImportError:
            raise ImportError("Visualization requires matplotlib and nilearn")
        
        # Get region mask
        try:
            mask = self.get_region_mask(region_id)
        except ValueError:
            # Try as region group
            mask = self.get_region_group_mask(region_id)
        
        # Convert to numpy array
        mask_data = mask.numpy()
        
        # Create visualization
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Get middle slices
        x_mid = mask_data.shape[0] // 2
        y_mid = mask_data.shape[1] // 2
        z_mid = mask_data.shape[2] // 2
        
        # Plot slices
        ax[0].imshow(mask_data[x_mid, :, :].T, origin='lower', cmap='hot')
        ax[0].set_title(f"Sagittal (x={x_mid})")
        
        ax[1].imshow(mask_data[:, y_mid, :].T, origin='lower', cmap='hot')
        ax[1].set_title(f"Coronal (y={y_mid})")
        
        ax[2].imshow(mask_data[:, :, z_mid].T, origin='lower', cmap='hot')
        ax[2].set_title(f"Axial (z={z_mid})")
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()