"""
Reference regions and templates for Centiloid calculation.

This module contains definitions of standard reference regions and templates
used in the Centiloid calculation process. It includes:
- Standard space templates
- Reference region definitions
- Target region definitions
- Tracer-specific reference data
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path

class CentiloidReference:
    """
    Class managing reference regions and templates for Centiloid calculation.
    """
    
    # Standard template paths (relative to package installation)
    TEMPLATES = {
        'MNI152': 'templates/mni152_t1_2mm.nii.gz',
        'CTX_COMPOSITE': 'templates/centiloid_ctx_composite.nii.gz',
        'WM_CEREBELLUM': 'templates/centiloid_wm_cerebellum.nii.gz'
    }
    
    # Reference region definitions for different tracers
    REFERENCE_REGIONS = {
        'PIB': {
            'name': 'whole_cerebellum',
            'description': 'Whole cerebellum gray and white matter',
            'mask_file': 'masks/pib_ref_whole_cerebellum.nii.gz'
        },
        'FBB': {
            'name': 'cerebellar_cortex',
            'description': 'Cerebellar cortex (gray matter)',
            'mask_file': 'masks/fbb_ref_cerebellar_cortex.nii.gz'
        },
        'FBP': {
            'name': 'cerebellar_white_matter',
            'description': 'Cerebellar white matter',
            'mask_file': 'masks/fbp_ref_cerebellum_wm.nii.gz'
        },
        'FMM': {
            'name': 'whole_cerebellum',
            'description': 'Whole cerebellum gray and white matter',
            'mask_file': 'masks/fmm_ref_whole_cerebellum.nii.gz'
        }
    }
    
    # Target region definitions
    TARGET_REGIONS = {
        'STANDARD': {
            'name': 'cortical_composite',
            'description': 'Standard centiloid cortical composite region',
            'mask_file': 'masks/centiloid_target_composite.nii.gz',
            'regions': [
                'frontal_cortex',
                'temporal_cortex',
                'parietal_cortex',
                'anterior_cingulate',
                'posterior_cingulate',
                'precuneus'
            ]
        }
    }
    
    def __init__(self, package_dir=None):
        """
        Initialize reference manager.
        
        Parameters
        ----------
        package_dir : str, optional
            Path to package installation directory
        """
        if package_dir is None:
            package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.package_dir = Path(package_dir)
        
    def get_template_path(self, template_name):
        """
        Get full path to template file.
        
        Parameters
        ----------
        template_name : str
            Name of template ('MNI152', 'CTX_COMPOSITE', 'WM_CEREBELLUM')
            
        Returns
        -------
        Path
            Full path to template file
        """
        if template_name not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}")
        return self.package_dir / self.TEMPLATES[template_name]
    
    def get_reference_region(self, tracer):
        """
        Get reference region information for a specific tracer.
        
        Parameters
        ----------
        tracer : str
            PET tracer type ('PIB', 'FBB', 'FBP', 'FMM')
            
        Returns
        -------
        dict
            Reference region information including mask path
        """
        if tracer not in self.REFERENCE_REGIONS:
            raise ValueError(f"Unsupported tracer: {tracer}")
        ref_info = self.REFERENCE_REGIONS[tracer].copy()
        ref_info['mask_path'] = self.package_dir / ref_info['mask_file']
        return ref_info
    
    def get_target_region(self, region_name='STANDARD'):
        """
        Get target region information.
        
        Parameters
        ----------
        region_name : str
            Name of target region definition
            
        Returns
        -------
        dict
            Target region information including mask path
        """
        if region_name not in self.TARGET_REGIONS:
            raise ValueError(f"Unknown target region: {region_name}")
        target_info = self.TARGET_REGIONS[region_name].copy()
        target_info['mask_path'] = self.package_dir / target_info['mask_file']
        return target_info
    
    def load_reference_mask(self, tracer):
        """
        Load reference region mask for a specific tracer.
        
        Parameters
        ----------
        tracer : str
            PET tracer type ('PIB', 'FBB', 'FBP', 'FMM')
            
        Returns
        -------
        nibabel.Nifti1Image
            Reference region mask image
        """
        ref_info = self.get_reference_region(tracer)
        return nib.load(ref_info['mask_path'])
    
    def load_target_mask(self, region_name='STANDARD'):
        """
        Load target region mask.
        
        Parameters
        ----------
        region_name : str
            Name of target region definition
            
        Returns
        -------
        nibabel.Nifti1Image
            Target region mask image
        """
        target_info = self.get_target_region(region_name)
        return nib.load(target_info['mask_path'])
    
    @staticmethod
    def get_reference_values():
        """
        Get reference values for Centiloid calculation.
        
        Returns
        -------
        dict
            Dictionary containing reference values for young controls (YC)
            and Alzheimer's disease (AD) subjects
        """
        return {
            'PIB': {
                'YC_mean': 1.0,
                'YC_sd': 0.067,
                'AD_mean': 2.07,
                'AD_sd': 0.236
            },
            'FBB': {
                'YC_mean': 1.08,
                'YC_sd': 0.076,
                'AD_mean': 1.67,
                'AD_sd': 0.158
            },
            'FBP': {
                'YC_mean': 1.06,
                'YC_sd': 0.054,
                'AD_mean': 1.71,
                'AD_sd': 0.177
            },
            'FMM': {
                'YC_mean': 1.03,
                'YC_sd': 0.065,
                'AD_mean': 1.61,
                'AD_sd': 0.169
            }
        }
