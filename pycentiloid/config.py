"""
Configuration settings for pycentiloid package.

This module contains all configuration settings including:
- File paths and directories
- Processing parameters
- User settings
- Logging configuration
"""

import os
import platform
from pathlib import Path
from datetime import datetime

class Config:
    """
    Configuration class for pycentiloid package.
    """
    
    # Version and metadata
    VERSION = "0.1.0"
    CREATED_DATE = "2025-02-07 08:44:02"
    CREATED_BY = "devhliu"
    
    # Base directories
    ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = ROOT_DIR / "data"
    TEMPLATE_DIR = DATA_DIR / "templates"
    MASK_DIR = DATA_DIR / "masks"
    OUTPUT_DIR = ROOT_DIR / "output"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, TEMPLATE_DIR, MASK_DIR, OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Template configurations with multiple resolutions
    TEMPLATES = {
        'MNI152': {
            '1mm': TEMPLATE_DIR / 'mni152_t1_1mm.nii.gz',
            '2mm': TEMPLATE_DIR / 'mni152_t1_2mm.nii.gz'
        },
        'MNI152_PET': {
            '1mm': TEMPLATE_DIR / 'mni152_pet_1mm.nii.gz',
            '2mm': TEMPLATE_DIR / 'mni152_pet_2mm.nii.gz'
        },
        'MNI152_CT': {
            '1mm': TEMPLATE_DIR / 'mni152_ct_1mm.nii.gz',
            '2mm': TEMPLATE_DIR / 'mni152_ct_2mm.nii.gz'
        },
        'AAL': {
            '1mm': TEMPLATE_DIR / 'aal_1mm.nii.gz',
            '2mm': TEMPLATE_DIR / 'aal_2mm.nii.gz'
        },
        'CTX_COMPOSITE': {
            '1mm': TEMPLATE_DIR / 'centiloid_ctx_composite_1mm.nii.gz',
            '2mm': TEMPLATE_DIR / 'centiloid_ctx_composite_2mm.nii.gz'
        },
        'WM_CEREBELLUM': {
            '1mm': TEMPLATE_DIR / 'centiloid_wm_cerebellum_1mm.nii.gz',
            '2mm': TEMPLATE_DIR / 'centiloid_wm_cerebellum_2mm.nii.gz'
        }
    }
    
    # Registration parameters for different modalities
    REGISTRATION = {
        'DEFAULT': {
            'type_of_transform': 'SyNRA',
            'grad_step': 0.1,
            'flow_sigma': 3,
            'total_sigma': 0,
            'aff_metric': 'mattes',
            'syn_metric': 'mattes',
            'reg_iterations': (40, 20, 0),
            'aff_iterations': (2100, 1200, 1200, 10),
            'aff_shrink_factors': (6, 4, 2, 1),
            'aff_smoothing_sigmas': (4, 2, 1, 0)
        },
        'PET': {
            'type_of_transform': 'SyNRA',
            'grad_step': 0.1,
            'flow_sigma': 3,
            'aff_metric': 'mattes',
            'syn_metric': 'mattes',
            'reg_iterations': (40, 20, 10),
            'aff_iterations': (2100, 1200, 1200, 10),
            'aff_shrink_factors': (8, 4, 2, 1),
            'aff_smoothing_sigmas': (4, 2, 1, 0),
            'smoothing_sigmas': [4, 2, 1],
            'shrink_factors': [4, 2, 1]
        },
        'CT': {
            'type_of_transform': 'SyNRA',
            'grad_step': 0.1,
            'flow_sigma': 2,
            'aff_metric': 'mattes',
            'syn_metric': 'meansquares',
            'reg_iterations': (50, 25, 10),
            'aff_iterations': (2100, 1200, 1200, 10),
            'aff_shrink_factors': (6, 4, 2, 1),
            'aff_smoothing_sigmas': (3, 2, 1, 0),
            'smoothing_sigmas': [3, 2, 1],
            'shrink_factors': [3, 2, 1]
        },
        'T1': {
            'type_of_transform': 'SyN',
            'grad_step': 0.1,
            'flow_sigma': 3,
            'aff_metric': 'mattes',
            'syn_metric': 'cc',
            'reg_iterations': (100, 70, 50),
            'aff_iterations': (2100, 1200, 1200, 10),
            'aff_shrink_factors': (8, 4, 2, 1),
            'aff_smoothing_sigmas': (3, 2, 1, 0),
            'smoothing_sigmas': [4, 2, 0],
            'shrink_factors': [4, 2, 1]
        }
    }
    
    # Modality-specific parameters
    MODALITY_PARAMS = {
        'PET': {
            'template': 'mni152_pet_2mm.nii.gz',
            'registration': {
                'cost_function': 'mutual_information',
                'smoothing_fwhm': 8
            }
        },
        'CT': {
            'template': 'mni152_ct_2mm.nii.gz',
            'registration': {
                'cost_function': 'mean_squares',
                'smoothing_fwhm': 4
            }
        },
        'T1': {
            'template': 'mni152_t1_2mm.nii.gz',
            'registration': {
                'cost_function': 'cross_correlation',
                'smoothing_fwhm': 4
            }
        }
    }
# Add to Config class
    DICOM_SR = {
        'manufacturer': 'PyCentiloid',
        'manufacturer_model_name': 'Amyloid Analysis',
        'software_versions': '1.0.0',
        'institution_name': 'Research Institution',
        'template_codes': {
            'amyloid_analysis': {
                'value': '126700',
                'scheme_designator': 'DCM',
                'meaning': 'Amyloid PET Analysis Report'
            }
        }
    }