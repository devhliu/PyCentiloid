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
    ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = ROOT_DIR / "data"
    TEMPLATE_DIR = DATA_DIR / "templates"
    MASK_DIR = DATA_DIR / "masks"
    OUTPUT_DIR = ROOT_DIR / "output"
    
    # Template configurations
    # Template configurations - update to support multiple resolutions
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
    
    # Tracer-specific templates and masks
    TRACER_DATA = {
        'PIB': {
            'template': {
                '1mm': TEMPLATE_DIR / 'pib/pib_template_1mm.nii.gz',
                '2mm': TEMPLATE_DIR / 'pib/pib_template_2mm.nii.gz'
            },
            'reference_mask': {
                '1mm': MASK_DIR / 'pib/cerebellum_1mm.nii.gz',
                '2mm': MASK_DIR / 'pib/cerebellum_2mm.nii.gz'
            },
            'target_mask': {
                '1mm': MASK_DIR / 'pib/cortical_1mm.nii.gz',
                '2mm': MASK_DIR / 'pib/cortical_2mm.nii.gz'
            }
        },
        # Similar structure for FBB, FBP, FMM, FBZ
    }
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, TEMPLATE_DIR, MASK_DIR, OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Standard templates
    TEMPLATES = {
        'MNI152': TEMPLATE_DIR / 'mni152_t1_2mm.nii.gz',
        'CTX_COMPOSITE': TEMPLATE_DIR / 'centiloid_ctx_composite.nii.gz',
        'WM_CEREBELLUM': TEMPLATE_DIR / 'centiloid_wm_cerebellum.nii.gz'
    }
    
    # Registration parameters - enhanced for different modalities
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
    
        'MRI': {
            'type_of_transform': 'SyN',
            'grad_step': 0.1,
            'syn_metric': 'CC'
        }
    }
    
    # Tracer-specific parameters
    TRACERS = {
        'PIB': {
            'name': 'Pittsburgh Compound B',
            'half_life': 20.4,
            'ref_region': 'whole_cerebellum',
            'scaling_factor': 1.0,
            'dynamic_frames': (40, 70),
            'thresholds': {'positive': 1.42, 'negative': 1.08},
            'template': TEMPLATE_DIR / 'pib_template_2mm.nii.gz',
            'reference_mask': MASK_DIR / 'pib_cerebellum_2mm.nii.gz',
            'target_mask': MASK_DIR / 'pib_cortical_2mm.nii.gz'
        },
        'FBB': {
            'name': 'Florbetaben',
            'half_life': 109.8,
            'ref_region': 'cerebellar_cortex',
            'scaling_factor': 0.985,
            'dynamic_frames': (90, 110),
            'thresholds': {'positive': 1.48, 'negative': 1.20},
            'template': TEMPLATE_DIR / 'fbb_template_2mm.nii.gz',
            'reference_mask': MASK_DIR / 'fbb_cerebellum_2mm.nii.gz',
            'target_mask': MASK_DIR / 'fbb_cortical_2mm.nii.gz'
        },
        'FBP': {
            'name': 'Florbetapir',
            'half_life': 109.8,
            'ref_region': 'cerebellar_white_matter',
            'scaling_factor': 0.994,
            'dynamic_frames': (50, 70),
            'thresholds': {'positive': 1.47, 'negative': 1.11},
            'template': TEMPLATE_DIR / 'fbp_template_2mm.nii.gz',
            'reference_mask': MASK_DIR / 'fbp_cerebellum_2mm.nii.gz',
            'target_mask': MASK_DIR / 'fbp_cortical_2mm.nii.gz'
        },
        'FMM': {
            'name': 'Flutemetamol',
            'half_life': 109.8,
            'ref_region': 'whole_cerebellum',
            'scaling_factor': 0.979,
            'dynamic_frames': (85, 105),
            'thresholds': {'positive': 1.56, 'negative': 1.28},
            'template': TEMPLATE_DIR / 'fmm_template_2mm.nii.gz',
            'reference_mask': MASK_DIR / 'fmm_cerebellum_2mm.nii.gz',
            'target_mask': MASK_DIR / 'fmm_cortical_2mm.nii.gz'
        },
        'FBZ': {
            'name': 'Florbetazine',
            'half_life': 109.8,
            'ref_region': 'cerebellar_cortex',
            'scaling_factor': 0.990,
            'dynamic_frames': (90, 110),
            'thresholds': {'positive': 1.43, 'negative': 1.17},
            'template': TEMPLATE_DIR / 'fbz_template_2mm.nii.gz',
            'reference_mask': MASK_DIR / 'fbz_cerebellum_2mm.nii.gz',
            'target_mask': MASK_DIR / 'fbz_cortical_2mm.nii.gz'
        }
    }
    
    # Reference region definitions
    REFERENCE_REGIONS = {
        'whole_cerebellum': {
            'description': 'Whole cerebellum (gray and white matter)',
            'mask_file': 'whole_cerebellum_mask.nii.gz'
        },
        'cerebellar_cortex': {
            'description': 'Cerebellar cortex (gray matter)',
            'mask_file': 'cerebellar_cortex_mask.nii.gz'
        },
        'cerebellar_white_matter': {
            'description': 'Cerebellar white matter',
            'mask_file': 'cerebellum_wm_mask.nii.gz'
        }
    }
    
    # Logging configuration
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
            'file': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': str(OUTPUT_DIR / 'pycentiloid.log'),
                'mode': 'a',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['default', 'file'],
                'level': 'INFO',
                'propagate': True
            }
        }
    }
    
    # System-specific settings
    SYSTEM = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'num_cores': os.cpu_count(),
        'memory_limit': '8G'  # Default memory limit for processing
    }
    
    @classmethod
    def get_tracer_config(cls, tracer):
        """
        Get configuration for specific tracer.
        
        Parameters
        ----------
        tracer : str
            Tracer name ('PIB', 'FBB', 'FBP', 'FMM')
            
        Returns
        -------
        dict
            Tracer-specific configuration
        """
        if tracer not in cls.TRACERS:
            raise ValueError(f"Unsupported tracer: {tracer}")
        return cls.TRACERS[tracer]
    
    @classmethod
    def get_reference_region(cls, region_name):
        """
        Get reference region configuration.
        
        Parameters
        ----------
        region_name : str
            Name of reference region
            
        Returns
        -------
        dict
            Reference region configuration
        """
        if region_name not in cls.REFERENCE_REGIONS:
            raise ValueError(f"Unknown reference region: {region_name}")
        return cls.REFERENCE_REGIONS[region_name]
    
    @classmethod
    def get_processing_params(cls, step):
        """
        Get processing parameters for specific step.
        
        Parameters
        ----------
        step : str
            Processing step ('REGISTRATION', 'HARMONIZATION', 
                           'NORMALIZATION', 'SEGMENTATION')
            
        Returns
        -------
        dict
            Processing parameters for specified step
        """
        if step not in cls.PROCESSING:
            raise ValueError(f"Unknown processing step: {step}")
        return cls.PROCESSING[step]
    
    @classmethod
    def get_system_info(cls):
        """
        Get system-specific information.
        
        Returns
        -------
        dict
            System configuration and information
        """
        return cls.SYSTEM
    
    # Add PET-only specific parameters for each tracer
    # Add modality-specific parameters
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