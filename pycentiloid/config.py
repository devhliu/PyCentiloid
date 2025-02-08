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
    TEMPLATE_DIR = ROOT_DIR / "templates"
    MASK_DIR = ROOT_DIR / "masks"
    OUTPUT_DIR = ROOT_DIR / "output"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, TEMPLATE_DIR, MASK_DIR, OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Template files
    TEMPLATES = {
        'MNI152': str(TEMPLATE_DIR / 'mni152_t1_2mm.nii.gz'),
        'CTX_COMPOSITE': str(TEMPLATE_DIR / 'centiloid_ctx_composite.nii.gz'),
        'WM_CEREBELLUM': str(TEMPLATE_DIR / 'centiloid_wm_cerebellum.nii.gz')
    }
    
    # Processing parameters
    PROCESSING = {
        # Registration parameters
        'REGISTRATION': {
            'cost_function': 'mutual_information',
            'interpolation': 'linear',
            'dof': 12,  # degrees of freedom
            'convergence_threshold': 1e-8,
            'convergence_window': 10
        },
        
        # Harmonization parameters
        'HARMONIZATION': {
            'target_resolution_min': 5.0,  # mm
            'target_resolution_max': 6.0,  # mm
            'smoothing_levels': [8, 6, 4],  # mm
            'combat_parameters': {
                'parametric': True,
                'eb': True,  # empirical Bayes
                'mean_only': False
            }
        },
        
        # Normalization parameters
        'NORMALIZATION': {
            'target_voxel_size': (2.0, 2.0, 2.0),  # mm
            'interpolation': 'linear',
            'cost_function': 'mutual_information'
        },
        
        # Segmentation parameters
        'SEGMENTATION': {
            'algorithm': 'fast',  # FSL FAST
            'number_of_classes': 3,
            'bias_field_correction': True,
            'bias_field_order': 3
        }
    }
    
    # Tracer-specific parameters
    TRACERS = {
        'PIB': {
            'half_life': 20.4,  # minutes
            'ref_region': 'whole_cerebellum',
            'scaling_factor': 1.0,
            'recommended_frames': (40, 70),  # minutes
            'thresholds': {
                'positive': 1.42,
                'negative': 1.08
            }
        },
        'FBB': {
            'half_life': 109.8,
            'ref_region': 'cerebellar_cortex',
            'scaling_factor': 0.985,
            'recommended_frames': (90, 110),
            'thresholds': {
                'positive': 1.48,
                'negative': 1.20
            }
        },
        'FBP': {
            'half_life': 109.8,
            'ref_region': 'cerebellar_white_matter',
            'scaling_factor': 0.994,
            'recommended_frames': (45, 65),
            'thresholds': {
                'positive': 1.47,
                'negative': 1.21
            }
        },
        'FMM': {
            'half_life': 109.8,
            'ref_region': 'whole_cerebellum',
            'scaling_factor': 0.979,
            'recommended_frames': (90, 110),
            'thresholds': {
                'positive': 1.45,
                'negative': 1.19
            }
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
