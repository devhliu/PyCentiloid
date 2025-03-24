"""
Segmentation module for PyCentiloid.

This module provides functions for brain segmentation and extraction
using deep learning approaches (BrainChop) and template-based methods.
"""

import os
import ants
import numpy as np
from pathlib import Path
from typing import Union, Optional, Literal, Dict, List, Tuple

# Try to import optional dependencies
try:
    import tensorflow as tf
    import brainchop
    HAS_BRAINCHOP = True
except ImportError:
    HAS_BRAINCHOP = False

class BrainSegmenter:
    """
    Class for brain segmentation and extraction.
    
    Supports both deep learning based segmentation (BrainChop)
    and template-based methods.
    """
    
    def __init__(self, 
                 method: Literal['deep', 'template'] = 'deep',
                 modality: Literal['PET', 'CT', 'MRI'] = 'MRI',
                 resolution: Literal['1mm', '2mm'] = '1mm'):
        """
        Initialize brain segmenter.
        
        Parameters
        ----------
        method : str
            Segmentation method ('deep' or 'template')
        modality : str
            Image modality ('PET', 'CT', 'MRI')
        resolution : str
            Image resolution ('1mm' or '2mm')
        """
        self.method = method
        self.modality = modality
        self.resolution = resolution
        
        # Check if deep learning is available
        if method == 'deep' and not HAS_BRAINCHOP:
            print("Warning: BrainChop not available. Falling back to template-based segmentation.")
            self.method = 'template'
        
        # Load templates and models
        self._load_resources()
    
    def _load_resources(self):
        """Load appropriate templates and models."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Load templates
        template_dir = package_dir / 'data' / 'templates'
        mask_dir = package_dir / 'data' / 'masks'
        
        # Set paths based on modality and resolution
        if self.modality == 'MRI':
            self.template_path = template_dir / f"mri_t1w_template_{self.resolution}.nii.gz"
            self.brain_mask_path = mask_dir / f"mri_brain_mask_{self.resolution}.nii.gz"
        elif self.modality == 'PET':
            self.template_path = template_dir / f"pet_template_{self.resolution}.nii.gz"
            self.brain_mask_path = mask_dir / f"pet_brain_mask_{self.resolution}.nii.gz"
        else:  # CT
            self.template_path = template_dir / f"ct_template_{self.resolution}.nii.gz"
            self.brain_mask_path = mask_dir / f"ct_brain_mask_{self.resolution}.nii.gz"
        
        # Load deep learning model if needed
        if self.method == 'deep' and HAS_BRAINCHOP:
            model_dir = package_dir / 'models'
            if self.modality == 'MRI':
                self.model_path = model_dir / "brainchop_t1w.h5"
                if self.model_path.exists():
                    self.model = brainchop.load_model(str(self.model_path))
                else:
                    print(f"Warning: Model file {self.model_path} not found. Downloading...")
                    self.model = brainchop.download_model("t1w")
            else:
                # For PET and CT, we'll use template-based methods
                self.method = 'template'
                print(f"Deep learning segmentation not available for {self.modality}. Using template-based method.")
    
    def extract_brain(self, 
                     image_path: Union[str, Path],
                     output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Extract brain from image (Brain Extraction Tool - BET).
        
        Parameters
        ----------
        image_path : str or Path
            Path to input image
        output_path : str or Path, optional
            Path to save brain-extracted image
            
        Returns
        -------
        str
            Path to brain-extracted image
        """
        # Create output path if not provided
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_brain{input_path.suffix}"
        
        # Perform brain extraction
        if self.method == 'deep' and HAS_BRAINCHOP and self.modality == 'MRI':
            return self._deep_brain_extraction(image_path, output_path)
        else:
            return self._template_brain_extraction(image_path, output_path)
    
    def _deep_brain_extraction(self, 
                              image_path: Union[str, Path],
                              output_path: Union[str, Path]) -> str:
        """Perform brain extraction using BrainChop."""
        if not HAS_BRAINCHOP:
            raise ImportError("BrainChop is required for deep learning segmentation")
        
        # Load image
        img = ants.image_read(str(image_path))
        
        # Preprocess for BrainChop
        img_np = img.numpy()
        img_preprocessed = brainchop.preprocess_image(img_np)
        
        # Run segmentation
        segmentation = self.model.predict(img_preprocessed)
        brain_mask = brainchop.postprocess_prediction(segmentation)
        
        # Apply mask to original image
        brain_img_np = img_np * brain_mask
        brain_img = ants.from_numpy(brain_img_np, origin=img.origin, 
                                   spacing=img.spacing, direction=img.direction)
        
        # Save result
        brain_img.to_filename(str(output_path))
        
        return str(output_path)
    
    def _template_brain_extraction(self, 
                                  image_path: Union[str, Path],
                                  output_path: Union[str, Path]) -> str:
        """Perform brain extraction using template-based method."""
        # Load image and template
        img = ants.image_read(str(image_path))
        template = ants.image_read(str(self.template_path))
        brain_mask = ants.image_read(str(self.brain_mask_path))
        
        # Register template to image
        registration = ants.registration(
            fixed=img,
            moving=template,
            type_of_transform='SyN'
        )
        
        # Transform brain mask to image space
        transformed_mask = ants.apply_transforms(
            fixed=img,
            moving=brain_mask,
            transformlist=registration['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        
        # Apply mask to original image
        brain_img = img.clone() * (transformed_mask > 0)
        
        # Save result
        brain_img.to_filename(str(output_path))
        
        return str(output_path)
    
    def segment_brain_regions(self, 
                             image_path: Union[str, Path],
                             output_dir: Optional[Union[str, Path]] = None) -> Dict[str, str]:
        """
        Segment brain into anatomical regions.
        
        Parameters
        ----------
        image_path : str or Path
            Path to input image
        output_dir : str or Path, optional
            Directory to save segmentation results
            
        Returns
        -------
        dict
            Dictionary mapping region names to file paths
        """
        # Only supported for MRI currently
        if self.modality != 'MRI':
            raise ValueError(f"Brain region segmentation not supported for {self.modality}")
        
        # Create output directory if not provided
        if output_dir is None:
            input_path = Path(image_path)
            output_dir = input_path.parent / f"{input_path.stem}_segments"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Perform segmentation
        if self.method == 'deep' and HAS_BRAINCHOP:
            return self._deep_brain_segmentation(image_path, output_dir)
        else:
            return self._template_brain_segmentation(image_path, output_dir)
    
    def _deep_brain_segmentation(self, 
                                image_path: Union[str, Path],
                                output_dir: Union[str, Path]) -> Dict[str, str]:
        """Perform brain segmentation using BrainChop."""
        if not HAS_BRAINCHOP:
            raise ImportError("BrainChop is required for deep learning segmentation")
        
        # Load image
        img = ants.image_read(str(image_path))
        
        # Preprocess for BrainChop
        img_np = img.numpy()
        img_preprocessed = brainchop.preprocess_image(img_np)
        
        # Run segmentation
        segmentation = self.model.predict(img_preprocessed)
        region_masks = brainchop.get_region_masks(segmentation)
        
        # Save each region
        region_paths = {}
        for region_name, mask in region_masks.items():
            # Create region image
            region_img = ants.from_numpy(mask.astype(np.float32), origin=img.origin, 
                                        spacing=img.spacing, direction=img.direction)
            
            # Save region
            region_path = output_dir / f"{region_name}.nii.gz"
            region_img.to_filename(str(region_path))
            region_paths[region_name] = str(region_path)
        
        return region_paths
    
    def _template_brain_segmentation(self, 
                                    image_path: Union[str, Path],
                                    output_dir: Union[str, Path]) -> Dict[str, str]:
        """Perform brain segmentation using template-based method."""
        # Load image
        img = ants.image_read(str(image_path))
        
        # Get atlas directory
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        atlas_dir = package_dir / 'data' / 'atlases' / f"{self.resolution}"
        
        # Load atlas template and labels
        atlas_template = ants.image_read(str(atlas_dir / "atlas_template.nii.gz"))
        atlas_labels = ants.image_read(str(atlas_dir / "atlas_labels.nii.gz"))
        
        # Register atlas to image
        registration = ants.registration(
            fixed=img,
            moving=atlas_template,
            type_of_transform='SyN'
        )
        
        # Transform atlas labels to image space
        transformed_labels = ants.apply_transforms(
            fixed=img,
            moving=atlas_labels,
            transformlist=registration['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        
        # Load region definitions
        region_def_path = atlas_dir / "region_definitions.txt"
        region_defs = {}
        with open(region_def_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        region_defs[int(parts[0])] = parts[1]
        
        # Extract each region
        region_paths = {}
        labels_np = transformed_labels.numpy()
        
        for label_id, region_name in region_defs.items():
            # Create region mask
            region_mask = (labels_np == label_id).astype(np.float32)
            region_img = ants.from_numpy(region_mask, origin=img.origin, 
                                        spacing=img.spacing, direction=img.direction)
            
            # Save region
            region_path = output_dir / f"{region_name}.nii.gz"
            region_img.to_filename(str(region_path))
            region_paths[region_name] = str(region_path)
        
        return region_paths

# Convenience functions
def extract_brain(image_path: Union[str, Path],
                 modality: Literal['PET', 'CT', 'MRI'] = 'MRI',
                 method: Literal['deep', 'template'] = 'deep',
                 output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Extract brain from image (Brain Extraction Tool - BET).
    
    Parameters
    ----------
    image_path : str or Path
        Path to input image
    modality : str
        Image modality ('PET', 'CT', 'MRI')
    method : str
        Segmentation method ('deep' or 'template')
    output_path : str or Path, optional
        Path to save brain-extracted image
        
    Returns
    -------
    str
        Path to brain-extracted image
    """
    segmenter = BrainSegmenter(method=method, modality=modality)
    return segmenter.extract_brain(image_path, output_path)