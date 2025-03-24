"""
Tracer-specific processing pipeline for PET-only workflow.
"""

import os
from pathlib import Path
import numpy as np
import nibabel as nib
from ..config import Config
from .registration import register_pet_to_mni
from .harmonization import CombatHarmonizer
from ..utils.validation import validate_input_images
from .template_generation import TemplateBuilder

class TracerPipeline:
    """
    Pipeline for processing specific PET tracers without MRI.
    """
    
    def __init__(self, tracer_type, config=None):
        """
        Initialize tracer-specific pipeline.
        
        Parameters
        ----------
        tracer_type : str
            Type of tracer ('PIB', 'FBB', 'FBP', 'FMM', 'FBZ')
        config : Config, optional
            Configuration object
        """
        self.config = config or Config()
        if tracer_type not in self.config.PET_ONLY_PARAMS:
            raise ValueError(f"Unsupported tracer: {tracer_type}")
        
        self.tracer_type = tracer_type
        self.tracer_params = self.config.PET_ONLY_PARAMS[tracer_type]
        self.harmonizer = CombatHarmonizer()
    def process_single_pet(self, 
                          pet_path: str,
                          resolution: str = '2mm',
                          output_dir: Optional[Path] = None,
                          apply_decay_correction: bool = True) -> Dict[str, str]:
        """Process single PET image."""
        validate_input_images([pet_path])
        
        output_dir = Path(output_dir or Path(pet_path).parent)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get tracer-specific templates and masks from new location
        template_path = (self.config.DATA_DIR / 'templates' / 
                        self.config.TRACER_DATA[self.tracer_type]['template'][resolution])
        ref_mask = (self.config.DATA_DIR / 'masks' / 
                   self.config.TRACER_DATA[self.tracer_type]['reference_mask'][resolution])
        target_mask = self.config.TRACER_DATA[self.tracer_type]['target_mask'][resolution]
        template_path = str(self.config.TEMPLATE_DIR / self.tracer_params['template'])
        
        # Apply decay correction if needed
        if apply_decay_correction and self.tracer_params['decay_correction']:
            pet_path = self._apply_decay_correction(pet_path, output_dir)
        
        # Register to tracer-specific template
        reg_params = self.tracer_params['registration']
        mni_pet = register_pet_to_mni(
            pet_path, 
            template_path,
            output_path=str(output_dir / f'pet_mni_{self.tracer_type.lower()}.nii.gz'),
            **reg_params
        )
        
        return {
            'mni_space': mni_pet,
            'original': pet_path
        }
    
    def _apply_decay_correction(self, pet_path, output_dir):
        """Apply decay correction based on tracer half-life."""
        img = nib.load(pet_path)
        data = img.get_fdata()
        
        # Get acquisition time from header or use mid-frame time
        frame_start, frame_end = self.tracer_params['dynamic_frames']
        mid_time = (frame_start + frame_end) / 2
        
        # Calculate decay factor
        half_life = self.tracer_params['half_life']
        decay_factor = np.exp(np.log(2) * mid_time / half_life)
        
        # Apply correction
        corrected_data = data * decay_factor
        
        # Save corrected image
        output_path = str(output_dir / f'decay_corrected_{self.tracer_type.lower()}.nii.gz')
        corrected_img = nib.Nifti1Image(corrected_data, img.affine, img.header)
        nib.save(corrected_img, output_path)
        
        return output_path
    
    def process_batch(self, pet_paths, scanners=None, covariates=None, output_dir=None):
        """
        Process multiple PET images with tracer-specific parameters.
        
        Parameters
        ----------
        pet_paths : list
            List of paths to PET images
        scanners : list, optional
            Scanner identifiers for harmonization
        covariates : pd.DataFrame, optional
            Additional covariates for harmonization
        output_dir : str, optional
            Output directory
            
        Returns
        -------
        list
            List of dictionaries containing processed image paths
        """
        results = []
        normalized_images = []
        
        # Process each image
        for pet_path in pet_paths:
            result = self.process_single_pet(pet_path, output_dir)
            results.append(result)
            normalized_images.append(result['mni_space'])
        
        # Perform harmonization if scanner information is provided
        if scanners is not None:
            harmonized_images = self.harmonizer.harmonize(
                normalized_images,
                scanners,
                covariates
            )
            
            # Update results with harmonized images
            for result, harmonized_path in zip(results, harmonized_images):
                result['harmonized'] = harmonized_path
        
        return results
    
    def create_tracer_template(self, pet_paths, output_dir=None, iterations=3):
        """
        Create a tracer-specific template from multiple PET images.
        
        Parameters
        ----------
        pet_paths : list
            List of paths to PET images
        output_dir : str, optional
            Output directory for template
        iterations : int, optional
            Number of iterations for template building
            
        Returns
        -------
        str
            Path to the generated template
        """
        template_builder = TemplateBuilder(self.tracer_type, self.config)
        template_path = template_builder.create_template(
            pet_paths,
            output_dir=output_dir,
            iterations=iterations
        )
        
        # Update tracer parameters with new template
        self.tracer_params['template'] = template_path
        
        return template_path