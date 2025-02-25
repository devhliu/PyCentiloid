"""
Centiloid calculation workflow.
"""

import ants
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from ..core.tracer import TracerRegistry
from ..preprocessing.registration import register_pet_to_mni

class CentiloidWorkflow:
    """Workflow for Centiloid calculation."""
    
    def __init__(self, tracer_id: str):
        """Initialize workflow for specific tracer."""
        self.tracer = TracerRegistry.get_tracer(tracer_id)
    
    def process_single_scan(self, 
                          pet_path: str,
                          output_dir: Optional[Path] = None) -> Dict[str, float]:
        """
        Process single PET scan and calculate Centiloid value.
        """
        output_dir = output_dir or Path(pet_path).parent
        
        # Register to template
        registered_pet = register_pet_to_mni(
            pet_path,
            self.tracer.template_path,
            output_path=str(output_dir / f"registered_{Path(pet_path).name}")
        )
        
        # Load images
        pet_img = ants.image_read(registered_pet)
        ref_mask = ants.image_read(self.tracer.reference_mask)
        target_mask = ants.image_read(self.tracer.target_mask)
        
        # Calculate SUVR
        ref_value = np.mean(pet_img.numpy()[ref_mask.numpy() > 0])
        target_value = np.mean(pet_img.numpy()[target_mask.numpy() > 0])
        suvr = target_value / ref_value
        
        # Convert to Centiloid
        centiloid = self._suvr_to_centiloid(suvr)
        
        return {
            'suvr': suvr,
            'centiloid': centiloid,
            'registered_pet': registered_pet
        }
    
    def _suvr_to_centiloid(self, suvr: float) -> float:
        """Convert SUVR to Centiloid units."""
        # Apply tracer-specific scaling
        scaled_suvr = (suvr - 1) * self.tracer.scaling_factor
        # Convert to Centiloid scale
        return scaled_suvr * 100