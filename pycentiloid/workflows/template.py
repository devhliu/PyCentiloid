"""
Template creation workflows for different modalities.
"""

import ants
import numpy as np
from pathlib import Path
from typing import List, Optional
from ..core.template import TemplateParams
from ..utils.validation import validate_input_images

class TemplateBuilder:
    """Template building workflow for different modalities."""
    
    def __init__(self, modality: str, output_dir: Optional[Path] = None):
        """
        Initialize template builder.
        
        Parameters
        ----------
        modality : str
            Image modality ('PET', 'CT', 'T1')
        output_dir : Path, optional
            Output directory for template files
        """
        self.modality = modality
        self.output_dir = output_dir or Path.cwd()
        self.params = TemplateParams.get_params(modality)
    
    def create_template(self, 
                       image_paths: List[str], 
                       resolution: str = '2mm',
                       iterations: int = 3,
                       shape: Optional[tuple] = None,
                       remove_outliers: bool = True,
                       verbose: bool = True) -> str:
        """Create template with specified resolution."""
        validate_input_images(image_paths)
        
        # Get reference templates for specified resolution
        mni_template = self.config.TEMPLATES['MNI152'][resolution]
        aal_template = self.config.TEMPLATES['AAL'][resolution]
        
        # Load and resample images if needed
        images = []
        for path in image_paths:
            img = ants.image_read(str(path))
            if img.spacing != ants.image_read(str(mni_template)).spacing:
                img = match_resolution(img, mni_template)
            images.append(img)
        
        # Load images
        images = [ants.image_read(str(path)) for path in image_paths]
        
        # Initialize template
        template = self._initialize_template(images)
        
        # Iterative refinement
        for i in range(iterations):
            if verbose:
                print(f"Template iteration {i+1}/{iterations}")
            
            # Register all images to current template
            registered = []
            for img in images:
                reg = ants.registration(
                    fixed=template,
                    moving=img,
                    type_of_transform=self.params.transform_type,
                    reg_iterations=self.params.iterations,
                    grad_step=0.1,
                    flow_sigma=3,
                    verbose=verbose
                )
                registered.append(reg['warpedmovout'])
            
            # Update template
            template = self._average_images(registered)
        
        # Save final template
        output_path = self.output_dir / f"{self.modality.lower()}_template.nii.gz"
        template.to_filename(str(output_path))
        
        return str(output_path)
"""
Template creation workflow using ANTsPy.
"""

import ants
import numpy as np
from pathlib import Path
from typing import List, Optional
from ..core.template import TemplateParams

def create_template(image_paths: List[str], 
                   output_path: str,
                   iterations: int = 4,
                   gradient_step: float = 0.1,
                   modality: str = 'PET') -> str:
    """
    Create a template from multiple images using ANTsPy.
    """
    # Load all images
    images = [ants.image_read(img) for img in image_paths]
    
    # Initialize template as mean of all images
    template = ants.average_images([img for img in images])
    
    # Iterative template refinement
    for i in range(iterations):
        # Register all images to current template
        transforms = []
        for img in images:
            transform = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform='SyN',
                reg_iterations=(100, 70, 50),
                grad_step=gradient_step,
                flow_sigma=3,
                total_sigma=0
            )
            transforms.append(transform['warpedmovout'])
        
        # Update template
        template = ants.average_images([t for t in transforms])
    
    # Save final template
    template.to_filename(output_path)
    
    return output_path