"""
Utilities for handling image resolutions.
"""

import ants
import numpy as np
from pathlib import Path
from typing import Tuple, Union

def get_image_resolution(image_path: Union[str, Path]) -> Tuple[float, float, float]:
    """Get image resolution in mm."""
    img = ants.image_read(str(image_path))
    return img.spacing

def resample_to_resolution(image: ants.ANTsImage, 
                         target_resolution: Tuple[float, float, float],
                         interpolation: str = 'linear') -> ants.ANTsImage:
    """Resample image to target resolution."""
    return ants.resample_image(
        image,
        target_resolution,
        use_voxels=False,
        interp_type=interpolation
    )

def match_resolution(source_path: Union[str, Path], 
                    target_path: Union[str, Path],
                    output_path: Union[str, Path] = None,
                    interpolation: str = 'linear') -> str:
    """Match source image resolution to target image."""
    source = ants.image_read(str(source_path))
    target = ants.image_read(str(target_path))
    
    resampled = ants.resample_image_to_target(
        source,
        target,
        interp_type=interpolation
    )
    
    if output_path:
        resampled.to_filename(str(output_path))
        return str(output_path)
    
    return resampled