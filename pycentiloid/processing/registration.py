"""
Spatial normalization module for PyCentiloid.

This module provides functions for spatial normalization of brain images
using both traditional methods (ANTsPy) and deep learning approaches (uniGradICON).
"""

import os
import ants
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Literal

# Import deep learning model if available
try:
    import torch
    from uniGradICON import SpatialTransformer
    HAS_UNIGRADICON = True
except ImportError:
    HAS_UNIGRADICON = False

class SpatialNormalizer:
    """
    Class for spatial normalization of brain images to standard space.
    
    Supports both traditional registration (ANTsPy) and deep learning
    based registration (uniGradICON).
    """
    
    def __init__(self, 
                 method: Literal['ants', 'deep'] = 'ants',
                 resolution: Literal['1mm', '2mm'] = '2mm',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu' if HAS_UNIGRADICON else None):
        """
        Initialize spatial normalizer.
        
        Parameters
        ----------
        method : str
            Registration method ('ants' or 'deep')
        resolution : str
            Target resolution ('1mm' or '2mm')
        device : str
            Device for deep learning model ('cuda' or 'cpu')
        """
        self.method = method
        self.resolution = resolution
        self.device = device
        
        # Load templates based on resolution
        self._load_templates()
        
        # Initialize deep learning model if needed
        if method == 'deep' and HAS_UNIGRADICON:
            self._initialize_dl_model()
        elif method == 'deep' and not HAS_UNIGRADICON:
            print("Warning: uniGradICON not available. Falling back to ANTs registration.")
            self.method = 'ants'
    
    def _load_templates(self):
        """Load appropriate templates based on resolution."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        template_dir = package_dir / 'data' / 'templates'
        
        # Template paths
        if self.resolution == '1mm':
            self.mni_template = template_dir / 'mni152_t1_1mm.nii.gz'
            self.pet_template = template_dir / 'pet_template_1mm.nii.gz'
            self.ct_template = template_dir / 'ct_template_1mm.nii.gz'
        else:  # 2mm
            self.mni_template = template_dir / 'mni152_t1_2mm.nii.gz'
            self.pet_template = template_dir / 'pet_template_2mm.nii.gz'
            self.ct_template = template_dir / 'ct_template_2mm.nii.gz'
    
    def _initialize_dl_model(self):
        """Initialize deep learning registration model."""
        if not HAS_UNIGRADICON:
            return
            
        # Load pre-trained model based on modality and resolution
        model_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'models'
        model_path = model_dir / f'uniGradICON_{self.resolution}.pt'
        
        if model_path.exists():
            self.dl_model = SpatialTransformer.load_from_checkpoint(str(model_path))
            self.dl_model.to(self.device)
            self.dl_model.eval()
        else:
            print(f"Warning: Model file {model_path} not found. Falling back to ANTs registration.")
            self.method = 'ants'
    
    def normalize(self, 
                 image_path: Union[str, Path],
                 modality: Literal['PET', 'CT', 'MRI'] = 'PET',
                 output_path: Optional[Union[str, Path]] = None,
                 transform_type: str = 'SyN',
                 rigid_only: bool = False) -> str:
        """
        Normalize image to standard space.
        
        Parameters
        ----------
        image_path : str or Path
            Path to input image
        modality : str
            Image modality ('PET', 'CT', 'MRI')
        output_path : str or Path, optional
            Path to save normalized image
        transform_type : str
            ANTs transform type (ignored if method is 'deep')
        rigid_only : bool
            Whether to perform rigid registration only
            
        Returns
        -------
        str
            Path to normalized image
        """
        # Determine template based on modality
        if modality == 'PET':
            template = self.pet_template
        elif modality == 'CT':
            template = self.ct_template
        else:  # MRI
            template = self.mni_template
        
        # Create output path if not provided
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_normalized{input_path.suffix}"
        
        # Perform registration
        if self.method == 'deep' and HAS_UNIGRADICON:
            return self._deep_registration(image_path, template, output_path)
        else:
            return self._ants_registration(image_path, template, output_path, 
                                         transform_type, rigid_only)
    
    def _ants_registration(self, 
                          image_path: Union[str, Path],
                          template_path: Union[str, Path],
                          output_path: Union[str, Path],
                          transform_type: str = 'SyN',
                          rigid_only: bool = False) -> str:
        """Perform registration using ANTsPy."""
        # Load images
        moving = ants.image_read(str(image_path))
        fixed = ants.image_read(str(template_path))
        
        # Determine transform type
        if rigid_only:
            transform_type = 'Rigid'
        
        # Perform registration
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=transform_type,
            reg_iterations=[100, 70, 50] if transform_type == 'SyN' else [100, 50, 25],
            verbose=True
        )
        
        # Save result
        warped_image = registration['warpedmovout']
        warped_image.to_filename(str(output_path))
        
        return str(output_path)
    
    def _deep_registration(self,
                          image_path: Union[str, Path],
                          template_path: Union[str, Path],
                          output_path: Union[str, Path]) -> str:
        """Perform registration using deep learning model."""
        if not HAS_UNIGRADICON:
            raise ImportError("uniGradICON is not available")
        
        # Load images
        moving_img = ants.image_read(str(image_path))
        fixed_img = ants.image_read(str(template_path))
        
        # Convert to numpy arrays and normalize
        moving_np = moving_img.numpy()
        fixed_np = fixed_img.numpy()
        
        # Normalize intensity
        moving_np = (moving_np - moving_np.min()) / (moving_np.max() - moving_np.min())
        fixed_np = (fixed_np - fixed_np.min()) / (fixed_np.max() - fixed_np.min())
        
        # Convert to torch tensors
        moving_tensor = torch.from_numpy(moving_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
        fixed_tensor = torch.from_numpy(fixed_np).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # Perform registration
        with torch.no_grad():
            warped_tensor, _ = self.dl_model(moving_tensor, fixed_tensor)
        
        # Convert back to numpy and ANTs image
        warped_np = warped_tensor.squeeze().cpu().numpy()
        warped_img = ants.from_numpy(warped_np, origin=fixed_img.origin, 
                                    spacing=fixed_img.spacing, direction=fixed_img.direction)
        
        # Save result
        warped_img.to_filename(str(output_path))
        
        return str(output_path)

# Convenience functions
"""
Image registration module for PyCentiloid.

This module provides functions for registering PET and MRI images to standard
templates, including rigid, affine, and non-linear registration.
"""

import os
import ants
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Literal

from ..core.template import TemplateRegistry

def register_to_mni(
    image_path: Union[str, Path],
    modality: str = 'MRI',
    output_path: Optional[Union[str, Path]] = None,
    resolution: str = '2mm',
    registration_type: str = 'SyN',
    verbose: bool = False
) -> str:
    """
    Register an image to MNI space.
    
    Parameters
    ----------
    image_path : str or Path
        Path to input image
    modality : str
        Image modality ('MRI', 'PET', 'CT')
    output_path : str or Path, optional
        Path to save registered image
    resolution : str
        Resolution of the template ('1mm', '2mm')
    registration_type : str
        Registration type ('Rigid', 'Affine', 'SyN')
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    str
        Path to registered image
    """
    # Load input image
    if verbose:
        print(f"Loading input image: {image_path}")
    
    input_img = ants.image_read(str(image_path))
    
    # Get template
    template = TemplateRegistry.get_default_template(modality, resolution)
    
    if verbose:
        print(f"Using template: {template.name}")
    
    template_img = template.load()
    
    # Set output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_mni{input_path.suffix}"
    
    # Set registration parameters
    if registration_type == 'Rigid':
        transform_type = 'Rigid'
    elif registration_type == 'Affine':
        transform_type = 'Affine'
    else:
        transform_type = 'SyN'
    
    # Perform registration
    if verbose:
        print(f"Performing {transform_type} registration...")
    
    if transform_type == 'SyN':
        registration = ants.registration(
            fixed=template_img,
            moving=input_img,
            type_of_transform='SyN',
            verbose=verbose
        )
    elif transform_type == 'Affine':
        registration = ants.registration(
            fixed=template_img,
            moving=input_img,
            type_of_transform='Affine',
            verbose=verbose
        )
    else:  # Rigid
        registration = ants.registration(
            fixed=template_img,
            moving=input_img,
            type_of_transform='Rigid',
            verbose=verbose
        )
    
    # Get registered image
    registered_img = registration['warpedmovout']
    
    # Save registered image
    if verbose:
        print(f"Saving registered image to: {output_path}")
    
    ants.image_write(registered_img, str(output_path))
    
    return str(output_path)

def register_pet_to_mri(
    pet_path: Union[str, Path],
    mri_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    registration_type: str = 'Rigid',
    verbose: bool = False
) -> str:
    """
    Register PET image to MRI.
    
    Parameters
    ----------
    pet_path : str or Path
        Path to PET image
    mri_path : str or Path
        Path to MRI image
    output_path : str or Path, optional
        Path to save registered PET image
    registration_type : str
        Registration type ('Rigid', 'Affine', 'SyN')
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    str
        Path to registered PET image
    """
    # Load input images
    if verbose:
        print(f"Loading PET image: {pet_path}")
        print(f"Loading MRI image: {mri_path}")
    
    pet_img = ants.image_read(str(pet_path))
    mri_img = ants.image_read(str(mri_path))
    
    # Set output path if not provided
    if output_path is None:
        pet_path_obj = Path(pet_path)
        output_path = pet_path_obj.parent / f"{pet_path_obj.stem}_mri{pet_path_obj.suffix}"
    
    # Set registration parameters
    if registration_type == 'Rigid':
        transform_type = 'Rigid'
    elif registration_type == 'Affine':
        transform_type = 'Affine'
    else:
        transform_type = 'SyN'
    
    # Perform registration
    if verbose:
        print(f"Performing {transform_type} registration...")
    
    if transform_type == 'SyN':
        registration = ants.registration(
            fixed=mri_img,
            moving=pet_img,
            type_of_transform='SyN',
            verbose=verbose
        )
    elif transform_type == 'Affine':
        registration = ants.registration(
            fixed=mri_img,
            moving=pet_img,
            type_of_transform='Affine',
            verbose=verbose
        )
    else:  # Rigid
        registration = ants.registration(
            fixed=mri_img,
            moving=pet_img,
            type_of_transform='Rigid',
            verbose=verbose
        )
    
    # Get registered image
    registered_img = registration['warpedmovout']
    
    # Save registered image
    if verbose:
        print(f"Saving registered PET image to: {output_path}")
    
    ants.image_write(registered_img, str(output_path))
    
    return str(output_path)

def register_to_mni(image_path: Union[str, Path],
                   modality: Literal['PET', 'CT', 'MRI'] = 'PET',
                   output_path: Optional[Union[str, Path]] = None,
                   method: Literal['ants', 'deep'] = 'ants',
                   resolution: Literal['1mm', '2mm'] = '2mm',
                   rigid_only: bool = False) -> str:
    """
    Register image to MNI space.
    
    Parameters
    ----------
    image_path : str or Path
        Path to input image
    modality : str
        Image modality ('PET', 'CT', 'MRI')
    output_path : str or Path, optional
        Path to save registered image
    method : str
        Registration method ('ants' or 'deep')
    resolution : str
        Target resolution ('1mm' or '2mm')
    rigid_only : bool
        Whether to perform rigid registration only
        
    Returns
    -------
    str
        Path to registered image
    """
    normalizer = SpatialNormalizer(method=method, resolution=resolution)
    return normalizer.normalize(image_path, modality, output_path, rigid_only=rigid_only)