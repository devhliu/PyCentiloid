"""
Harmonization module for pycentiloid package.
Implements ComBat harmonization for PET images.
"""

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from ..utils.image_utils import load_image, save_image

class CombatHarmonizer:
    """
    Class for performing ComBat harmonization on PET images.
    """
    
    def __init__(self, target_resolution=(6.0, 12.0)):
        """
        Initialize the harmonizer.
        
        Parameters
        ----------
        target_resolution : tuple
            Target effective resolution range (min, max) in mm
        """
        self.target_resolution = target_resolution
        
    def harmonize(self, image_paths, scanners, covariates=None):
        """
        Perform ComBat harmonization on a set of images.
        
        Parameters
        ----------
        image_paths : list
            List of paths to input images
        scanners : list
            List of scanner identifiers for each image
        covariates : pd.DataFrame, optional
            Covariates for harmonization (age, sex, etc.)
            
        Returns
        -------
        list
            Paths to harmonized images
        """
        # Load images and prepare data
        data = []
        for path in image_paths:
            img = load_image(path)
            data.append(img.get_fdata().reshape(-1))
        
        data = np.array(data).T
        
        # Prepare batch (scanner) information
        batch = pd.Series(scanners)
        
        # Run ComBat
        harmonized_data = neuroCombat(
            dat=data,
            batch=batch,
            categorical_cols=covariates.select_dtypes(include=['object']).columns.tolist() if covariates is not None else None,
            continuous_cols=covariates.select_dtypes(include=['float64', 'int64']).columns.tolist() if covariates is not None else None,
            eb=True
        )['data']
        
        # Save harmonized images
        harmonized_paths = []
        for i, path in enumerate(image_paths):
            img = load_image(path)
            harmonized_img = harmonized_data[:, i].reshape(img.shape)
            output_path = path.replace('.nii', '_harmonized.nii')
            save_image(harmonized_img, img.affine, output_path)
            harmonized_paths.append(output_path)
            
        return harmonized_paths