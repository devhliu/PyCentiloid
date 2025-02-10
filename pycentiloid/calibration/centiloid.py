import numpy as np
from sklearn.linear_model import LinearRegression
from ..preprocessing.normalization import normalize_to_mni
import nibabel as nib
from pathlib import Path
import pandas as pd

class CentiloidCalibration:
    """
    Calibrate a new amyloid tracer to PIB-based Centiloid units using GAAIN method.
    """
    
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.r_squared = None
        
    def calibrate_tracer(self, new_tracer_images, pib_images, ctx_mask_path, cb_mask_path):
        """
        Perform calibration between new tracer and PIB images.
        
        Parameters
        ----------
        new_tracer_images : list
            Paths to spatially normalized new tracer images
        pib_images : list
            Paths to spatially normalized PIB images for same subjects
        ctx_mask_path : str
            Path to cortical target mask
        cb_mask_path : str
            Path to cerebellar reference mask
            
        Returns
        -------
        dict
            Calibration parameters and statistics
        """
        # Calculate SUVRs for both tracers
        new_suvrs = self._calculate_suvrs(new_tracer_images, ctx_mask_path, cb_mask_path)
        pib_suvrs = self._calculate_suvrs(pib_images, ctx_mask_path, cb_mask_path)
        
        # Perform linear regression
        model = LinearRegression()
        X = pib_suvrs.reshape(-1, 1)
        y = new_suvrs.reshape(-1, 1)
        model.fit(X, y)
        
        # Store calibration parameters
        self.slope = model.coef_[0][0]
        self.intercept = model.intercept_[0]
        self.r_squared = model.score(X, y)
        
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'r_squared': self.r_squared
        }
    
    def convert_to_centiloid(self, suvr):
        """
        Convert new tracer SUVR to Centiloid units.
        
        Parameters
        ----------
        suvr : float
            SUVR value from new tracer
            
        Returns
        -------
        float
            Centiloid value
        """
        if self.slope is None or self.intercept is None:
            raise ValueError("Calibration must be performed first")
        
        # Convert to PIB SUVR equivalent
        pib_suvr = (suvr - self.intercept) / self.slope
        
        # Convert to Centiloid using standard equation
        centiloid = 100 * (pib_suvr - 1.009) / 1.067
        
        return centiloid
    
    def _calculate_suvrs(self, image_paths, ctx_mask_path, cb_mask_path):
        """Calculate SUVRs for a list of images."""
        suvrs = []
        
        ctx_mask = nib.load(ctx_mask_path).get_fdata()
        cb_mask = nib.load(cb_mask_path).get_fdata()
        
        for img_path in image_paths:
            img_data = nib.load(img_path).get_fdata()
            ctx_mean = np.mean(img_data[ctx_mask > 0])
            cb_mean = np.mean(img_data[cb_mask > 0])
            suvr = ctx_mean / cb_mean
            suvrs.append(suvr)
            
        return np.array(suvrs)
    
    def save_calibration(self, output_path):
        """Save calibration parameters to CSV."""
        if self.slope is None or self.intercept is None:
            raise ValueError("Calibration must be performed first")
            
        data = {
            'Parameter': ['Slope', 'Intercept', 'R_squared'],
            'Value': [self.slope, self.intercept, self.r_squared]
        }
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)