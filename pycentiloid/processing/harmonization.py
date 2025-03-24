"""
Image harmonization module for PyCentiloid.

This module provides functions for harmonizing PET images across different scanners
and protocols, including phantom-based harmonization, paired MRI-based harmonization,
ComBat harmonization, and partial volume correction as preprocessing.
"""

import os
import ants
import numpy as np
import pickle
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple, Literal
from scipy import ndimage, signal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('pycentiloid.harmonization')

class ImageHarmonization:
    """
    Class for harmonizing PET images across different scanners and protocols.
    
    This class implements multiple harmonization methods:
    - Phantom: Hoffman phantom-based harmonization
    - MRI: Paired MRI-based harmonization
    - ComBat: Statistical harmonization for multi-site data
    
    Preprocessing options:
    - PVC: Partial volume correction for Amyloid PET enhancement
    
    All harmonization methods can target a specific Effective Image Resolution (EIR).
    """
    
    def __init__(self, 
                method: Literal['combat', 'phantom', 'mri'] = 'combat',
                model_path: Optional[Union[str, Path]] = None,
                apply_pvc: bool = False,
                target_eir: Optional[float] = None,
                verbose: bool = False):
        """
        Initialize image harmonization.
        
        Parameters
        ----------
        method : str
            Harmonization method ('combat', 'phantom', 'mri')
        model_path : str or Path, optional
            Path to pre-trained harmonization model
        apply_pvc : bool, optional
            Whether to apply partial volume correction as preprocessing
        target_eir : float, optional
            Target effective image resolution in mm (default is determined from model or set to 8.0mm)
        verbose : bool, optional
            Whether to print detailed information during processing
        """
        self.method = method
        self.model_path = model_path
        self.model = None
        self.apply_pvc = apply_pvc
        self.target_eir = target_eir
        self.verbose = verbose
        
        # Set up logging level based on verbosity
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        self._load_models()
    
    def calculate_eir(self, img: ants.ANTsImage) -> float:
        """
        Calculate the Effective Image Resolution (EIR) of a PET image.
        
        This is an optimized implementation that uses gradient analysis to 
        estimate the effective resolution of the image.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input PET image
            
        Returns
        -------
        float
            Estimated EIR in mm
        """
        # Extract image data
        img_data = img.numpy()
        
        # Apply mask to focus on brain tissue (improved threshold)
        # Use a more robust threshold based on percentile
        threshold = np.percentile(img_data[img_data > 0], 15)
        mask = img_data > threshold
        
        # Calculate gradient magnitude using Sobel filter for better edge detection
        grad_x = ndimage.sobel(img_data, axis=0)
        grad_y = ndimage.sobel(img_data, axis=1)
        grad_z = ndimage.sobel(img_data, axis=2) if img_data.ndim > 2 else np.zeros_like(grad_x)
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Apply mask to gradient magnitude
        grad_mag = grad_mag * mask
        
        # Calculate autocorrelation of gradient magnitude
        # Use FFT for faster computation of autocorrelation
        from scipy import fftpack
        
        # Pad the array to avoid circular correlation effects
        padded_grad = np.pad(grad_mag, [(s//2, s//2) for s in grad_mag.shape], mode='constant')
        
        # Compute autocorrelation via FFT
        fft_grad = fftpack.fftn(padded_grad)
        power_spectrum = np.abs(fft_grad)**2
        autocorr = np.real(fftpack.ifftn(power_spectrum))
        
        # Extract the central region corresponding to the original size
        slices = tuple(slice(s//2, s//2 + s) for s in grad_mag.shape)
        autocorr = autocorr[slices]
        
        # Find the FWHM of the autocorrelation peak
        center = np.array(autocorr.shape) // 2
        max_val = autocorr[tuple(center)]
        half_max = max_val / 2.0
        
        # Calculate FWHM in each dimension using interpolation for sub-voxel precision
        fwhm_mm = []
        for axis in range(autocorr.ndim):
            line = [center[i] if i != axis else slice(None) for i in range(autocorr.ndim)]
            profile = autocorr[tuple(line)]
            
            # Find points where profile crosses half maximum
            above_half_max = profile > half_max
            transitions = np.where(np.diff(above_half_max.astype(int)))[0]
            
            if len(transitions) >= 2:
                # Use linear interpolation for sub-voxel precision
                x1, x2 = transitions[0], transitions[1]
                y1 = profile[x1] - half_max
                y2 = profile[x1+1] - half_max
                dx1 = -y1 / (y2 - y1) if y2 != y1 else 0
                
                y1 = profile[x2] - half_max
                y2 = profile[x2+1] - half_max
                dx2 = -y1 / (y2 - y1) if y2 != y1 else 0
                
                # Calculate FWHM in voxels with sub-voxel precision
                fwhm_voxels = (x2 + dx2) - (x1 + dx1)
                
                # Convert to mm using image spacing
                fwhm_mm.append(fwhm_voxels * img.spacing[axis])
            else:
                # Fallback if we can't find clear transitions
                logger.warning(f"Could not determine FWHM for axis {axis}, using fallback")
                fwhm_mm.append(4.0 * img.spacing[axis])  # Reasonable fallback
        
        # Calculate average EIR across dimensions
        eir = np.mean(fwhm_mm)
        
        if self.verbose:
            logger.debug(f"Calculated EIR: {eir:.2f}mm (FWHM per dimension: {[f'{x:.2f}mm' for x in fwhm_mm]})")
        
        return eir

    def _mri_guided_transform(self, pet_data: np.ndarray, 
                             mri_features: Dict[str, float],
                             model_pairs: List[Dict]) -> np.ndarray:
        """
        Apply MRI-guided transformation to PET data.
        
        Parameters
        ----------
        pet_data : np.ndarray
            Input PET image data
        mri_features : dict
            MRI features extracted from paired MRI
        model_pairs : list of dict
            Model data from training
            
        Returns
        -------
        np.ndarray
            Transformed PET data
        """
        # Find the most similar MRI features in the model
        best_match_idx = 0
        best_match_score = float('inf')
        
        for i, pair in enumerate(model_pairs):
            pair_features = pair['mri_features']
            
            # Calculate feature distance (weighted Euclidean distance)
            distance = 0
            # Give more weight to tissue ratios
            distance += 3.0 * (pair_features['gm_ratio'] - mri_features['gm_ratio'])**2
            distance += 2.0 * (pair_features['wm_ratio'] - mri_features['wm_ratio'])**2
            distance += 1.0 * (pair_features['csf_ratio'] - mri_features['csf_ratio'])**2
            
            # Add other features with lower weights
            distance += 0.5 * ((pair_features['brain_volume'] - mri_features['brain_volume']) / 
                              max(pair_features['brain_volume'], 1))**2
            
            if distance < best_match_score:
                best_match_score = distance
                best_match_idx = i
        
        # Get the best matching pair
        best_pair = model_pairs[best_match_idx]
        
        if self.verbose:
            logger.debug(f"Best matching MRI features found with score: {best_match_score:.4f}")
        
        # Apply transformation based on the best matching pair
        # Z-score normalization and rescaling
        mask = pet_data > 0
        if np.sum(mask) > 0:
            pet_mean = np.mean(pet_data[mask])
            pet_std = np.std(pet_data[mask])
            
            if pet_std > 0:
                # Normalize to z-scores
                normalized = np.zeros_like(pet_data)
                normalized[mask] = (pet_data[mask] - pet_mean) / pet_std
                
                # Rescale to target statistics
                target_mean = best_pair['pet_mean']
                target_std = best_pair['pet_std']
                
                normalized[mask] = normalized[mask] * target_std + target_mean
                return normalized
        
        return pet_data
    
    def _apply_simple_harmonization(self, img: ants.ANTsImage) -> ants.ANTsImage:
        """
        Apply simple harmonization when no model is available.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
            
        Returns
        -------
        ants.ANTsImage
            Harmonized image
        """
        logger.info("Applying simple harmonization (intensity normalization)")
        
        # Extract image data
        img_data = img.numpy()
        
        # Apply mask to focus on brain tissue
        mask = img_data > 0
        
        if np.sum(mask) > 0:
            # Z-score normalization
            mean = np.mean(img_data[mask])
            std = np.std(img_data[mask])
            
            if std > 0:
                normalized = np.zeros_like(img_data)
                normalized[mask] = (img_data[mask] - mean) / std
                
                # Create new ANTs image
                harmonized_img = ants.from_numpy(normalized, origin=img.origin,
                                               spacing=img.spacing,
                                               direction=img.direction)
                return harmonized_img
        
        return img
    
    def _apply_simple_pvc(self, img: ants.ANTsImage, metadata: Optional[Dict] = None) -> ants.ANTsImage:
        """
        Apply simple partial volume correction when no segmentation is available.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
        metadata : dict, optional
            Image metadata
            
        Returns
        -------
        ants.ANTsImage
            PVC-corrected image
        """
        logger.info("Applying simple PVC (edge-preserving smoothing)")
        
        # Get PVC parameters
        fwhm = metadata.get('fwhm', [4.0, 4.0, 4.0]) if metadata else [4.0, 4.0, 4.0]
        
        # Convert FWHM to sigma for Gaussian kernel
        sigma = [f / (2 * np.sqrt(2 * np.log(2))) for f in fwhm]
        
        # Apply edge-preserving smoothing
        # Use ANTs bilateral filter for edge preservation
        try:
            # Try to use ANTs bilateral filter
            corrected_img = ants.denoise_image(img, smoothing=min(sigma))
        except Exception:
            # Fallback to simple Gaussian smoothing with small sigma
            img_data = img.numpy()
            corrected_data = ndimage.gaussian_filter(img_data, [s/2 for s in sigma])
            corrected_img = ants.from_numpy(corrected_data, origin=img.origin,
                                          spacing=img.spacing,
                                          direction=img.direction)
        
        return corrected_img
    
    def _load_models(self):
        """Load harmonization models based on the selected method."""
        if self.model_path is None:
            logger.info(f"No model path provided for {self.method} harmonization")
            return
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return
        
        try:
            if self.method == 'phantom':
                self._load_phantom_model(model_path)
            elif self.method == 'mri':
                self._load_mri_model(model_path)
            elif self.method == 'combat':
                self._load_combat_model(model_path)
            else:
                logger.warning(f"No model loader available for method: {self.method}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def _load_phantom_model(self, model_path: Path):
        """Load phantom-based harmonization model."""
        logger.info(f"Loading phantom model from {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                serialized_model = pickle.load(f)
            
            # Reconstruct the model with callable functions
            self.model = {
                'target_resolution': serialized_model.get('target_resolution', 8.0),
                'phantom_stats': serialized_model.get('phantom_stats', [])
            }
            
            # Add the intensity transform function
            self.model['intensity_transform'] = lambda img_data: self._phantom_intensity_transform(
                img_data, self.model['phantom_stats'])
            
            logger.info(f"Phantom model loaded with {len(self.model['phantom_stats'])} reference images")
            
        except Exception as e:
            logger.error(f"Failed to load phantom model: {e}")
            self.model = None
    
    def _load_mri_model(self, model_path: Path):
        """Load MRI-based harmonization model."""
        logger.info(f"Loading MRI model from {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                serialized_model = pickle.load(f)
            
            # Reconstruct the model with callable functions
            self.model = {
                'target_resolution': serialized_model.get('target_resolution', 8.0),
                'pairs': serialized_model.get('pairs', [])
            }
            
            # Add the transform function
            self.model['transform'] = lambda pet_data, mri_features: self._mri_guided_transform(
                pet_data, mri_features, self.model['pairs'])
            
            logger.info(f"MRI model loaded with {len(self.model['pairs'])} reference pairs")
            
        except Exception as e:
            logger.error(f"Failed to load MRI model: {e}")
            self.model = None
    
    def _load_combat_model(self, model_path: Path):
        """Load ComBat harmonization model."""
        logger.info(f"Loading ComBat model from {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info("ComBat model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ComBat model: {e}")
            self.model = None
    
    def adjust_resolution(self, img: ants.ANTsImage, target_eir: float) -> ants.ANTsImage:
        """
        Adjust the resolution of an image to match the target EIR.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
        target_eir : float
            Target effective image resolution in mm
            
        Returns
        -------
        ants.ANTsImage
            Resolution-adjusted image
        """
        # Calculate current EIR
        current_eir = self.calculate_eir(img)
        
        # If current resolution is already close to target, return original
        if abs(current_eir - target_eir) < 0.5:
            logger.info(f"Current EIR ({current_eir:.2f}mm) already close to target ({target_eir:.2f}mm)")
            return img
        
        # Determine if we need to increase or decrease resolution
        if current_eir < target_eir:
            # Need to decrease resolution (smooth)
            logger.info(f"Decreasing resolution from {current_eir:.2f}mm to {target_eir:.2f}mm")
            
            # Calculate required sigma for Gaussian smoothing
            # Using quadrature rule: target² = current² + sigma²
            sigma_mm = np.sqrt(target_eir**2 - current_eir**2)
            
            # Convert sigma from mm to voxels
            sigma_voxels = [sigma_mm / s for s in img.spacing]
            
            # Apply Gaussian smoothing
            img_data = img.numpy()
            smoothed_data = ndimage.gaussian_filter(img_data, sigma_voxels)
            
            # Create new ANTs image
            adjusted_img = ants.from_numpy(smoothed_data, origin=img.origin,
                                         spacing=img.spacing,
                                         direction=img.direction)
            
        else:
            # Need to increase resolution (sharpen)
            logger.info(f"Increasing resolution from {current_eir:.2f}mm to {target_eir:.2f}mm")
            
            # Use Richardson-Lucy deconvolution for sharpening
            img_data = img.numpy()
            
            # Calculate required sigma for PSF
            # Using quadrature rule: current² = target² + sigma²
            sigma_mm = np.sqrt(current_eir**2 - target_eir**2)
            
            # Convert sigma from mm to voxels
            sigma_voxels = [sigma_mm / s for s in img.spacing]
            
            # Create PSF
            psf_size = [int(4 * s) | 1 for s in sigma_voxels]  # Ensure odd size
            
            # Create coordinate grids for PSF
            coords = [np.arange(-(s//2), -(s//2) + s) for s in psf_size]
            grid = np.meshgrid(*coords, indexing='ij')
            
            # Create Gaussian PSF
            psf = np.exp(-0.5 * sum([x**2 / s**2 for x, s in zip(grid, sigma_voxels)]))
            psf /= psf.sum()  # Normalize
            
            # Apply Richardson-Lucy deconvolution with regularization
            sharpened_data = img_data.copy()
            for i in range(5):  # Fewer iterations to avoid noise amplification
                # Forward model: blur current estimate
                blurred = ndimage.convolve(sharpened_data, psf, mode='constant')
                
                # Avoid division by zero
                blurred = np.maximum(blurred, 1e-10)
                
                # Update step
                relative_blur = img_data / blurred
                correction = ndimage.correlate(relative_blur, psf, mode='constant')
                
                # Apply correction
                sharpened_data *= correction
                
                # Apply regularization to suppress noise
                if i % 2 == 0:
                    sharpened_data = ndimage.gaussian_filter(sharpened_data, 0.5)
            
            # Create new ANTs image
            adjusted_img = ants.from_numpy(sharpened_data, origin=img.origin,
                                         spacing=img.spacing,
                                         direction=img.direction)
        
        # Verify the new EIR
        new_eir = self.calculate_eir(adjusted_img)
        logger.info(f"Resolution adjusted: {current_eir:.2f}mm -> {new_eir:.2f}mm (target: {target_eir:.2f}mm)")
        
        return adjusted_img
    
    def harmonize(self, img: Union[str, Path, ants.ANTsImage], 
                 metadata: Optional[Dict] = None) -> ants.ANTsImage:
        """
        Harmonize a PET image using the selected method.
        
        Parameters
        ----------
        img : str, Path, or ants.ANTsImage
            Input image or path to image
        metadata : dict, optional
            Image metadata for harmonization
            
        Returns
        -------
        ants.ANTsImage
            Harmonized image
        """
        # Load image if path is provided
        if isinstance(img, (str, Path)):
            img = ants.image_read(str(img))
        
        logger.info(f"Harmonizing image using {self.method} method")
        
        # Step 1: Apply PVC preprocessing if requested
        if self.apply_pvc:
            logger.info("Applying PVC preprocessing")
            img = self._apply_pvc(img, metadata)
        
        # Step 2: Determine target EIR
        target_eir = self._determine_target_eir(metadata)
        logger.info(f"Target EIR: {target_eir:.2f}mm")
        
        # Step 3: Apply harmonization based on method
        if self.method == 'phantom':
            return self._apply_phantom_harmonization(img, target_eir, metadata)
        elif self.method == 'mri':
            return self._apply_mri_harmonization(img, target_eir, metadata)
        elif self.method == 'combat':
            return self._apply_combat_harmonization(img, target_eir, metadata)
        else:
            logger.warning(f"Unknown harmonization method: {self.method}")
            return img
    
    def _determine_target_eir(self, metadata: Optional[Dict] = None) -> float:
        """
        Determine the target EIR from available sources.
        
        Parameters
        ----------
        metadata : dict, optional
            Image metadata that may contain target_resolution
            
        Returns
        -------
        float
            Target EIR in mm
        """
        # Priority order: 
        # 1. Explicitly set in constructor
        # 2. From metadata
        # 3. From model
        # 4. Default value (8.0mm)
        
        if self.target_eir is not None:
            return self.target_eir
        elif metadata is not None and 'target_resolution' in metadata:
            return metadata['target_resolution']
        elif self.model is not None and 'target_resolution' in self.model:
            return self.model['target_resolution']
        else:
            return 8.0  # Default target EIR for PET
    
    def _apply_pvc(self, img: ants.ANTsImage, metadata: Optional[Dict] = None) -> ants.ANTsImage:
        """
        Apply partial volume correction as preprocessing.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
        metadata : dict, optional
            Image metadata including segmentation information
            
        Returns
        -------
        ants.ANTsImage
            PVC-corrected image
        """
        # Check if we have segmentation information
        if metadata is not None and 'segmentation' in metadata:
            # Apply advanced PVC using segmentation
            logger.info("Applying advanced PVC with segmentation")
            # Implementation would depend on the specific PVC algorithm
            # This is a placeholder for a more sophisticated implementation
            return self._apply_simple_pvc(img, metadata)
        else:
            # Apply simple PVC without segmentation
            logger.info("No segmentation available. Applying simple PVC.")
            return self._apply_simple_pvc(img, metadata)
    
    def _apply_phantom_harmonization(self, img: ants.ANTsImage, 
                                    target_eir: float,
                                    metadata: Optional[Dict] = None) -> ants.ANTsImage:
        """
        Apply Hoffman phantom-based harmonization.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
        target_eir : float
            Target effective image resolution in mm
        metadata : dict, optional
            Image metadata
            
        Returns
        -------
        ants.ANTsImage
            Harmonized image
        """
        if self.model is None:
            logger.warning("No phantom model available. Falling back to simple harmonization.")
            return self._apply_simple_harmonization(img, target_eir)
        
        # Step 1: Adjust resolution to match target EIR
        current_eir = self.calculate_eir(img)
        logger.info(f"Current image EIR: {current_eir:.2f}mm")
        
        if abs(current_eir - target_eir) > 0.5:
            logger.info(f"Adjusting resolution from {current_eir:.2f}mm to {target_eir:.2f}mm")
            img = self.adjust_resolution(img, target_eir)
        
        # Step 2: Apply phantom-based intensity transformation
        if 'intensity_transform' in self.model and callable(self.model['intensity_transform']):
            logger.info("Applying phantom-based intensity transformation")
            img_data = img.numpy()
            harmonized_data = self.model['intensity_transform'](img_data)
            
            # Create new ANTs image
            harmonized_img = ants.from_numpy(harmonized_data, origin=img.origin,
                                           spacing=img.spacing,
                                           direction=img.direction)
            return harmonized_img
        
        return img
    
    def _apply_mri_harmonization(self, img: ants.ANTsImage, 
                                target_eir: float,
                                metadata: Optional[Dict] = None) -> ants.ANTsImage:
        """
        Apply MRI-guided harmonization.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
        target_eir : float
            Target effective image resolution in mm
        metadata : dict, optional
            Image metadata including MRI features
            
        Returns
        -------
        ants.ANTsImage
            Harmonized image
        """
        if self.model is None:
            logger.warning("No MRI model available. Falling back to simple harmonization.")
            return self._apply_simple_harmonization(img, target_eir)
        
        # Check if we have MRI features
        if metadata is None or 'mri_features' not in metadata:
            logger.warning("No MRI features provided for MRI-guided harmonization.")
            return self._apply_simple_harmonization(img, target_eir)
        
        # Step 1: Adjust resolution to match target EIR
        current_eir = self.calculate_eir(img)
        logger.info(f"Current image EIR: {current_eir:.2f}mm")
        
        if abs(current_eir - target_eir) > 0.5:
            logger.info(f"Adjusting resolution from {current_eir:.2f}mm to {target_eir:.2f}mm")
            img = self.adjust_resolution(img, target_eir)
        
        # Step 2: Apply MRI-guided transformation
        if 'transform' in self.model and callable(self.model['transform']):
            logger.info("Applying MRI-guided transformation")
            img_data = img.numpy()
            mri_features = metadata['mri_features']
            
            try:
                harmonized_data = self.model['transform'](img_data, mri_features)
                
                # Create new ANTs image
                harmonized_img = ants.from_numpy(harmonized_data, origin=img.origin,
                                               spacing=img.spacing,
                                               direction=img.direction)
                return harmonized_img
            except Exception as e:
                logger.error(f"Error applying MRI-guided transformation: {e}")
                return self._apply_simple_harmonization(img, target_eir)
        
        return img
    
    def _apply_combat_harmonization(self, img: ants.ANTsImage, 
                                   target_eir: float,
                                   metadata: Optional[Dict] = None) -> ants.ANTsImage:
        """
        Apply ComBat harmonization for multi-site data.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
        target_eir : float
            Target effective image resolution in mm
        metadata : dict, optional
            Image metadata including batch information
            
        Returns
        -------
        ants.ANTsImage
            Harmonized image
        """
        if self.model is None:
            logger.warning("No ComBat model available. Falling back to simple harmonization.")
            return self._apply_simple_harmonization(img, target_eir)
        
        # Step 1: Adjust resolution to match target EIR
        current_eir = self.calculate_eir(img)
        logger.info(f"Current image EIR: {current_eir:.2f}mm")
        
        if abs(current_eir - target_eir) > 0.5:
            logger.info(f"Adjusting resolution from {current_eir:.2f}mm to {target_eir:.2f}mm")
            img = self.adjust_resolution(img, target_eir)
        
        # Step 2: Apply ComBat transformation
        # Check if we have batch information
        if metadata is None or 'batch' not in metadata:
            logger.warning("No batch information provided for ComBat. Using default batch.")
            batch = 'unknown'
        else:
            batch = metadata['batch']
        
        # Extract image data
        img_data = img.numpy()
        
        # Apply mask to focus on brain tissue
        mask = img_data > 0
        
        if np.sum(mask) == 0:
            logger.warning("Empty image provided for ComBat harmonization.")
            return img
        
        # Flatten masked data for ComBat
        data_flat = img_data[mask]
        
        # Apply ComBat transformation
        if 'transform' in self.model and callable(self.model['transform']):
            try:
                # Apply batch-specific transformation
                harmonized_flat = self.model['transform'](data_flat, batch)
                
                # Reconstruct image
                harmonized_data = np.zeros_like(img_data)
                harmonized_data[mask] = harmonized_flat
                
                # Create new ANTs image
                harmonized_img = ants.from_numpy(harmonized_data, origin=img.origin,
                                               spacing=img.spacing,
                                               direction=img.direction)
                return harmonized_img
            except Exception as e:
                logger.error(f"Error applying ComBat harmonization: {e}")
                return self._apply_simple_harmonization(img, target_eir)
        else:
            logger.warning("Invalid ComBat model. Falling back to simple harmonization.")
            return self._apply_simple_harmonization(img, target_eir)
    
    def _apply_simple_harmonization(self, img: ants.ANTsImage, target_eir: float) -> ants.ANTsImage:
        """
        Apply simple harmonization when no model is available.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
        target_eir : float
            Target effective image resolution in mm
            
        Returns
        -------
        ants.ANTsImage
            Harmonized image
        """
        logger.info("Applying simple harmonization (intensity normalization)")
        
        # Step 1: Adjust resolution to match target EIR
        current_eir = self.calculate_eir(img)
        
        if abs(current_eir - target_eir) > 0.5:
            logger.info(f"Adjusting resolution from {current_eir:.2f}mm to {target_eir:.2f}mm")
            img = self.adjust_resolution(img, target_eir)
        
        # Step 2: Apply simple intensity normalization
        img_data = img.numpy()
        
        # Apply mask to focus on brain tissue
        mask = img_data > 0
        
        if np.sum(mask) > 0:
            # Z-score normalization
            mean = np.mean(img_data[mask])
            std = np.std(img_data[mask])
            
            if std > 0:
                normalized = np.zeros_like(img_data)
                normalized[mask] = (img_data[mask] - mean) / std
                
                # Create new ANTs image
                harmonized_img = ants.from_numpy(normalized, origin=img.origin,
                                               spacing=img.spacing,
                                               direction=img.direction)
                return harmonized_img
        
        return img
    
    def _phantom_intensity_transform(self, img_data: np.ndarray, 
                                    phantom_stats: List[Dict]) -> np.ndarray:
        """
        Apply phantom-based intensity transformation.
        
        Parameters
        ----------
        img_data : np.ndarray
            Input image data
        phantom_stats : list of dict
            Phantom statistics from model
            
        Returns
        -------
        np.ndarray
            Transformed image data
        """
        # Apply mask to focus on brain tissue
        mask = img_data > 0
        
        if np.sum(mask) == 0:
            logger.warning("Empty image provided for phantom harmonization.")
            return img_data
        
        # Calculate image statistics
        img_mean = np.mean(img_data[mask])
        img_std = np.std(img_data[mask])
        
        if img_std == 0:
            logger.warning("Zero standard deviation in image.")
            return img_data
        
        # Find reference phantom statistics (use average of all phantoms)
        ref_mean = np.mean([stats['mean'] for stats in phantom_stats])
        ref_std = np.mean([stats['std'] for stats in phantom_stats])
        
        # Apply z-score normalization and rescaling
        normalized = np.zeros_like(img_data)
        normalized[mask] = (img_data[mask] - img_mean) / img_std
        
        # Transform to reference statistics
        transformed = normalized * ref_std + ref_mean
        
        return transformed
    
    def train_phantom_model(self,
                           phantom_images: List[Union[str, Path, ants.ANTsImage]],
                           target_eir: float = 8.0,
                           output_model_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Train a phantom-based harmonization model.
        
        Parameters
        ----------
        phantom_images : list
            List of phantom image paths or ANTsImage objects
        target_eir : float, optional
            Target effective image resolution in mm
        output_model_path : str or Path, optional
            Path to save the trained model
            
        Returns
        -------
        dict
            Trained phantom model
        """
        logger.info(f"Training phantom model with {len(phantom_images)} images")
        
        # Process phantom images
        phantom_stats = []
        
        for i, img_path in enumerate(phantom_images):
            try:
                # Load image if path is provided
                if isinstance(img_path, (str, Path)):
                    img = ants.image_read(str(img_path))
                else:
                    img = img_path
                
                # Calculate EIR
                eir = self.calculate_eir(img)
                logger.info(f"Phantom {i}: EIR = {eir:.2f}mm")
                
                # Adjust resolution if needed
                if abs(eir - target_eir) > 0.5:
                    logger.info(f"Adjusting phantom resolution from {eir:.2f}mm to {target_eir:.2f}mm")
                    img = self.adjust_resolution(img, target_eir)
                
                # Extract image data
                img_data = img.numpy()
                
                # Apply mask to focus on phantom regions
                mask = img_data > 0
                
                if np.sum(mask) > 0:
                    # Calculate statistics
                    mean = np.mean(img_data[mask])
                    std = np.std(img_data[mask])
                    
                    # Store statistics
                    phantom_stats.append({
                        'mean': mean,
                        'std': std,
                        'eir': target_eir
                    })
                else:
                    logger.warning(f"Skipping empty phantom image at index {i}")
            except Exception as e:
                logger.warning(f"Error processing phantom image at index {i}: {e}")
        
        if not phantom_stats:
            logger.error("No valid phantom images processed for model training")
            return {}
        
        # Create model
        model = {
            'target_resolution': target_eir,
            'phantom_stats': phantom_stats,
            'intensity_transform': lambda img_data: self._phantom_intensity_transform(
                img_data, phantom_stats
            )
        }
        
        # Save model if path is provided
        if output_model_path is not None:
            output_model_path = Path(output_model_path)
            output_model_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Create a serializable version of the model (without lambda functions)
            serializable_model = {k: v for k, v in model.items() if k != 'intensity_transform'}
            
            with open(output_model_path, 'wb') as f:
                pickle.dump(serializable_model, f)
            
            logger.info(f"Model saved to {output_model_path}")
        
        self.model = model
        return model
    
    def train_mri_model(self,
                       pet_images: List[Union[str, Path, ants.ANTsImage]],
                       mri_features: List[Dict[str, float]],
                       target_eir: float = 8.0,
                       output_model_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Train an MRI-guided harmonization model.
        
        Parameters
        ----------
        pet_images : list
            List of PET image paths or ANTsImage objects
        mri_features : list of dict
            List of MRI features for each PET image
        target_eir : float, optional
            Target effective image resolution in mm
        output_model_path : str or Path, optional
            Path to save the trained model
            
        Returns
        -------
        dict
            Trained MRI model
        """
        logger.info(f"Training MRI model with {len(pet_images)} image pairs")
        
        if len(pet_images) != len(mri_features):
            logger.error("Number of PET images and MRI features must match")
            return {}
        
        # Process image pairs
        pairs = []
        
        for i, (img_path, features) in enumerate(zip(pet_images, mri_features)):
            try:
                # Load image if path is provided
                if isinstance(img_path, (str, Path)):
                    img = ants.image_read(str(img_path))
                else:
                    img = img_path
                
                # Calculate EIR
                eir = self.calculate_eir(img)
                logger.info(f"Image {i}: EIR = {eir:.2f}mm")
                
                # Adjust resolution if needed
                if abs(eir - target_eir) > 0.5:
                    logger.info(f"Adjusting image resolution from {eir:.2f}mm to {target_eir:.2f}mm")
                    img = self.adjust_resolution(img, target_eir)
                
                # Extract image data
                img_data = img.numpy()
                
                # Apply mask to focus on brain tissue
                mask = img_data > 0
                
                if np.sum(mask) > 0:
                    # Calculate statistics
                    pet_mean = np.mean(img_data[mask])
                    pet_std = np.std(img_data[mask])
                    
                    # Store pair
                    pairs.append({
                        'pet_mean': pet_mean,
                        'pet_std': pet_std,
                        'mri_features': features,
                        'eir': target_eir
                    })
                else:
                    logger.warning(f"Skipping empty image at index {i}")
            except Exception as e:
                logger.warning(f"Error processing image pair at index {i}: {e}")
        
        if not pairs:
            logger.error("No valid image pairs processed for model training")
            return {}
        
        # Create model
        model = {
            'target_resolution': target_eir,
            'pairs': pairs,
            'transform': lambda pet_data, mri_features: self._mri_guided_transform(
                pet_data, mri_features, pairs
            )
        }
        
        # Save model if path is provided
        if output_model_path is not None:
            output_model_path = Path(output_model_path)
            output_model_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Create a serializable version of the model (without lambda functions)
            serializable_model = {k: v for k, v in model.items() if k != 'transform'}
            
            with open(output_model_path, 'wb') as f:
                pickle.dump(serializable_model, f)
            
            logger.info(f"Model saved to {output_model_path}")
        
        self.model = model
        return model
    
    def train_combat_model(self,
                          images: List[Union[str, Path, ants.ANTsImage]],
                          batches: List[str],
                          covariates: Optional[Dict[str, List]] = None,
                          target_eir: float = 8.0,
                          output_model_path: Optional[Union[str, Path]] = None) -> Dict:
        """
        Train a ComBat harmonization model.
        
        Parameters
        ----------
        images : list
            List of image paths or ANTsImage objects
        batches : list
            List of batch identifiers for each image
        covariates : dict, optional
            Dictionary of covariates for ComBat harmonization
        target_eir : float, optional
            Target effective image resolution in mm
        output_model_path : str or Path, optional
            Path to save the trained model
            
        Returns
        -------
        dict
            Trained ComBat model
        """
        logger.info(f"Training ComBat model with {len(images)} images from {len(set(batches))} batches")
        
        if len(images) != len(batches):
            logger.error("Number of images and batch identifiers must match")
            return {}
        
        # Process images
        data_list = []
        valid_batches = []
        valid_covariates = {}
        
        if covariates:
            for key, values in covariates.items():
                if len(values) != len(images):
                    logger.error(f"Covariate '{key}' length does not match number of images")
                    return {}
                valid_covariates[key] = []
        
        for i, (img_path, batch) in enumerate(zip(images, batches)):
            try:
                # Load image if path is provided
                if isinstance(img_path, (str, Path)):
                    img = ants.image_read(str(img_path))
                else:
                    img = img_path
                
                # Calculate EIR
                eir = self.calculate_eir(img)
                logger.info(f"Image {i} (batch {batch}): EIR = {eir:.2f}mm")
                
                # Adjust resolution if needed
                if abs(eir - target_eir) > 0.5:
                    logger.info(f"Adjusting image resolution from {eir:.2f}mm to {target_eir:.2f}mm")
                    img = self.adjust_resolution(img, target_eir)
                
                # Extract image data
                img_data = img.numpy()
                
                # Apply mask to focus on brain tissue
                mask = img_data > 0
                
                if np.sum(mask) > 0:
                    # Extract masked data
                    masked_data = img_data[mask]
                    
                    # Store data
                    data_list.append(masked_data)
                    valid_batches.append(batch)
                    
                    # Store covariates
                    if covariates:
                        for key, values in covariates.items():
                            valid_covariates[key].append(values[i])
                else:
                    logger.warning(f"Skipping empty image at index {i}")
            except Exception as e:
                logger.warning(f"Error processing image at index {i}: {e}")
        
        if not data_list:
            logger.error("No valid images processed for model training")
            return {}
        
        # Calculate batch statistics
        batch_stats = {}
        reference_batch = None
        reference_mean = None
        reference_std = None
        
        # Find the batch with the most samples to use as reference
        batch_counts = {}
        for batch in valid_batches:
            if batch not in batch_counts:
                batch_counts[batch] = 0
            batch_counts[batch] += 1
        
        reference_batch = max(batch_counts, key=batch_counts.get)
        logger.info(f"Using batch '{reference_batch}' as reference (most samples: {batch_counts[reference_batch]})")
        
        # Calculate statistics for each batch
        for batch in set(valid_batches):
            batch_indices = [i for i, b in enumerate(valid_batches) if b == batch]
            batch_data = [data_list[i] for i in batch_indices]
            
            # Concatenate data from the same batch
            batch_data_concat = np.concatenate(batch_data)
            
            # Calculate statistics
            mean = np.mean(batch_data_concat)
            std = np.std(batch_data_concat)
            
            batch_stats[batch] = {
                'mean': mean,
                'std': std,
                'count': len(batch_indices)
            }
            
            if batch == reference_batch:
                reference_mean = mean
                reference_std = std
        
        logger.info(f"Calculated statistics for {len(batch_stats)} batches")
        
        # Create ComBat transform function
        def combat_transform(data, batch):
            if batch not in batch_stats:
                logger.warning(f"Unknown batch: {batch}, using reference batch")
                return data
            
            # Get batch-specific statistics
            batch_mean = batch_stats[batch]['mean']
            batch_std = batch_stats[batch]['std']
            
            # Ensure we have valid statistics
            if batch_std == 0:
                logger.warning(f"Zero standard deviation for batch: {batch}")
                return data
            
            # Apply standardization
            standardized = (data - batch_mean) / batch_std
            
            # Transform to reference batch
            transformed = standardized * reference_std + reference_mean
            
            return transformed
        
        # Create model
        model = {
            'target_resolution': target_eir,
            'batch_stats': batch_stats,
            'reference_batch': reference_batch,
            'reference_mean': reference_mean,
            'reference_std': reference_std,
            'transform': combat_transform
        }
        
        # Save model if path is provided
        if output_model_path is not None:
            output_model_path = Path(output_model_path)
            output_model_path.parent.mkdir(exist_ok=True, parents=True)
            
            # Create a serializable version of the model (without lambda functions)
            serializable_model = {k: v for k, v in model.items() if k != 'transform'}
            
            with open(output_model_path, 'wb') as f:
                pickle.dump(serializable_model, f)
            
            logger.info(f"Model saved to {output_model_path}")
        
        self.model = model
        return model
    
    def extract_mri_features(self, mri_img: Union[str, Path, ants.ANTsImage],
                           segmentation: Optional[Union[str, Path, ants.ANTsImage]] = None) -> Dict[str, float]:
        """
        Extract features from MRI image for MRI-guided harmonization.
        
        Parameters
        ----------
        mri_img : str, Path, or ants.ANTsImage
            MRI image or path to image
        segmentation : str, Path, or ants.ANTsImage, optional
            Segmentation image or path to segmentation
            
        Returns
        -------
        dict
            Dictionary of MRI features
        """
        logger.info("Extracting MRI features for harmonization")
        
        # Load image if path is provided
        if isinstance(mri_img, (str, Path)):
            mri_img = ants.image_read(str(mri_img))
        
        # Extract basic image features
        features = {}
        
        # Calculate brain volume
        mri_data = mri_img.numpy()
        brain_mask = mri_data > 0
        features['brain_volume'] = np.sum(brain_mask) * np.prod(mri_img.spacing) / 1000.0  # in cm³
        
        # If segmentation is provided, extract tissue-specific features
        if segmentation is not None:
            # Load segmentation if path is provided
            if isinstance(segmentation, (str, Path)):
                seg_img = ants.image_read(str(segmentation))
            else:
                seg_img = segmentation
            
            # Extract segmentation data
            seg_data = seg_img.numpy()
            
            # Calculate tissue volumes and ratios
            # Assuming standard segmentation labels: 1=GM, 2=WM, 3=CSF
            gm_mask = seg_data == 1
            wm_mask = seg_data == 2
            csf_mask = seg_data == 3
            
            # Calculate volumes in cm³
            gm_volume = np.sum(gm_mask) * np.prod(seg_img.spacing) / 1000.0
            wm_volume = np.sum(wm_mask) * np.prod(seg_img.spacing) / 1000.0
            csf_volume = np.sum(csf_mask) * np.prod(seg_img.spacing) / 1000.0
            
            # Calculate tissue ratios
            total_volume = gm_volume + wm_volume + csf_volume
            if total_volume > 0:
                features['gm_ratio'] = gm_volume / total_volume
                features['wm_ratio'] = wm_volume / total_volume
                features['csf_ratio'] = csf_volume / total_volume
            else:
                features['gm_ratio'] = 0.0
                features['wm_ratio'] = 0.0
                features['csf_ratio'] = 0.0
            
            features['gm_volume'] = gm_volume
            features['wm_volume'] = wm_volume
            features['csf_volume'] = csf_volume
            
            # Calculate intensity statistics for each tissue type
            if np.sum(gm_mask) > 0:
                features['gm_mean'] = np.mean(mri_data[gm_mask])
                features['gm_std'] = np.std(mri_data[gm_mask])
            else:
                features['gm_mean'] = 0.0
                features['gm_std'] = 0.0
            
            if np.sum(wm_mask) > 0:
                features['wm_mean'] = np.mean(mri_data[wm_mask])
                features['wm_std'] = np.std(mri_data[wm_mask])
            else:
                features['wm_mean'] = 0.0
                features['wm_std'] = 0.0
            
            if np.sum(csf_mask) > 0:
                features['csf_mean'] = np.mean(mri_data[csf_mask])
                features['csf_std'] = np.std(mri_data[csf_mask])
            else:
                features['csf_mean'] = 0.0
                features['csf_std'] = 0.0
            
            # Calculate GM/WM contrast ratio
            if features['wm_mean'] > 0:
                features['gm_wm_ratio'] = features['gm_mean'] / features['wm_mean']
            else:
                features['gm_wm_ratio'] = 1.0
        else:
            # If no segmentation is provided, use intensity-based features
            logger.info("No segmentation provided, using intensity-based features")
            
            # Apply simple thresholding to estimate tissue types
            # Normalize image first
            mri_data_norm = (mri_data - np.min(mri_data[brain_mask])) / (np.max(mri_data[brain_mask]) - np.min(mri_data[brain_mask]))
            
            # Estimate tissue types using Otsu's method
            try:
                from skimage.filters import threshold_otsu
                from skimage.filters import threshold_multiotsu
                
                # Try multi-Otsu for 3 classes
                try:
                    thresholds = threshold_multiotsu(mri_data_norm[brain_mask], classes=3)
                    
                    # Create masks for estimated tissue types
                    est_csf_mask = np.logical_and(brain_mask, mri_data_norm < thresholds[0])
                    est_gm_mask = np.logical_and(brain_mask, 
                                              np.logical_and(mri_data_norm >= thresholds[0], 
                                                           mri_data_norm < thresholds[1]))
                    est_wm_mask = np.logical_and(brain_mask, mri_data_norm >= thresholds[1])
                except Exception:
                    # Fallback to single Otsu threshold
                    threshold = threshold_otsu(mri_data_norm[brain_mask])
                    
                    # Estimate GM/WM boundary
                    est_gm_mask = np.logical_and(brain_mask, mri_data_norm < threshold)
                    est_wm_mask = np.logical_and(brain_mask, mri_data_norm >= threshold)
                    est_csf_mask = np.zeros_like(brain_mask)
                
                # Calculate estimated tissue volumes and ratios
                est_gm_volume = np.sum(est_gm_mask) * np.prod(mri_img.spacing) / 1000.0
                est_wm_volume = np.sum(est_wm_mask) * np.prod(mri_img.spacing) / 1000.0
                est_csf_volume = np.sum(est_csf_mask) * np.prod(mri_img.spacing) / 1000.0
                
                # Calculate tissue ratios
                total_volume = est_gm_volume + est_wm_volume + est_csf_volume
                if total_volume > 0:
                    features['gm_ratio'] = est_gm_volume / total_volume
                    features['wm_ratio'] = est_wm_volume / total_volume
                    features['csf_ratio'] = est_csf_volume / total_volume
                else:
                    features['gm_ratio'] = 0.0
                    features['wm_ratio'] = 0.0
                    features['csf_ratio'] = 0.0
                
                features['gm_volume'] = est_gm_volume
                features['wm_volume'] = est_wm_volume
                features['csf_volume'] = est_csf_volume
                
                # Calculate intensity statistics for each tissue type
                if np.sum(est_gm_mask) > 0:
                    features['gm_mean'] = np.mean(mri_data[est_gm_mask])
                    features['gm_std'] = np.std(mri_data[est_gm_mask])
                else:
                    features['gm_mean'] = 0.0
                    features['gm_std'] = 0.0
                
                if np.sum(est_wm_mask) > 0:
                    features['wm_mean'] = np.mean(mri_data[est_wm_mask])
                    features['wm_std'] = np.std(mri_data[est_wm_mask])
                else:
                    features['wm_mean'] = 0.0
                    features['wm_std'] = 0.0
                
                if np.sum(est_csf_mask) > 0:
                    features['csf_mean'] = np.mean(mri_data[est_csf_mask])
                    features['csf_std'] = np.std(mri_data[est_csf_mask])
                else:
                    features['csf_mean'] = 0.0
                    features['csf_std'] = 0.0
                
                # Calculate GM/WM contrast ratio
                if features['wm_mean'] > 0:
                    features['gm_wm_ratio'] = features['gm_mean'] / features['wm_mean']
                else:
                    features['gm_wm_ratio'] = 1.0
                
            except ImportError:
                logger.warning("scikit-image not available for Otsu thresholding")
                # Use simple percentile-based thresholding
                sorted_intensities = np.sort(mri_data[brain_mask])
                total_voxels = len(sorted_intensities)
                
                # Estimate CSF as lowest 20%, WM as highest 30%, GM as the rest
                csf_threshold = sorted_intensities[int(total_voxels * 0.2)]
                wm_threshold = sorted_intensities[int(total_voxels * 0.7)]
                
                est_csf_mask = np.logical_and(brain_mask, mri_data <= csf_threshold)
                est_wm_mask = np.logical_and(brain_mask, mri_data >= wm_threshold)
                est_gm_mask = np.logical_and(brain_mask, 
                                          np.logical_and(mri_data > csf_threshold, 
                                                       mri_data < wm_threshold))
                
                # Calculate estimated tissue volumes and ratios
                est_gm_volume = np.sum(est_gm_mask) * np.prod(mri_img.spacing) / 1000.0
                est_wm_volume = np.sum(est_wm_mask) * np.prod(mri_img.spacing) / 1000.0
                est_csf_volume = np.sum(est_csf_mask) * np.prod(mri_img.spacing) / 1000.0
                
                # Calculate tissue ratios
                total_volume = est_gm_volume + est_wm_volume + est_csf_volume
                if total_volume > 0:
                    features['gm_ratio'] = est_gm_volume / total_volume
                    features['wm_ratio'] = est_wm_volume / total_volume
                    features['csf_ratio'] = est_csf_volume / total_volume
                else:
                    features['gm_ratio'] = 0.0
                    features['wm_ratio'] = 0.0
                    features['csf_ratio'] = 0.0
                
                features['gm_volume'] = est_gm_volume
                features['wm_volume'] = est_wm_volume
                features['csf_volume'] = est_csf_volume
                
                # Calculate intensity statistics for each tissue type
                if np.sum(est_gm_mask) > 0:
                    features['gm_mean'] = np.mean(mri_data[est_gm_mask])
                    features['gm_std'] = np.std(mri_data[est_gm_mask])
                else:
                    features['gm_mean'] = 0.0
                    features['gm_std'] = 0.0
                
                if np.sum(est_wm_mask) > 0:
                    features['wm_mean'] = np.mean(mri_data[est_wm_mask])
                    features['wm_std'] = np.std(mri_data[est_wm_mask])
                else:
                    features['wm_mean'] = 0.0
                    features['wm_std'] = 0.0
                
                if np.sum(est_csf_mask) > 0:
                    features['csf_mean'] = np.mean(mri_data[est_csf_mask])
                    features['csf_std'] = np.std(mri_data[est_csf_mask])
                else:
                    features['csf_mean'] = 0.0
                    features['csf_std'] = 0.0
                
                # Calculate GM/WM contrast ratio
                if features['wm_mean'] > 0:
                    features['gm_wm_ratio'] = features['gm_mean'] / features['wm_mean']
                else:
                    features['gm_wm_ratio'] = 1.0
        
        # Add additional features
        # Calculate histogram features
        hist, bin_edges = np.histogram(mri_data[brain_mask], bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find histogram peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=np.max(hist) * 0.2)
        
        features['histogram_peaks'] = len(peaks)
        if len(peaks) > 0:
            features['peak_positions'] = [bin_centers[p] for p in peaks]
        else:
            features['peak_positions'] = []
        
        logger.info(f"Extracted {len(features)} MRI features")
        return features

    def _mri_guided_transform(self, pet_data: np.ndarray, 
                             mri_features: Dict[str, float],
                             model_pairs: List[Dict]) -> np.ndarray:
        """
        Apply MRI-guided transformation to PET data.
        
        Parameters
        ----------
        pet_data : np.ndarray
            Input PET image data
        mri_features : dict
            MRI features for the current image
        model_pairs : list of dict
            Model data containing paired PET-MRI information
            
        Returns
        -------
        np.ndarray
            Transformed PET data
        """
        # Apply mask to focus on brain tissue
        mask = pet_data > 0
        
        if np.sum(mask) == 0:
            logger.warning("Empty image provided for MRI-guided harmonization.")
            return pet_data
        
        # Calculate image statistics
        pet_mean = np.mean(pet_data[mask])
        pet_std = np.std(pet_data[mask])
        
        if pet_std == 0:
            logger.warning("Zero standard deviation in PET image.")
            return pet_data
        
        # Find most similar MRI features in the model
        # Calculate similarity scores based on MRI features
        similarity_scores = []
        
        # Define feature weights (can be adjusted based on importance)
        feature_weights = {
            'gm_ratio': 2.0,
            'wm_ratio': 2.0,
            'csf_ratio': 1.0,
            'gm_wm_ratio': 3.0,
            'brain_volume': 0.5
        }
        
        # Calculate weighted similarity for each pair in the model
        for pair in model_pairs:
            pair_features = pair['mri_features']
            
            # Calculate normalized feature differences
            diff_sum = 0
            weight_sum = 0
            
            for feature, weight in feature_weights.items():
                if feature in mri_features and feature in pair_features:
                    # Normalize the difference
                    feature_diff = abs(mri_features[feature] - pair_features[feature])
                    
                    # Scale by feature-specific normalization factor
                    if feature == 'brain_volume':
                        # Volume differences can be large, normalize more aggressively
                        norm_diff = feature_diff / (pair_features[feature] + 1e-6)
                    elif feature.endswith('_ratio'):
                        # Ratios are already normalized
                        norm_diff = feature_diff
                    else:
                        # Default normalization
                        norm_diff = feature_diff / (abs(pair_features[feature]) + 1e-6)
                    
                    diff_sum += weight * norm_diff
                    weight_sum += weight
            
            # Calculate final similarity score (lower is better)
            if weight_sum > 0:
                similarity = diff_sum / weight_sum
            else:
                similarity = float('inf')
            
            similarity_scores.append((similarity, pair))
        
        # Sort by similarity (lowest first)
        similarity_scores.sort(key=lambda x: x[0])
        
        # Take the top N most similar pairs
        n_similar = min(5, len(similarity_scores))
        top_pairs = [pair for _, pair in similarity_scores[:n_similar]]
        
        if not top_pairs:
            logger.warning("No similar MRI features found in model.")
            return pet_data
        
        # Calculate weighted average of reference statistics
        total_weight = 0
        weighted_mean = 0
        weighted_std = 0
        
        for i, pair in enumerate(top_pairs):
            # Weight by similarity (closer matches get higher weight)
            weight = 1.0 / (i + 1)
            total_weight += weight
            
            weighted_mean += weight * pair['pet_mean']
            weighted_std += weight * pair['pet_std']
        
        # Normalize weights
        ref_mean = weighted_mean / total_weight
        ref_std = weighted_std / total_weight
        
        # Apply z-score normalization and rescaling
        normalized = np.zeros_like(pet_data)
        normalized[mask] = (pet_data[mask] - pet_mean) / pet_std
        
        # Transform to reference statistics
        transformed = normalized * ref_std + ref_mean
        
        return transformed
    
    def _apply_simple_pvc(self, img: ants.ANTsImage, 
                         metadata: Optional[Dict] = None) -> ants.ANTsImage:
        """
        Apply simple partial volume correction without segmentation.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
        metadata : dict, optional
            Image metadata
            
        Returns
        -------
        ants.ANTsImage
            PVC-corrected image
        """
        # Simple PVC using Richardson-Lucy deconvolution
        img_data = img.numpy()
        
        # Create mask for brain tissue
        mask = img_data > 0
        
        if np.sum(mask) == 0:
            logger.warning("Empty image provided for PVC.")
            return img
        
        # Estimate PSF based on image resolution
        eir = self.calculate_eir(img)
        
        # Convert FWHM to sigma
        sigma_mm = eir / 2.355
        
        # Convert sigma from mm to voxels
        sigma_voxels = [sigma_mm / s for s in img.spacing]
        
        # Create PSF
        psf_size = [int(4 * s) | 1 for s in sigma_voxels]  # Ensure odd size
        
        # Create coordinate grids for PSF
        coords = [np.arange(-(s//2), -(s//2) + s) for s in psf_size]
        grid = np.meshgrid(*coords, indexing='ij')
        
        # Create Gaussian PSF
        psf = np.exp(-0.5 * sum([x**2 / s**2 for x, s in zip(grid, sigma_voxels)]))
        psf /= psf.sum()  # Normalize
        
        # Apply Richardson-Lucy deconvolution
        pvc_data = img_data.copy()
        
        # Only process within mask to save computation
        masked_data = img_data.copy()
        
        # Apply deconvolution iterations
        for i in range(10):  # Number of iterations
            # Forward model: blur current estimate
            blurred = ndimage.convolve(masked_data, psf, mode='constant')
            
            # Avoid division by zero
            blurred = np.maximum(blurred, 1e-10)
            
            # Update step
            relative_blur = img_data / blurred
            correction = ndimage.correlate(relative_blur, psf, mode='constant')
            
            # Apply correction
            masked_data *= correction
            
            # Apply regularization to suppress noise
            if i % 2 == 0:
                masked_data = ndimage.gaussian_filter(masked_data, 0.5)
        
        # Apply result only within mask
        pvc_data[mask] = masked_data[mask]
        
        # Create new ANTs image
        pvc_img = ants.from_numpy(pvc_data, origin=img.origin,
                                spacing=img.spacing,
                                direction=img.direction)
        
        return pvc_img
    
    def load_model(self, model_path: Union[str, Path]) -> Dict:
        """
        Load a previously trained harmonization model.
        
        Parameters
        ----------
        model_path : str or Path
            Path to the saved model file
            
        Returns
        -------
        dict
            Loaded model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return {}
        
        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            logger.info(f"Model loaded from {model_path}")
            
            # Reconstruct transform functions based on model type
            if 'phantom_stats' in loaded_model:
                # Phantom model
                loaded_model['intensity_transform'] = lambda img_data: self._phantom_intensity_transform(
                    img_data, loaded_model['phantom_stats']
                )
                self.method = 'phantom'
            elif 'pairs' in loaded_model:
                # MRI model
                loaded_model['transform'] = lambda pet_data, mri_features: self._mri_guided_transform(
                    pet_data, mri_features, loaded_model['pairs']
                )
                self.method = 'mri'
            elif 'batch_stats' in loaded_model:
                # ComBat model
                def combat_transform(data, batch):
                    if batch not in loaded_model['batch_stats']:
                        logger.warning(f"Unknown batch: {batch}, using reference batch")
                        return data
                    
                    # Get batch-specific statistics
                    batch_mean = loaded_model['batch_stats'][batch]['mean']
                    batch_std = loaded_model['batch_stats'][batch]['std']
                    
                    # Ensure we have valid statistics
                    if batch_std == 0:
                        logger.warning(f"Zero standard deviation for batch: {batch}")
                        return data
                    
                    # Apply standardization
                    standardized = (data - batch_mean) / batch_std
                    
                    # Transform to reference batch
                    transformed = standardized * loaded_model['reference_std'] + loaded_model['reference_mean']
                    
                    return transformed
                
                loaded_model['transform'] = combat_transform
                self.method = 'combat'
            else:
                logger.warning("Unknown model type")
            
            # Set target EIR if available
            if 'target_resolution' in loaded_model:
                self.target_eir = loaded_model['target_resolution']
            
            self.model = loaded_model
            return loaded_model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {}
    
    def calculate_eir(self, img: ants.ANTsImage) -> float:
        """
        Calculate the effective image resolution (EIR) of a PET image.
        
        Parameters
        ----------
        img : ants.ANTsImage
            Input image
            
        Returns
        -------
        float
            Estimated EIR in mm
        """
        # Extract image data
        img_data = img.numpy()
        
        # Create mask for brain tissue
        mask = img_data > 0
        
        if np.sum(mask) == 0:
            logger.warning("Empty image provided for EIR calculation.")
            return 8.0  # Default value
        
        # Method 1: Estimate from gradient distribution
        # Calculate gradients in each direction
        gradients = np.gradient(img_data)
        
        # Calculate gradient magnitudes within mask
        grad_mag = np.zeros_like(img_data)
        for grad in gradients:
            grad_mag += grad**2
        grad_mag = np.sqrt(grad_mag)
        
        # Calculate normalized gradient histogram
        hist, bin_edges = np.histogram(grad_mag[mask], bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find the peak of the gradient histogram
        peak_idx = np.argmax(hist)
        peak_grad = bin_centers[peak_idx]
        
        # Estimate FWHM from gradient peak
        # Higher gradient peak indicates sharper image (lower FWHM)
        # This is an empirical relationship
        if peak_grad > 0:
            estimated_fwhm = 10.0 / (peak_grad + 0.5)  # Empirical formula
            estimated_fwhm = np.clip(estimated_fwhm, 4.0, 12.0)  # Reasonable range for PET
        else:
            estimated_fwhm = 8.0  # Default value
        
        # Method 2: Estimate from spatial autocorrelation
        # Calculate autocorrelation in each direction
        ac_widths = []
        
        for axis in range(img_data.ndim):
            # Extract 1D profiles along this axis
            profiles = []
            slices = [slice(None)] * img_data.ndim
            
            # Sample multiple profiles
            for _ in range(10):
                # Random position for other dimensions
                for other_axis in range(img_data.ndim):
                    if other_axis != axis:
                        max_idx = img_data.shape[other_axis] - 1
                        if max_idx > 0:
                            slices[other_axis] = np.random.randint(0, max_idx)
                
                # Extract profile
                profile = img_data[tuple(slices)]
                
                # Only use profiles with sufficient signal
                if np.max(profile) > 0.1 * np.max(img_data):
                    profiles.append(profile)
            
            if profiles:
                # Average autocorrelation width
                width_sum = 0
                count = 0
                
                for profile in profiles:
                    # Calculate autocorrelation
                    autocorr = np.correlate(profile, profile, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
                    
                    # Normalize
                    autocorr = autocorr / autocorr[0]
                    
                    # Find width at half maximum
                    try:
                        half_idx = np.where(autocorr < 0.5)[0][0]
                        width_voxels = half_idx
                        
                        # Convert to mm
                        width_mm = width_voxels * img.spacing[axis]
                        
                        width_sum += width_mm
                        count += 1
                    except IndexError:
                        pass
                
                if count > 0:
                    ac_widths.append(width_sum / count)
        
        # Calculate average autocorrelation width
        if ac_widths:
            ac_fwhm = 2.0 * np.mean(ac_widths)  # Convert to FWHM
            ac_fwhm = np.clip(ac_fwhm, 4.0, 12.0)  # Reasonable range for PET
            
            # Combine estimates
            estimated_fwhm = 0.7 * estimated_fwhm + 0.3 * ac_fwhm
        
        logger.info(f"Estimated EIR: {estimated_fwhm:.2f}mm")
        return estimated_fwhm
