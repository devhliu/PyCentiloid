"""
PSF estimation and deconvolution module for PyCentiloid.
This module provides functionality to estimate the Point Spread Function (PSF)
of Brain PET images using paired T1W MRI images and perform deconvolution
using the estimated PSF kernel.
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.optimize import minimize
from skimage.restoration import richardson_lucy, wiener


def estimate_psf_from_mri_pet(pet_img, mri_img, mask_img=None, initial_fwhm=6.0):
    """
    Estimate the PSF of a PET image using a paired T1W MRI image.
    
    Parameters
    ----------
    pet_img : str or nibabel.Nifti1Image
        Path to PET image or nibabel image object
    mri_img : str or nibabel.Nifti1Image
        Path to T1W MRI image or nibabel image object
    mask_img : str or nibabel.Nifti1Image, optional
        Brain mask to restrict analysis
    initial_fwhm : float, optional
        Initial guess for FWHM in mm
        
    Returns
    -------
    dict
        Dictionary containing estimated FWHM values for x, y, z dimensions
        and the corresponding Gaussian PSF kernel
    """
    # Load images if paths are provided
    if isinstance(pet_img, str):
        pet_img = nib.load(pet_img)
    if isinstance(mri_img, str):
        mri_img = nib.load(mri_img)
    if isinstance(mask_img, str) and mask_img is not None:
        mask_img = nib.load(mask_img)
    
    # Get image data
    pet_data = pet_img.get_fdata()
    mri_data = mri_img.get_fdata()
    
    # Apply mask if provided
    if mask_img is not None:
        mask_data = mask_img.get_fdata() > 0
        pet_data = pet_data * mask_data
        mri_data = mri_data * mask_data
    
    # Normalize images
    pet_data = (pet_data - pet_data.min()) / (pet_data.max() - pet_data.min())
    mri_data = (mri_data - mri_data.min()) / (mri_data.max() - mri_data.min())
    
    # Get voxel dimensions
    voxel_size = pet_img.header.get_zooms()[:3]
    
    # Define objective function for optimization
    def objective_function(fwhm_values):
        # Create Gaussian kernel with current FWHM values
        sigma_values = [fwhm / (2 * np.sqrt(2 * np.log(2))) for fwhm in fwhm_values]
        blurred_mri = gaussian_blur_3d(mri_data, sigma_values, voxel_size)
        
        # Calculate mean squared error between blurred MRI and PET
        mse = np.mean((blurred_mri - pet_data)**2)
        return mse
    
    # Optimize FWHM values
    initial_guess = [initial_fwhm, initial_fwhm, initial_fwhm]
    bounds = [(1.0, 20.0), (1.0, 20.0), (1.0, 20.0)]  # Reasonable bounds for FWHM in mm
    
    result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    optimal_fwhm = result.x
    
    # Create optimal PSF kernel
    optimal_sigma = [fwhm / (2 * np.sqrt(2 * np.log(2))) for fwhm in optimal_fwhm]
    kernel = create_gaussian_kernel_3d(optimal_sigma, voxel_size)
    
    return {
        'fwhm_x': optimal_fwhm[0],
        'fwhm_y': optimal_fwhm[1],
        'fwhm_z': optimal_fwhm[2],
        'kernel': kernel
    }


def gaussian_blur_3d(image, sigma_values, voxel_size=None):
    """
    Apply 3D Gaussian blur to an image.
    
    Parameters
    ----------
    image : ndarray
        3D image data
    sigma_values : list of float
        Sigma values for x, y, z dimensions in mm
    voxel_size : list of float, optional
        Voxel dimensions in mm
        
    Returns
    -------
    ndarray
        Blurred image
    """
    if voxel_size is not None:
        # Convert sigma from mm to voxels
        sigma_voxels = [sigma / voxel for sigma, voxel in zip(sigma_values, voxel_size)]
    else:
        sigma_voxels = sigma_values
    
    return ndimage.gaussian_filter(image, sigma_voxels)


def create_gaussian_kernel_3d(sigma_values, voxel_size=None, kernel_size=None):
    """
    Create a 3D Gaussian kernel.
    
    Parameters
    ----------
    sigma_values : list of float
        Sigma values for x, y, z dimensions in mm
    voxel_size : list of float, optional
        Voxel dimensions in mm
    kernel_size : list of int, optional
        Size of kernel in voxels
        
    Returns
    -------
    ndarray
        3D Gaussian kernel
    """
    if voxel_size is not None:
        # Convert sigma from mm to voxels
        sigma_voxels = [sigma / voxel for sigma, voxel in zip(sigma_values, voxel_size)]
    else:
        sigma_voxels = sigma_values
    
    # Determine kernel size if not provided
    if kernel_size is None:
        # Rule of thumb: kernel size = 2 * ceil(3 * sigma) + 1
        kernel_size = [2 * int(np.ceil(3 * sigma)) + 1 for sigma in sigma_voxels]
    
    # Create coordinate grids
    x = np.arange(-(kernel_size[0] // 2), kernel_size[0] // 2 + 1)
    y = np.arange(-(kernel_size[1] // 2), kernel_size[1] // 2 + 1)
    z = np.arange(-(kernel_size[2] // 2), kernel_size[2] // 2 + 1)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Create Gaussian kernel
    kernel = np.exp(-(xx**2 / (2 * sigma_voxels[0]**2) + 
                      yy**2 / (2 * sigma_voxels[1]**2) + 
                      zz**2 / (2 * sigma_voxels[2]**2)))
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    return kernel


def deconvolve_pet_image(pet_img, psf_kernel=None, psf_fwhm=None, voxel_size=None, 
                         method='richardson_lucy', iterations=10, reg_param=0.01):
    """
    Deconvolve a PET image using a PSF kernel.
    
    Parameters
    ----------
    pet_img : str or nibabel.Nifti1Image
        Path to PET image or nibabel image object
    psf_kernel : ndarray, optional
        PSF kernel to use for deconvolution
    psf_fwhm : list of float, optional
        FWHM values for x, y, z dimensions in mm (used if psf_kernel not provided)
    voxel_size : list of float, optional
        Voxel dimensions in mm (used if psf_kernel not provided)
    method : str, optional
        Deconvolution method: 'richardson_lucy' or 'wiener'
    iterations : int, optional
        Number of iterations for Richardson-Lucy deconvolution
    reg_param : float, optional
        Regularization parameter for Wiener deconvolution
        
    Returns
    -------
    nibabel.Nifti1Image
        Deconvolved PET image
    """
    # Load image if path is provided
    if isinstance(pet_img, str):
        pet_img = nib.load(pet_img)
    
    pet_data = pet_img.get_fdata()
    
    # Get voxel dimensions if not provided
    if voxel_size is None:
        voxel_size = pet_img.header.get_zooms()[:3]
    
    # Create PSF kernel if not provided
    if psf_kernel is None and psf_fwhm is not None:
        sigma_values = [fwhm / (2 * np.sqrt(2 * np.log(2))) for fwhm in psf_fwhm]
        psf_kernel = create_gaussian_kernel_3d(sigma_values, voxel_size)
    
    if psf_kernel is None:
        raise ValueError("Either psf_kernel or psf_fwhm must be provided")
    
    # Perform deconvolution
    if method == 'richardson_lucy':
        deconvolved_data = richardson_lucy(pet_data, psf_kernel, iterations=iterations)
    elif method == 'wiener':
        deconvolved_data = wiener(pet_data, psf_kernel, reg=reg_param)
    else:
        raise ValueError(f"Unknown deconvolution method: {method}")
    
    # Create new image with deconvolved data
    deconvolved_img = nib.Nifti1Image(deconvolved_data, pet_img.affine, pet_img.header)
    
    return deconvolved_img


def apply_psf_correction(pet_path, mri_path, output_path, mask_path=None, 
                         method='richardson_lucy', iterations=10):
    """
    Apply PSF estimation and correction to a PET image using a paired T1W MRI image.
    
    Parameters
    ----------
    pet_path : str
        Path to PET image
    mri_path : str
        Path to T1W MRI image
    output_path : str
        Path to save deconvolved PET image
    mask_path : str, optional
        Path to brain mask
    method : str, optional
        Deconvolution method: 'richardson_lucy' or 'wiener'
    iterations : int, optional
        Number of iterations for Richardson-Lucy deconvolution
        
    Returns
    -------
    dict
        Dictionary containing PSF estimation results and output path
    """
    # Estimate PSF
    psf_results = estimate_psf_from_mri_pet(pet_path, mri_path, mask_path)
    
    # Deconvolve PET image
    deconvolved_img = deconvolve_pet_image(
        pet_path, 
        psf_kernel=psf_results['kernel'],
        method=method,
        iterations=iterations
    )
    
    # Save deconvolved image
    nib.save(deconvolved_img, output_path)
    
    # Return results
    return {
        'fwhm_x': psf_results['fwhm_x'],
        'fwhm_y': psf_results['fwhm_y'],
        'fwhm_z': psf_results['fwhm_z'],
        'output_path': output_path
    }