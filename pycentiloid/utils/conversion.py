"""
Conversion utilities for PyCentiloid.

This module provides functions for converting between different medical image formats,
particularly NIFTI and DICOM.
"""

import os
import numpy as np
import pydicom
import nibabel as nib
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, List, Dict, Tuple

def nifti_to_dicom(nifti_path: Union[str, Path],
                  dicom_template_path: Union[str, Path],
                  output_dir: Union[str, Path],
                  series_description: Optional[str] = None,
                  series_number: Optional[int] = None,
                  patient_info: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Convert NIFTI image to DICOM series.
    
    Parameters
    ----------
    nifti_path : str or Path
        Path to input NIFTI file
    dicom_template_path : str or Path
        Path to template DICOM file or directory
    output_dir : str or Path
        Directory to save DICOM files
    series_description : str, optional
        Description for the DICOM series
    series_number : int, optional
        Series number for the DICOM series
    patient_info : dict, optional
        Patient information to include in DICOM headers
        
    Returns
    -------
    list
        List of paths to created DICOM files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load NIFTI image
    nii_img = nib.load(str(nifti_path))
    nii_data = nii_img.get_fdata()
    
    # Get image dimensions and orientation
    dims = nii_data.shape
    affine = nii_img.affine
    
    # Load template DICOM
    template_path = Path(dicom_template_path)
    if template_path.is_dir():
        # Find first DICOM file in directory
        dicom_files = list(template_path.glob('*.dcm'))
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {template_path}")
        template_dcm = pydicom.dcmread(str(dicom_files[0]))
    else:
        template_dcm = pydicom.dcmread(str(template_path))
    
    # Generate new UIDs
    study_instance_uid = template_dcm.StudyInstanceUID
    series_instance_uid = pydicom.uid.generate_uid()
    
    # Set series description and number if provided
    if series_description is not None:
        template_dcm.SeriesDescription = series_description
    
    if series_number is not None:
        template_dcm.SeriesNumber = series_number
    
    # Update patient info if provided
    if patient_info:
        for key, value in patient_info.items():
            if hasattr(template_dcm, key):
                setattr(template_dcm, key, value)
    
    # Calculate voxel spacing and orientation from NIFTI affine
    voxel_spacing = nib.affines.voxel_sizes(affine)
    orientation_matrix = affine[:3, :3] / voxel_spacing
    
    # Create DICOM files for each slice
    output_files = []
    for i in range(dims[2]):
        # Create new DICOM dataset from template
        ds = template_dcm.copy()
        
        # Update UIDs
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = series_instance_uid
        ds.StudyInstanceUID = study_instance_uid
        
        # Set instance number
        ds.InstanceNumber = i + 1
        
        # Set image position and orientation
        slice_pos = affine.dot(np.array([0, 0, i, 1]))[:3]
        ds.ImagePositionPatient = slice_pos.tolist()
        
        # Calculate image orientation
        row_vector = orientation_matrix[:, 0]
        col_vector = orientation_matrix[:, 1]
        ds.ImageOrientationPatient = row_vector.tolist() + col_vector.tolist()
        
        # Set pixel spacing
        ds.PixelSpacing = [voxel_spacing[0], voxel_spacing[1]]
        ds.SliceThickness = voxel_spacing[2]
        
        # Extract slice data and rescale to uint16
        slice_data = nii_data[:, :, i]
        
        # Normalize to 16-bit range
        if slice_data.min() != slice_data.max():
            slice_data = ((slice_data - slice_data.min()) / 
                         (slice_data.max() - slice_data.min()) * 65535)
        slice_data = slice_data.astype(np.uint16)
        
        # Set pixel data
        ds.Rows, ds.Columns = slice_data.shape
        ds.PixelData = slice_data.tobytes()
        
        # Set transfer syntax and other required attributes
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        # Set date and time
        now = datetime.now()
        ds.ContentDate = now.strftime("%Y%m%d")
        ds.ContentTime = now.strftime("%H%M%S")
        
        # Save DICOM file
        output_file = output_dir / f"slice_{i:04d}.dcm"
        ds.save_as(str(output_file))
        output_files.append(str(output_file))
    
    return output_files

def dicom_to_nifti(dicom_dir: Union[str, Path],
                  output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Convert DICOM series to NIFTI image.
    
    Parameters
    ----------
    dicom_dir : str or Path
        Directory containing DICOM files
    output_path : str or Path, optional
        Path to save NIFTI file
        
    Returns
    -------
    str
        Path to created NIFTI file
    """
    # Create output path if not provided
    if output_path is None:
        dicom_dir = Path(dicom_dir)
        output_path = dicom_dir.parent / f"{dicom_dir.name}.nii.gz"
    
    # Find all DICOM files
    dicom_dir = Path(dicom_dir)
    dicom_files = sorted(list(dicom_dir.glob('*.dcm')))
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")
    
    # Load all DICOM files
    slices = [pydicom.dcmread(str(f)) for f in dicom_files]
    
    # Sort slices by instance number or position
    try:
        slices.sort(key=lambda x: x.InstanceNumber)
    except AttributeError:
        # If InstanceNumber is not available, sort by ImagePositionPatient
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Extract image data
    img_shape = [len(slices), slices[0].Rows, slices[0].Columns]
    img_data = np.zeros(img_shape, dtype=np.float32)
    
    for i, slice_data in enumerate(slices):
        img_data[i, :, :] = slice_data.pixel_array
    
    # Calculate affine matrix
    # Get position and orientation from first slice
    pos = slices[0].ImagePositionPatient
    orientation = slices[0].ImageOrientationPatient
    pixel_spacing = slices[0].PixelSpacing
    
    # Extract orientation vectors
    row_vector = np.array(orientation[:3])
    col_vector = np.array(orientation[3:])
    
    # Calculate slice vector from positions of first and last slice
    if len(slices) > 1:
        last_pos = slices[-1].ImagePositionPatient
        slice_vector = np.array(last_pos) - np.array(pos)
        slice_vector = slice_vector / (len(slices) - 1)
    else:
        # For single slice, use cross product of row and column vectors
        slice_vector = np.cross(row_vector, col_vector)
        if hasattr(slices[0], 'SliceThickness'):
            slice_vector = slice_vector * slices[0].SliceThickness
    
    # Create affine matrix
    affine = np.zeros((4, 4))
    affine[:3, 0] = row_vector * pixel_spacing[0]
    affine[:3, 1] = col_vector * pixel_spacing[1]
    affine[:3, 2] = slice_vector
    affine[:3, 3] = pos
    affine[3, 3] = 1.0
    
    # Create NIFTI image
    nii_img = nib.Nifti1Image(img_data, affine)
    
    # Save NIFTI file
    nib.save(nii_img, str(output_path))
    
    return str(output_path)