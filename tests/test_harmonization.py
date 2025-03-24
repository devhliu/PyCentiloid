"""
Tests for the harmonization module in PyCentiloid.

This module tests the functionality of the ImageHarmonization class,
including resolution adjustment, different harmonization methods,
and model training.
"""

import os
import pytest
import numpy as np
import ants
import tempfile
from pathlib import Path
import argparse
import logging

from pycentiloid.processing.harmonization import ImageHarmonization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_harmonization')

def test_real_data_harmonization(pet_path, mri_path, output_dir=None, method='mri', apply_pvc=False):
    """
    Test harmonization with real PET and MRI data.
    
    Parameters
    ----------
    pet_path : str or Path
        Path to the PET image file (.nii.gz)
    mri_path : str or Path
        Path to the paired MRI T1w image file (.nii.gz)
    output_dir : str or Path, optional
        Directory to save output files
    method : str, optional
        Harmonization method ('combat', 'phantom', 'mri')
    apply_pvc : bool, optional
        Whether to apply partial volume correction
    """
    logger.info(f"Testing harmonization with real data using {method} method")
    logger.info(f"PET image: {pet_path}")
    logger.info(f"MRI image: {mri_path}")
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load images
    try:
        pet_img = ants.image_read(str(pet_path))
        mri_img = ants.image_read(str(mri_path))
        
        logger.info(f"PET image loaded: shape={pet_img.shape}, spacing={pet_img.spacing}")
        logger.info(f"MRI image loaded: shape={mri_img.shape}, spacing={mri_img.spacing}")
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        return
    
    # Initialize harmonizer
    harmonizer = ImageHarmonization(method=method, apply_pvc=apply_pvc, verbose=True)
    
    # Calculate initial EIR
    initial_eir = harmonizer.calculate_eir(pet_img)
    logger.info(f"Initial PET EIR: {initial_eir:.2f}mm")
    
    # Extract MRI features if using MRI method
    metadata = {}
    if method == 'mri':
        logger.info("Extracting MRI features")
        mri_features = harmonizer.extract_mri_features(mri_img)
        logger.info(f"MRI features: {mri_features}")
        metadata['mri_features'] = mri_features
    
    # Apply harmonization
    logger.info("Applying harmonization")
    harmonized_img = harmonizer.harmonize(pet_img, metadata)
    
    # Calculate final EIR
    final_eir = harmonizer.calculate_eir(harmonized_img)
    logger.info(f"Final PET EIR: {final_eir:.2f}mm")
    
    # Save results if output directory is provided
    if output_dir:
        # Save harmonized image
        output_pet_path = output_dir / f"harmonized_{method}_pet.nii.gz"
        ants.image_write(harmonized_img, str(output_pet_path))
        logger.info(f"Harmonized PET saved to: {output_pet_path}")
        
        # Save original PET for comparison
        orig_pet_path = output_dir / "original_pet.nii.gz"
        ants.image_write(pet_img, str(orig_pet_path))
        logger.info(f"Original PET saved to: {orig_pet_path}")
        
        # Save MRI for reference
        mri_path_out = output_dir / "mri_t1w.nii.gz"
        ants.image_write(mri_img, str(mri_path_out))
        logger.info(f"MRI saved to: {mri_path_out}")
    
    # Calculate and report statistics
    pet_data = pet_img.numpy()
    harmonized_data = harmonized_img.numpy()
    
    # Create mask for brain tissue
    mask = pet_data > 0
    
    if np.sum(mask) > 0:
        # Calculate statistics
        orig_mean = np.mean(pet_data[mask])
        orig_std = np.std(pet_data[mask])
        harm_mean = np.mean(harmonized_data[mask])
        harm_std = np.std(harmonized_data[mask])
        
        logger.info(f"Original PET statistics: mean={orig_mean:.4f}, std={orig_std:.4f}")
        logger.info(f"Harmonized PET statistics: mean={harm_mean:.4f}, std={harm_std:.4f}")
        
        # Calculate correlation
        correlation = np.corrcoef(pet_data[mask], harmonized_data[mask])[0, 1]
        logger.info(f"Correlation between original and harmonized: {correlation:.4f}")
    
    logger.info("Harmonization test completed successfully")
    return harmonized_img


def test_batch_harmonization(pet_dir, mri_dir, output_dir=None, method='mri', apply_pvc=False):
    """
    Test batch harmonization with multiple PET and MRI images.
    
    Parameters
    ----------
    pet_dir : str or Path
        Directory containing PET images (.nii.gz)
    mri_dir : str or Path
        Directory containing paired MRI T1w images (.nii.gz)
    output_dir : str or Path, optional
        Directory to save output files
    method : str, optional
        Harmonization method ('combat', 'phantom', 'mri')
    apply_pvc : bool, optional
        Whether to apply partial volume correction
    """
    logger.info(f"Testing batch harmonization using {method} method")
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all PET and MRI files
    pet_dir = Path(pet_dir)
    mri_dir = Path(mri_dir)
    
    pet_files = list(pet_dir.glob("*.nii.gz"))
    mri_files = list(mri_dir.glob("*.nii.gz"))
    
    logger.info(f"Found {len(pet_files)} PET files and {len(mri_files)} MRI files")
    
    # Match PET and MRI files by name
    paired_files = []
    
    for pet_file in pet_files:
        pet_stem = pet_file.stem.replace(".nii", "")
        
        # Try to find matching MRI file
        for mri_file in mri_files:
            mri_stem = mri_file.stem.replace(".nii", "")
            
            # Check if stems match or if MRI file contains PET file stem
            if pet_stem == mri_stem or pet_stem in mri_stem or mri_stem in pet_stem:
                paired_files.append((pet_file, mri_file))
                break
    
    logger.info(f"Found {len(paired_files)} paired PET-MRI files")
    
    if not paired_files:
        logger.error("No paired PET-MRI files found")
        return
    
    # Initialize harmonizer
    harmonizer = ImageHarmonization(method=method, apply_pvc=apply_pvc, verbose=True)
    
    # Process each pair
    for i, (pet_file, mri_file) in enumerate(paired_files):
        logger.info(f"Processing pair {i+1}/{len(paired_files)}")
        logger.info(f"PET: {pet_file.name}, MRI: {mri_file.name}")
        
        # Create subject-specific output directory
        if output_dir:
            subj_output_dir = output_dir / f"subject_{i+1}"
            subj_output_dir.mkdir(exist_ok=True, parents=True)
        else:
            subj_output_dir = None
        
        # Process the pair
        try:
            test_real_data_harmonization(
                pet_path=pet_file,
                mri_path=mri_file,
                output_dir=subj_output_dir,
                method=method,
                apply_pvc=apply_pvc
            )
        except Exception as e:
            logger.error(f"Error processing pair {i+1}: {e}")
    
    logger.info("Batch harmonization test completed")


def test_model_training_and_application(pet_dir, mri_dir, output_dir=None, method='mri'):
    """
    Test model training and application with multiple PET and MRI images.
    
    Parameters
    ----------
    pet_dir : str or Path
        Directory containing PET images (.nii.gz)
    mri_dir : str or Path
        Directory containing paired MRI T1w images (.nii.gz)
    output_dir : str or Path, optional
        Directory to save output files
    method : str, optional
        Harmonization method ('combat', 'phantom', 'mri')
    """
    logger.info(f"Testing model training and application using {method} method")
    
    # Create output directory if needed
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        model_dir = output_dir / "models"
        model_dir.mkdir(exist_ok=True, parents=True)
    else:
        model_dir = Path("./models")
        model_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all PET and MRI files
    pet_dir = Path(pet_dir)
    mri_dir = Path(mri_dir)
    
    pet_files = list(pet_dir.glob("*.nii.gz"))
    mri_files = list(mri_dir.glob("*.nii.gz"))
    
    # Match PET and MRI files by name
    paired_files = []
    
    for pet_file in pet_files:
        pet_stem = pet_file.stem.replace(".nii", "")
        
        # Try to find matching MRI file
        for mri_file in mri_files:
            mri_stem = mri_file.stem.replace(".nii", "")
            
            # Check if stems match or if MRI file contains PET file stem
            if pet_stem == mri_stem or pet_stem in mri_stem or mri_stem in pet_stem:
                paired_files.append((pet_file, mri_file))
                break
    
    logger.info(f"Found {len(paired_files)} paired PET-MRI files")
    
    if len(paired_files) < 2:
        logger.error("Need at least 2 paired PET-MRI files for model training")
        return
    
    # Split into training and testing sets
    train_pairs = paired_files[:-1]  # Use all but the last pair for training
    test_pair = paired_files[-1]     # Use the last pair for testing
    
    logger.info(f"Using {len(train_pairs)} pairs for training and 1 pair for testing")
    
    # Initialize harmonizer
    harmonizer = ImageHarmonization(method=method, verbose=True)
    
    # Load training images and extract features
    pet_images = []
    mri_features = []
    batches = []
    
    for i, (pet_file, mri_file) in enumerate(train_pairs):
        try:
            # Load images
            pet_img = ants.image_read(str(pet_file))
            mri_img = ants.image_read(str(mri_file))
            
            # Extract MRI features if using MRI method
            if method == 'mri':
                features = harmonizer.extract_mri_features(mri_img)
                mri_features.append(features)
            
            # Add to lists
            pet_images.append(pet_img)
            batches.append(f"batch_{i}")
            
            logger.info(f"Processed training pair {i+1}: {pet_file.name}")
            
        except Exception as e:
            logger.error(f"Error processing training pair {i+1}: {e}")
    
    # Train model
    model_path = model_dir / f"{method}_model.pkl"
    
    if method == 'phantom':
        model = harmonizer.train_phantom_model(
            phantom_images=pet_images,
            target_eir=8.0,
            output_model_path=model_path
        )
    elif method == 'mri':
        model = harmonizer.train_mri_model(
            pet_images=pet_images,
            mri_features=mri_features,
            target_eir=8.0,
            output_model_path=model_path
        )
    elif method == 'combat':
        model = harmonizer.train_combat_model(
            images=pet_images,
            batches=batches,
            target_eir=8.0,
            output_model_path=model_path
        )
    
    logger.info(f"Model trained and saved to {model_path}")
    
    # Test model on test pair
    test_pet_file, test_mri_file = test_pair
    
    # Load test images
    test_pet_img = ants.image_read(str(test_pet_file))
    test_mri_img = ants.image_read(str(test_mri_file))
    
    # Create new harmonizer with trained model
    test_harmonizer = ImageHarmonization(
        method=method,
        model_path=model_path,
        verbose=True
    )
    
    # Prepare metadata
    metadata = {}
    
    if method == 'mri':
        # Extract MRI features
        test_features = test_harmonizer.extract_mri_features(test_mri_img)
        metadata['mri_features'] = test_features
    elif method == 'combat':
        # Use a batch identifier
        metadata['batch'] = 'batch_0'  # Use first batch for testing
    
    # Apply harmonization
    harmonized_img = test_harmonizer.harmonize(test_pet_img, metadata)
    
    # Save results
    if output_dir:
        # Save harmonized image
        output_pet_path = output_dir / f"test_harmonized_{method}_pet.nii.gz"
        ants.image_write(harmonized_img, str(output_pet_path))
        logger.info(f"Harmonized test PET saved to: {output_pet_path}")
        
        # Save original PET for comparison
        orig_pet_path = output_dir / "test_original_pet.nii.gz"
        ants.image_write(test_pet_img, str(orig_pet_path))
        logger.info(f"Original test PET saved to: {orig_pet_path}")
    
    logger.info("Model training and application test completed")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test PET image harmonization")
    
    parser.add_argument("--pet", type=str, help="Path to PET image (.nii.gz) or directory of PET images")
    parser.add_argument("--mri", type=str, help="Path to MRI T1w image (.nii.gz) or directory of MRI images")
    parser.add_argument("--output", type=str, default="./output", help="Output directory for harmonized images")
    parser.add_argument("--method", type=str, default="mri", choices=["combat", "phantom", "mri"], 
                        help="Harmonization method to use")
    parser.add_argument("--pvc", action="store_true", help="Apply partial volume correction")
    parser.add_argument("--batch", action="store_true", help="Process multiple images in batch mode")
    parser.add_argument("--train", action="store_true", help="Train a harmonization model")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine mode of operation
    if args.pet and args.mri:
        pet_path = Path(args.pet)
        mri_path = Path(args.mri)
        
        if pet_path.is_file() and mri_path.is_file():
            # Single file mode
            logger.info("Running in single file mode")
            test_real_data_harmonization(
                pet_path=pet_path,
                mri_path=mri_path,
                output_dir=output_dir,
                method=args.method,
                apply_pvc=args.pvc
            )
        elif pet_path.is_dir() and mri_path.is_dir():
            # Directory mode
            logger.info("Running in directory mode")
            
            if args.train:
                # Train and apply model
                logger.info("Training and applying harmonization model")
                test_model_training_and_application(
                    pet_dir=pet_path,
                    mri_dir=mri_path,
                    output_dir=output_dir,
                    method=args.method
                )
            elif args.batch:
                # Batch processing
                logger.info("Processing in batch mode")
                test_batch_harmonization(
                    pet_dir=pet_path,
                    mri_dir=mri_path,
                    output_dir=output_dir,
                    method=args.method,
                    apply_pvc=args.pvc
                )
            else:
                logger.error("For directory inputs, either --batch or --train must be specified")
        else:
            logger.error("Both --pet and --mri must be either files or directories")
    else:
        logger.error("Both --pet and --mri arguments are required")
        parser.print_help()

    # Example usage:
    # Single file mode:
    # python test_harmonization.py --pet subject1_pet.nii.gz --mri subject1_t1w.nii.gz --method mri --output ./output
    
    # Batch mode:
    # python test_harmonization.py --pet ./pet_data --mri ./mri_data --batch --method mri --output ./output
    
    # Train and apply model:
    # python test_harmonization.py --pet ./pet_data --mri ./mri_data --train --method mri --output ./output