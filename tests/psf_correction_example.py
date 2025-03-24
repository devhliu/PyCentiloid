"""
Example script demonstrating PSF estimation and correction using PyCentiloid.
"""

import os
import argparse
from PyCentiloid.psf import apply_psf_correction

def main():
    parser = argparse.ArgumentParser(description='Apply PSF correction to PET image')
    parser.add_argument('--pet', required=True, help='Path to PET image')
    parser.add_argument('--mri', required=True, help='Path to T1W MRI image')
    parser.add_argument('--mask', help='Path to brain mask (optional)')
    parser.add_argument('--output', required=True, help='Path to save deconvolved PET image')
    parser.add_argument('--method', default='richardson_lucy', 
                        choices=['richardson_lucy', 'wiener'],
                        help='Deconvolution method')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations for Richardson-Lucy deconvolution')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Apply PSF correction
    results = apply_psf_correction(
        pet_path=args.pet,
        mri_path=args.mri,
        output_path=args.output,
        mask_path=args.mask,
        method=args.method,
        iterations=args.iterations
    )
    
    # Print results
    print("PSF Correction Results:")
    print(f"FWHM (x, y, z): {results['fwhm_x']:.2f}, {results['fwhm_y']:.2f}, {results['fwhm_z']:.2f} mm")
    print(f"Deconvolved image saved to: {results['output_path']}")

if __name__ == '__main__':
    main()