# Example of how to use the calibration module
calibrator = CentiloidCalibration()

# Paths to paired images (same subjects scanned with both tracers)
new_tracer_scans = ['subject1_new.nii.gz', 'subject2_new.nii.gz']
pib_scans = ['subject1_pib.nii.gz', 'subject2_pib.nii.gz']

# Perform calibration
results = calibrator.calibrate_tracer(
    new_tracer_scans,
    pib_scans,
    ctx_mask_path='path/to/ctx_mask.nii.gz',
    cb_mask_path='path/to/cb_mask.nii.gz'
)

# Convert a new SUVR to Centiloid
suvr_value = 1.5
centiloid_value = calibrator.convert_to_centiloid(suvr_value)

# Save calibration parameters
calibrator.save_calibration('calibration_parameters.csv')