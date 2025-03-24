## Processing

### spatial normalization (registration.py)
- antspyx
- deep learning based spatial normalization using registration foundation model uniGradICON (https://github.com/uncbiag/uniGradICON)
- capable of spatial normalization Amyloid PET, CT and MRI into its corresponding template in standard space (mni space)
- implemented 1mm and 2mm istropic resolution
- registration should be done in 1 minutes in a standard PC
- rigid and deformative registration

### segmentation (segmentation.py)
- deep learning based segmentation using brainchop for mri t1w images (https://github.com/neuroneural/brainchop)
- brain extraction (BET) for Amyloid PET, CT and MRI images
- templated based segmentation optimization

### harmonization (harmonization.py)
- hoffman phantom based harmonization using EIR
- paired MRI based harmonization using EIR
- Combat harmonization
- Amyloid PET enhancement using partial volume correction

## Atlas

### atlas management (manager.py)
- atlas I/O
- atlas calling API
- atlas visualization

### atlas
- the atlas templates and masks are stored in the atlas folder
- the atlas reading and writing is managed by manager.py (combine existing functions defined in manager.py and template.py, then remove template.py)
- two types of spatial resolutions (1mm and 2mm)
- template and mask covering various AD stage of amyloid PET, CT and MRI T1W


## Centiloid and SUVr

- centiloid score calculation
- centiloid calibration
- centiloid configuration for different sites and tracers, and its I/O API
- tracer including C11-PIB, F18-FBP, F18-FBB, F18-FMM, F18-FBZ, and extended to new tracers

- SUVr calculation based anatomic masks, and grouped them to larger brain regions

## Reporting

- report generation based on the atlas and centiloid score
- report is generated from a predefined docx template, and this template is configured to be site adaptive
- report I/O API for generation and reporting
- report to pdf, DICOM SR
- the report template include [Patient Information, Injection and Imaging Information, Comments, Centiloid Score, SUVr Table, Snap View of Amyloid PET in private and standard speces]
- the report include site information in header and end.

## Workflow

- centiloid scoring workflow
- template generation worklflow
- calibration workflow

## Utils
- various tools
- DICOM 
- NIFTI <> DICOM
