from pycentiloid.preprocessing.tracer_pipeline import TracerPipeline

# Initialize pipeline for specific tracer
pipeline = TracerPipeline(tracer_type='PIB')

# Process single PET image
result = pipeline.process_single_pet('subject1_pib_pet.nii.gz')

# Process multiple images with harmonization
pet_files = ['sub1_pib.nii.gz', 'sub2_pib.nii.gz']
scanners = ['scanner1', 'scanner2']
results = pipeline.process_batch(pet_files, scanners)



# Example of creating a template

from pycentiloid.preprocessing.tracer_pipeline import TracerPipeline

# Initialize pipeline for specific tracer
pipeline = TracerPipeline(tracer_type='PIB')

# List of PET images for template creation
pet_images = [
    'd:/data/sub1_pet.nii.gz',
    'd:/data/sub2_pet.nii.gz',
    'd:/data/sub3_pet.nii.gz',
    # ... more images ...
]

# Create tracer-specific template
template_path = pipeline.create_tracer_template(
    pet_images,
    output_dir='d:/templates/PIB',
    iterations=3
)

# Process new images using the created template
result = pipeline.process_single_pet('new_subject_pet.nii.gz')

# 
from pycentiloid.preprocessing.template_generation import TemplateBuilder

# Create PET template
pet_builder = TemplateBuilder(modality='PET')
pet_template = pet_builder.create_template(
    pet_images,
    output_dir='d:/templates/PET',
    iterations=3
)

# Create T1 MRI template
t1_builder = TemplateBuilder(modality='T1')
t1_template = t1_builder.create_template(
    t1_images,
    output_dir='d:/templates/T1',
    iterations=3
)

# Create CT template
ct_builder = TemplateBuilder(modality='CT')
ct_template = ct_builder.create_template(
    ct_images,
    output_dir='d:/templates/CT',
    iterations=3
)