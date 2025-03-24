"""
Template management module for PyCentiloid.

This module provides classes and functions for managing templates used in
spatial normalization and reference for Centiloid calculation.
"""

import os
import json
import ants
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

class Template:
    """
    Class representing an imaging template.
    
    Stores information about a template, including its path, modality,
    and associated metadata.
    """
    
    def __init__(self, 
                 template_id: str,
                 name: str,
                 modality: str,
                 path: Union[str, Path],
                 resolution: str,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize template.
        
        Parameters
        ----------
        template_id : str
            Unique identifier for the template
        name : str
            Descriptive name of the template
        modality : str
            Imaging modality ('PET', 'MRI', 'CT')
        path : str or Path
            Path to the template image
        resolution : str
            Resolution of the template ('1mm', '2mm')
        metadata : dict, optional
            Additional metadata
        """
        self.template_id = template_id
        self.name = name
        self.modality = modality
        self.path = Path(path)
        self.resolution = resolution
        self.metadata = metadata or {}
        
        # Validate template
        self._validate()
    
    def _validate(self):
        """Validate template."""
        if not self.path.exists():
            raise FileNotFoundError(f"Template file not found: {self.path}")
    
    def load(self) -> ants.ANTsImage:
        """
        Load template image.
        
        Returns
        -------
        ants.ANTsImage
            Template image
        """
        return ants.image_read(str(self.path))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert template to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the template
        """
        return {
            'template_id': self.template_id,
            'name': self.name,
            'modality': self.modality,
            'path': str(self.path),
            'resolution': self.resolution,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Template':
        """
        Create template from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary representation of the template
            
        Returns
        -------
        Template
            Template object
        """
        return cls(
            template_id=data['template_id'],
            name=data['name'],
            modality=data['modality'],
            path=data['path'],
            resolution=data['resolution'],
            metadata=data.get('metadata', {})
        )

class TemplateRegistry:
    """
    Registry for templates.
    
    Manages a collection of templates and provides methods for accessing them.
    """
    
    _templates = {}
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Initialize template registry with standard templates."""
        if cls._initialized:
            return
        
        # Get package directory
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        templates_dir = package_dir / 'data' / 'templates'
        
        # Define standard templates
        standard_templates = [
            {
                'template_id': 'mni_t1_1mm',
                'name': 'MNI T1 1mm',
                'modality': 'MRI',
                'path': templates_dir / 'mni' / '1mm' / 'mni_t1_1mm.nii.gz',
                'resolution': '1mm',
                'metadata': {
                    'space': 'MNI',
                    'description': 'MNI T1 template at 1mm resolution'
                }
            },
            {
                'template_id': 'mni_t1_2mm',
                'name': 'MNI T1 2mm',
                'modality': 'MRI',
                'path': templates_dir / 'mni' / '2mm' / 'mni_t1_2mm.nii.gz',
                'resolution': '2mm',
                'metadata': {
                    'space': 'MNI',
                    'description': 'MNI T1 template at 2mm resolution'
                }
            },
            {
                'template_id': 'pib_1mm',
                'name': 'PIB PET 1mm',
                'modality': 'PET',
                'path': templates_dir / 'pet' / '1mm' / 'pib_template_1mm.nii.gz',
                'resolution': '1mm',
                'metadata': {
                    'space': 'MNI',
                    'tracer': 'PIB',
                    'description': 'PIB PET template at 1mm resolution'
                }
            },
            {
                'template_id': 'pib_2mm',
                'name': 'PIB PET 2mm',
                'modality': 'PET',
                'path': templates_dir / 'pet' / '2mm' / 'pib_template_2mm.nii.gz',
                'resolution': '2mm',
                'metadata': {
                    'space': 'MNI',
                    'tracer': 'PIB',
                    'description': 'PIB PET template at 2mm resolution'
                }
            }
        ]
        
        # Register standard templates
        for template_data in standard_templates:
            try:
                template = Template(**template_data)
                cls.register_template(template)
            except FileNotFoundError:
                # Skip if template file not found
                print(f"Warning: Template file not found: {template_data['path']}")
        
        # Load custom templates
        cls._load_custom_templates()
        
        cls._initialized = True
    
    @classmethod
    def _load_custom_templates(cls):
        """Load custom templates from configuration file."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = package_dir / 'config'
        
        custom_templates_path = config_dir / 'custom_templates.json'
        if custom_templates_path.exists():
            try:
                with open(custom_templates_path, 'r') as f:
                    custom_templates = json.load(f)
                
                for template_data in custom_templates:
                    try:
                        template = Template.from_dict(template_data)
                        cls.register_template(template)
                    except FileNotFoundError:
                        # Skip if template file not found
                        print(f"Warning: Custom template file not found: {template_data['path']}")
            except Exception as e:
                print(f"Error loading custom templates: {e}")
    
    @classmethod
    def register_template(cls, template: Template):
        """
        Register a template.
        
        Parameters
        ----------
        template : Template
            Template to register
        """
        cls._templates[template.template_id] = template
    
    @classmethod
    def get_template(cls, template_id: str) -> Template:
        """
        Get a template by ID.
        
        Parameters
        ----------
        template_id : str
            Template ID
            
        Returns
        -------
        Template
            Template object
        """
        if not cls._initialized:
            cls.initialize()
        
        if template_id not in cls._templates:
            raise ValueError(f"Template not found: {template_id}")
        
        return cls._templates[template_id]
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """
        List all registered templates.
        
        Returns
        -------
        list
            List of template IDs
        """
        if not cls._initialized:
            cls.initialize()
        
        return list(cls._templates.keys())
    
    @classmethod
    def get_templates_by_modality(cls, modality: str) -> List[Template]:
        """
        Get templates by modality.
        
        Parameters
        ----------
        modality : str
            Modality ('PET', 'MRI', 'CT')
            
        Returns
        -------
        list
            List of templates with the specified modality
        """
        if not cls._initialized:
            cls.initialize()
        
        return [
            template for template in cls._templates.values()
            if template.modality == modality
        ]
    
    @classmethod
    def get_templates_by_resolution(cls, resolution: str) -> List[Template]:
        """
        Get templates by resolution.
        
        Parameters
        ----------
        resolution : str
            Resolution ('1mm', '2mm')
            
        Returns
        -------
        list
            List of templates with the specified resolution
        """
        if not cls._initialized:
            cls.initialize()
        
        return [
            template for template in cls._templates.values()
            if template.resolution == resolution
        ]
    @classmethod
    def add_custom_template(cls, 
                           template_id: str,
                           name: str,
                           modality: str,
                           path: Union[str, Path],
                           resolution: str,
                           metadata: Optional[Dict[str, Any]] = None) -> Template:
        """
        Add a custom template.
        
        Parameters
        ----------
        template_id : str
            Unique identifier for the template
        name : str
            Descriptive name of the template
        modality : str
            Imaging modality ('PET', 'MRI', 'CT')
        path : str or Path
            Path to the template image
        resolution : str
            Resolution of the template ('1mm', '2mm')
        metadata : dict, optional
            Additional metadata
            
        Returns
        -------
        Template
            Created template
        """
        if not cls._initialized:
            cls.initialize()
        
        if template_id in cls._templates:
            raise ValueError(f"Template ID already exists: {template_id}")
        
        template = Template(
            template_id=template_id,
            name=name,
            modality=modality,
            path=path,
            resolution=resolution,
            metadata=metadata
        )
        
        cls.register_template(template)
        
        # Save custom templates to configuration
        cls._save_custom_templates()
        
        return template
    
    @classmethod
    def _save_custom_templates(cls):
        """Save custom templates to configuration file."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = package_dir / 'config'
        config_dir.mkdir(exist_ok=True, parents=True)
        
        custom_templates_path = config_dir / 'custom_templates.json'
        
        # Get standard template IDs
        standard_ids = {'mni_t1_1mm', 'mni_t1_2mm', 'pib_1mm', 'pib_2mm'}
        
        # Filter custom templates
        custom_templates = [
            template.to_dict() for template_id, template in cls._templates.items()
            if template_id not in standard_ids
        ]
        
        # Save to file
        with open(custom_templates_path, 'w') as f:
            json.dump(custom_templates, f, indent=2)
    
    @classmethod
    def remove_template(cls, template_id: str):
        """
        Remove a template from the registry.
        
        Parameters
        ----------
        template_id : str
            Template ID
        """
        if not cls._initialized:
            cls.initialize()
        
        if template_id not in cls._templates:
            raise ValueError(f"Template not found: {template_id}")
        
        # Don't allow removing standard templates
        standard_ids = {'mni_t1_1mm', 'mni_t1_2mm', 'pib_1mm', 'pib_2mm'}
        if template_id in standard_ids:
            raise ValueError(f"Cannot remove standard template: {template_id}")
        
        del cls._templates[template_id]
        
        # Save custom templates to configuration
        cls._save_custom_templates()
    
    @classmethod
    def get_templates_by_tracer(cls, tracer_id: str) -> List[Template]:
        """
        Get templates for a specific tracer.
        
        Parameters
        ----------
        tracer_id : str
            Tracer ID
            
        Returns
        -------
        list
            List of templates for the specified tracer
        """
        if not cls._initialized:
            cls.initialize()
        
        return [
            template for template in cls._templates.values()
            if template.modality == 'PET' and 
            template.metadata.get('tracer') == tracer_id
        ]
    
    @classmethod
    def get_template_by_name(cls, name: str) -> Optional[Template]:
        """
        Get a template by name.
        
        Parameters
        ----------
        name : str
            Template name
            
        Returns
        -------
        Template or None
            Template object if found, None otherwise
        """
        if not cls._initialized:
            cls.initialize()
        
        for template in cls._templates.values():
            if template.name == name:
                return template
        
        return None
    
    @classmethod
    def get_default_template(cls, modality: str, resolution: str) -> Template:
        """
        Get default template for a modality and resolution.
        
        Parameters
        ----------
        modality : str
            Modality ('PET', 'MRI', 'CT')
        resolution : str
            Resolution ('1mm', '2mm')
            
        Returns
        -------
        Template
            Default template
        """
        if not cls._initialized:
            cls.initialize()
        
        # Default templates by modality and resolution
        defaults = {
            ('MRI', '1mm'): 'mni_t1_1mm',
            ('MRI', '2mm'): 'mni_t1_2mm',
            ('PET', '1mm'): 'pib_1mm',
            ('PET', '2mm'): 'pib_2mm'
        }
        
        template_id = defaults.get((modality, resolution))
        if template_id and template_id in cls._templates:
            return cls._templates[template_id]
        
        # If no default found, return first matching template
        matching_templates = [
            template for template in cls._templates.values()
            if template.modality == modality and template.resolution == resolution
        ]
        
        if matching_templates:
            return matching_templates[0]
        
        raise ValueError(f"No template found for modality '{modality}' and resolution '{resolution}'")