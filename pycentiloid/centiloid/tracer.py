"""
Tracer definitions and parameters for PyCentiloid.

This module provides classes and functions for managing different PET tracers
used in amyloid imaging, including their properties and reference values.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

class Tracer:
    """
    Class representing a PET tracer.
    
    Stores information about a specific PET tracer, including its properties,
    reference values, and calibration data.
    """
    
    def __init__(self, 
                 tracer_id: str,
                 name: str,
                 half_life: float,
                 reference_region: str,
                 properties: Optional[Dict[str, Any]] = None):
        """
        Initialize tracer.
        
        Parameters
        ----------
        tracer_id : str
            Unique identifier for the tracer
        name : str
            Full name of the tracer
        half_life : float
            Half-life in minutes
        reference_region : str
            Standard reference region for SUVR calculation
        properties : dict, optional
            Additional properties
        """
        self.tracer_id = tracer_id
        self.name = name
        self.half_life = half_life
        self.reference_region = reference_region
        self.properties = properties or {}
        
        # Set default values
        self.calibration_data = {}
        self.site_corrections = {}
    
    def set_calibration_data(self, calibration_data: Dict[str, Any]):
        """
        Set calibration data for the tracer.
        
        Parameters
        ----------
        calibration_data : dict
            Calibration data
        """
        self.calibration_data = calibration_data
    
    def add_site_correction(self, site_id: str, correction_factor: float):
        """
        Add site-specific correction factor.
        
        Parameters
        ----------
        site_id : str
            Site identifier
        correction_factor : float
            Correction factor for the site
        """
        self.site_corrections[site_id] = correction_factor
    
    def get_site_correction(self, site_id: str) -> float:
        """
        Get site-specific correction factor.
        
        Parameters
        ----------
        site_id : str
            Site identifier
            
        Returns
        -------
        float
            Correction factor for the site
        """
        return self.site_corrections.get(site_id, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tracer to dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the tracer
        """
        return {
            'tracer_id': self.tracer_id,
            'name': self.name,
            'half_life': self.half_life,
            'reference_region': self.reference_region,
            'properties': self.properties,
            'calibration_data': self.calibration_data,
            'site_corrections': self.site_corrections
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tracer':
        """
        Create tracer from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary representation of the tracer
            
        Returns
        -------
        Tracer
            Tracer object
        """
        tracer = cls(
            tracer_id=data['tracer_id'],
            name=data['name'],
            half_life=data['half_life'],
            reference_region=data['reference_region'],
            properties=data.get('properties', {})
        )
        
        tracer.calibration_data = data.get('calibration_data', {})
        tracer.site_corrections = data.get('site_corrections', {})
        
        return tracer

class TracerRegistry:
    """
    Registry for PET tracers.
    
    Manages a collection of tracers and provides methods for accessing them.
    """
    
    _tracers = {}
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Initialize tracer registry with standard tracers."""
        if cls._initialized:
            return
        
        # Define standard tracers
        cls.register_tracer(Tracer(
            tracer_id='PIB',
            name='[11C]Pittsburgh compound B',
            half_life=20.4,
            reference_region='whole_cerebellum',
            properties={
                'isotope': 'C11',
                'target': 'amyloid',
                'full_name': '[11C]Pittsburgh compound B',
                'short_name': 'PiB'
            }
        ))
        
        cls.register_tracer(Tracer(
            tracer_id='FBB',
            name='[18F]Florbetaben',
            half_life=109.8,
            reference_region='cerebellar_cortex',
            properties={
                'isotope': 'F18',
                'target': 'amyloid',
                'full_name': '[18F]Florbetaben',
                'short_name': 'FBB',
                'trade_name': 'NeuraCeq'
            }
        ))
        
        cls.register_tracer(Tracer(
            tracer_id='FBP',
            name='[18F]Florbetapir',
            half_life=109.8,
            reference_region='cerebellar_white_matter',
            properties={
                'isotope': 'F18',
                'target': 'amyloid',
                'full_name': '[18F]Florbetapir',
                'short_name': 'FBP',
                'trade_name': 'Amyvid'
            }
        ))
        
        cls.register_tracer(Tracer(
            tracer_id='FMM',
            name='[18F]Flutemetamol',
            half_life=109.8,
            reference_region='whole_cerebellum',
            properties={
                'isotope': 'F18',
                'target': 'amyloid',
                'full_name': '[18F]Flutemetamol',
                'short_name': 'FMM',
                'trade_name': 'Vizamyl'
            }
        ))
        
        cls.register_tracer(Tracer(
            tracer_id='FBZ',
            name='[18F]Florbenzazine',
            half_life=109.8,
            reference_region='pons',
            properties={
                'isotope': 'F18',
                'target': 'amyloid',
                'full_name': '[18F]Florbenzazine',
                'short_name': 'FBZ'
            }
        ))
        
        # Load custom tracers from configuration
        cls._load_custom_tracers()
        
        cls._initialized = True
    
    @classmethod
    def _load_custom_tracers(cls):
        """Load custom tracers from configuration file."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = package_dir / 'config'
        
        custom_tracers_path = config_dir / 'custom_tracers.json'
        if custom_tracers_path.exists():
            try:
                with open(custom_tracers_path, 'r') as f:
                    custom_tracers = json.load(f)
                
                for tracer_data in custom_tracers:
                    tracer = Tracer.from_dict(tracer_data)
                    cls.register_tracer(tracer)
            except Exception as e:
                print(f"Error loading custom tracers: {e}")
    
    @classmethod
    def register_tracer(cls, tracer: Tracer):
        """
        Register a tracer.
        
        Parameters
        ----------
        tracer : Tracer
            Tracer to register
        """
        cls._tracers[tracer.tracer_id] = tracer
    
    @classmethod
    def get_tracer(cls, tracer_id: str) -> Tracer:
        """
        Get a tracer by ID.
        
        Parameters
        ----------
        tracer_id : str
            Tracer ID
            
        Returns
        -------
        Tracer
            Tracer object
        """
        if not cls._initialized:
            cls.initialize()
        
        if tracer_id not in cls._tracers:
            raise ValueError(f"Tracer not found: {tracer_id}")
        
        return cls._tracers[tracer_id]
    
    @classmethod
    def list_tracers(cls) -> List[str]:
        """
        List all registered tracers.
        
        Returns
        -------
        list
            List of tracer IDs
        """
        if not cls._initialized:
            cls.initialize()
        
        return list(cls._tracers.keys())
    
    @classmethod
    def save_tracers(cls, output_path: Union[str, Path]):
        """
        Save all tracers to a file.
        
        Parameters
        ----------
        output_path : str or Path
            Path to save tracers
        """
        if not cls._initialized:
            cls.initialize()
        
        tracers_data = [tracer.to_dict() for tracer in cls._tracers.values()]
        
        with open(output_path, 'w') as f:
            json.dump(tracers_data, f, indent=2)
    
    @classmethod
    def load_tracers(cls, input_path: Union[str, Path]):
        """
        Load tracers from a file.
        
        Parameters
        ----------
        input_path : str or Path
            Path to load tracers from
        """
        with open(input_path, 'r') as f:
            tracers_data = json.load(f)
        
        for tracer_data in tracers_data:
            tracer = Tracer.from_dict(tracer_data)
            cls.register_tracer(tracer)
    
    @classmethod
    def get_tracer_by_name(cls, name: str) -> Optional[Tracer]:
        """
        Get a tracer by name.
        
        Parameters
        ----------
        name : str
            Tracer name
            
        Returns
        -------
        Tracer or None
            Tracer object if found, None otherwise
        """
        if not cls._initialized:
            cls.initialize()
        
        for tracer in cls._tracers.values():
            if tracer.name == name or name in tracer.properties.values():
                return tracer
        
        return None
    
    @classmethod
    def get_tracers_by_isotope(cls, isotope: str) -> List[Tracer]:
        """
        Get tracers by isotope.
        
        Parameters
        ----------
        isotope : str
            Isotope (e.g., 'F18', 'C11')
            
        Returns
        -------
        list
            List of tracers with the specified isotope
        """
        if not cls._initialized:
            cls.initialize()
        
        return [
            tracer for tracer in cls._tracers.values()
            if tracer.properties.get('isotope') == isotope
        ]
    
    @classmethod
    def create_custom_tracer(cls, 
                            tracer_id: str,
                            name: str,
                            half_life: float,
                            reference_region: str,
                            properties: Optional[Dict[str, Any]] = None) -> Tracer:
        """
        Create and register a custom tracer.
        
        Parameters
        ----------
        tracer_id : str
            Unique identifier for the tracer
        name : str
            Full name of the tracer
        half_life : float
            Half-life in minutes
        reference_region : str
            Standard reference region for SUVR calculation
        properties : dict, optional
            Additional properties
            
        Returns
        -------
        Tracer
            Created tracer
        """
        if not cls._initialized:
            cls.initialize()
        
        if tracer_id in cls._tracers:
            raise ValueError(f"Tracer ID already exists: {tracer_id}")
        
        tracer = Tracer(
            tracer_id=tracer_id,
            name=name,
            half_life=half_life,
            reference_region=reference_region,
            properties=properties
        )
        
        cls.register_tracer(tracer)
        
        # Save custom tracers to configuration
        cls._save_custom_tracers()
        
        return tracer
    
    @classmethod
    def _save_custom_tracers(cls):
        """Save custom tracers to configuration file."""
        package_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = package_dir / 'config'
        config_dir.mkdir(exist_ok=True, parents=True)
        
        custom_tracers_path = config_dir / 'custom_tracers.json'
        
        # Get standard tracer IDs
        standard_ids = {'PIB', 'FBB', 'FBP', 'FMM', 'FBZ'}
        
        # Filter custom tracers
        custom_tracers = [
            tracer.to_dict() for tracer_id, tracer in cls._tracers.items()
            if tracer_id not in standard_ids
        ]
        
        # Save to file
        with open(custom_tracers_path, 'w') as f:
            json.dump(custom_tracers, f, indent=2)
    
    @classmethod
    def remove_tracer(cls, tracer_id: str):
        """
        Remove a tracer from the registry.
        
        Parameters
        ----------
        tracer_id : str
            Tracer ID
        """
        if not cls._initialized:
            cls.initialize()
        
        if tracer_id not in cls._tracers:
            raise ValueError(f"Tracer not found: {tracer_id}")
        
        # Don't allow removing standard tracers
        standard_ids = {'PIB', 'FBB', 'FBP', 'FMM', 'FBZ'}
        if tracer_id in standard_ids:
            raise ValueError(f"Cannot remove standard tracer: {tracer_id}")
        
        del cls._tracers[tracer_id]
        
        # Save custom tracers to configuration
        cls._save_custom_tracers()